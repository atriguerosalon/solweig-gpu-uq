"""Store SOLWEIG-GPU's sky-patch masks compactly, and compute ``diffsh`` on the fly.

In SOLWEIG-GPU v2.0.0 most GPU memory is static: three sky-patch visibility masks
(``shmat``, ``vegshmat``, ``vbshvegshmat``; rows x cols x 153) loaded as float32, plus a
derived float32 array ``diffsh = shmat - (1 - vegshmat) * (1 - transVeg)`` of the same size.
On a 1000 x 1000 tile that is about 2.3 GiB of a 2.9 GiB peak. The masks hold only 0 and 1.

Modes:
- ``"bits"`` (default, recommended): masks packed 8 patches per byte along the patch axis (about
  30x smaller); ``PackedMask`` unpacks the requested patches, as float32, when they are indexed,
  so code using them sees exactly the type and values of the original arrays.
- ``"uint8"``: masks as plain byte tensors (4x smaller). Exact for every use in SOLWEIG-GPU
  v2.0.0 (only ``== 0``, ``== 1``, ``1 - mask`` and ``mask * float``), but byte arithmetic that
  goes negative (``mask - 1``) would wrap around silently, so new code using it must be checked.
In both modes ``diffsh`` is not stored: ``LazyDiffsh`` computes the requested patch with
the original float32 arithmetic when it is indexed, so values are the same bits.

If a mask file holds any value other than 0 or 1, the loader keeps that mask as float32
(the original behaviour) rather than corrupting it.

Loading the cache: SOLWEIG-GPU stores the three masks as compressed float32 (1.8 GB of 0/1 values
at 1000 x 1000, 7 GB at 2000 x 2000), and decompressing them is most of each run's start-up
(3.6 of 4.2 s at 1000 px, scripts/where_time_goes.py). The three are now decompressed in parallel
threads (about 2x) when they fit in memory (see MaskFile), each packed as soon as it is decompressed. With ``packed_cache=True`` (bits mode only) the first load also writes a
bit-packed copy next to SOLWEIG's file, ``shadowmats_<n>.bits.npz``, tagged with the SHA-256 of
the file it came from; later loads read that copy (about 30x less to decompress) as long as the
tag matches, and rebuild it otherwise. SOLWEIG's own cache file is never changed.

Applied by compiling edited copies of ``load_cached_svf_outputs`` and ``compute_utci`` into
the live ``solweig_gpu.utci_process`` namespace and restoring the originals afterwards.
Both are fingerprint-pinned. Verification: tests/test_compact_masks.py and
scripts/verify_compact.py.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import textwrap
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, NamedTuple

import numpy as np
import torch

FINGERPRINTS = {
    "load_cached_svf_outputs": "95e5d104f221e73107f5060ee00f85f68859740582dae0db5cca290d58628090",
    "compute_utci": "07ec3db27a2b78cd3cf3a17f64141bedeee7c327bd44a8f382a8698a5fdeaa61",
}
SVF_FINGERPRINT = "406e20d2b6807cd34bef3244f86185c8f339ca6886e52303b61cbcecb6feca2b"  # shadow.svf_calculator
_SVF_EDITS = [
    ("    shmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)",
     '    shmat = _MaskBuilder(rows, cols, int(sum(aziinterval)), device, "{mode}")'),
    ("    vegshmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)",
     '    vegshmat = _MaskBuilder(rows, cols, int(sum(aziinterval)), device, "{mode}")'),
    ("    vbshvegshmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)",
     '    vbshvegshmat = _MaskBuilder(rows, cols, int(sum(aziinterval)), device, "{mode}")'),
    # The cache file is written exactly as before (float32 0/1 arrays), from a CPU copy.
    ("shmat=shmat, vegshmat=vegshmat, vbshvegshmat=vbshvegshmat,",
     "shmat=shmat.saveable(), vegshmat=vegshmat.saveable(), vbshvegshmat=vbshvegshmat.saveable(),"),
    ("vegshmat, vbshvegshmat, shmat, SVFtotal", "vegshmat.final(), vbshvegshmat.final(), shmat.final(), SVFtotal"),
]
_DIFFSH_OLD = (
    "    diffsh = torch.zeros((rows, cols, shmat.shape[2]), device=device)\n"
    "    for i in range(shmat.shape[2]):\n"
    "        diffsh[:, :, i] = shmat[:, :, i] - (1 - vegshmat[:, :, i]) * (1 - transVeg)\n"
)
_DIFFSH_NEW = "    diffsh = _LazyDiffsh(shmat, vegshmat, transVeg)\n"
_LOAD_KEYS = ("shadowmat", "vegshadowmat", "vbshmat")
_LOAD_OPEN = "np.load(npz_path)"


class PackedMask:
    """A 0/1 mask (rows, cols, n) stored as bits; indexing returns float32 0/1 tensors.

    Indexing returns float32, the type the original arrays had, so any arithmetic on the result
    behaves exactly as in the original (with uint8, ``mask - 1`` would silently wrap to 255). Only
    the requested patches are expanded, so the extra memory is one patch or one chunk.
    """

    def __init__(self, packed: torch.Tensor, n: int):
        self.packed = packed  # (rows, cols, ceil(n / 8)) uint8, bit i of byte j is patch 8j + i
        self.n = n

    @classmethod
    def from_numpy(cls, arr: np.ndarray, device) -> "PackedMask":
        packed = np.packbits(arr.astype(np.uint8), axis=2, bitorder="little")
        return cls(torch.from_numpy(packed).to(device), arr.shape[2])

    @property
    def shape(self):
        return (*self.packed.shape[:2], self.n)

    @property
    def device(self):
        return self.packed.device

    @property
    def dtype(self):
        return torch.float32

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == slice(None) and key[1] == slice(None)):
            raise IndexError("PackedMask supports only [:, :, patch] and [:, :, slice] indexing")
        idx = key[2]
        if isinstance(idx, torch.Tensor):
            idx = int(idx)
        if isinstance(idx, (int, np.integer)):
            i = int(idx) % self.n
            bits = (self.packed[:, :, i >> 3] >> (i & 7)) & 1
        elif isinstance(idx, slice):
            ids = torch.arange(self.n, device=self.packed.device)[idx]
            bits = (self.packed[:, :, ids >> 3] >> (ids & 7).to(torch.uint8)) & 1
        else:
            raise IndexError(f"unsupported patch index {idx!r}")
        return bits.to(torch.float32)

    def nbytes(self) -> int:
        return self.packed.numel()


class LazyDiffsh:
    """diffsh[:, :, i] = shmat[:, :, i] - (1 - vegshmat[:, :, i]) * (1 - transVeg), on demand."""

    def __init__(self, shmat, vegshmat, transVeg: float):
        self.shmat, self.vegshmat, self.transVeg = shmat, vegshmat, transVeg

    @property
    def shape(self):
        return tuple(self.shmat.shape)

    @property
    def device(self):
        return self.shmat.device

    def _f(self, x):
        return x if x.dtype == torch.float32 else x.to(torch.float32)

    def __getitem__(self, key):
        # Same operations, in float32, as the original's precomputed array.
        return self._f(self.shmat[key]) - (1 - self._f(self.vegshmat[key])) * (1 - self.transVeg)


class MaskBuilder:
    """Collects one 0/1 mask per sky patch, as svf_calculator computes them, without a float32 array.

    Each patch is checked as it is stored; if any patch holds a value other than 0 or 1, the
    builder switches to a float32 array (the original representation) and keeps it.
    """

    def __init__(self, rows: int, cols: int, n: int, device, mode: str):
        self.rows, self.cols, self.n, self.device, self.mode = rows, cols, n, device, mode
        if mode == "bits":
            self.data = torch.zeros((rows, cols, (n + 7) // 8), dtype=torch.uint8, device=device)
        elif mode == "uint8":
            self.data = torch.zeros((rows, cols, n), dtype=torch.uint8, device=device)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.float_data = None

    @property
    def shape(self):
        return (self.rows, self.cols, self.n)

    def _expanded(self) -> torch.Tensor:
        if self.mode == "bits":
            return PackedMask(self.data, self.n)[:, :, :]
        return self.data

    def __setitem__(self, key, value):
        idx = int(key[2])
        if self.float_data is None and not bool(((value == 0) | (value == 1)).all()):
            warnings.warn(f"sky-patch mask holds values other than 0 and 1 (patch {idx}); "
                          "keeping it as float32 (original representation, more memory)", RuntimeWarning, stacklevel=2)
            self.float_data = self._expanded().to(torch.float32)
            self.data = None
        if self.float_data is not None:
            self.float_data[:, :, idx] = value
            return
        b = value.to(torch.uint8)
        if self.mode == "bits":
            self.data[:, :, idx >> 3] |= b << (idx & 7)
        else:
            self.data[:, :, idx] = b

    def final(self):
        """What svf_calculator returns: compact if every patch was 0/1, else float32."""
        if self.float_data is not None:
            return self.float_data
        return PackedMask(self.data, self.n) if self.mode == "bits" else self.data

    def saveable(self) -> torch.Tensor:
        """A CPU tensor with the full (rows, cols, n) values, for writing the cache file."""
        if self.float_data is not None:
            return self.float_data.cpu()
        if self.mode == "bits":
            arr = np.unpackbits(self.data.cpu().numpy(), axis=2, count=self.n, bitorder="little")
            return torch.from_numpy(arr)
        return self.data.cpu()


class Packed(NamedTuple):
    """A 0/1 mask already packed on the CPU: bits as in PackedMask, n patches."""
    packed: np.ndarray
    n: int


def _is_binary(arr: np.ndarray) -> bool:
    return bool(np.isin(arr, (0.0, 1.0)).all())


def _pack(arr: np.ndarray) -> Packed:
    return Packed(np.packbits(arr.astype(np.uint8), axis=2, bitorder="little"), arr.shape[2])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _available_bytes():
    """Memory available to this process (Linux: MemAvailable), or None if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


class MaskFile:
    """Stands in for ``np.load(npz_path)`` on SOLWEIG's mask cache (only ``[key]`` is used).

    Decompresses the three masks in parallel when they fit in memory, and packs each one as soon
    as it is decompressed, so at most one float32 mask per thread is held at a time (as the
    original loader, which converts them one after another). In bits mode, binary masks come
    back as ``Packed`` (compact_mask turns them into a PackedMask without repacking), in uint8
    mode as uint8 arrays; with ``packed_cache``, a bit-packed copy is read if its source tag
    matches, or written after a full load.

    Memory guard: three float32 masks are 1.8 GB at 1000 x 1000 and 7.3 GB at 2000 x 2000. They
    are decompressed in parallel only if three of them, plus the packing temporaries, take at most
    half of the available memory; otherwise one after another.
    """

    PEAK_PER_ARRAY = 1.25  # the float32 array plus its uint8 copy while packing
    MEMORY_SHARE = 0.5

    def __init__(self, path, mode: str, packed_cache: bool = False):
        self.path, self.mode = str(path), mode
        self.packed_cache = packed_cache and mode == "bits"
        stem = self.path[: -len(".npz")] if self.path.endswith(".npz") else self.path
        self.sidecar = stem + ".bits.npz"
        self.arrays = None
        self.workers = None

    def _load(self, key: str):
        with np.load(self.path) as npz:  # one handle per thread
            a = npz[key]
        if not _is_binary(a):
            return a  # compact_mask keeps it as float32, with a warning
        if self.mode == "bits":
            return _pack(a)
        return a.astype(np.uint8) if self.mode == "uint8" else a

    def _parallel_workers(self) -> int:
        """3 if three decompressed masks fit in half of the available memory, else 1."""
        import zipfile

        with zipfile.ZipFile(self.path) as z:
            sizes = [z.getinfo(f"{k}.npy").file_size for k in _LOAD_KEYS]
        available = _available_bytes()
        n = len(_LOAD_KEYS)
        if available is None or n * self.PEAK_PER_ARRAY * max(sizes) > self.MEMORY_SHARE * available:
            return 1
        return n

    def _read_sidecar(self, digest: str):
        try:
            with np.load(self.sidecar) as z:
                if str(z["source_sha256"]) != digest:
                    return None
                return {k: Packed(z[k], int(z[k + "_n"])) for k in _LOAD_KEYS}
        except (OSError, ValueError, KeyError):
            return None

    def _write_sidecar(self, digest: str) -> None:
        tmp = self.sidecar + ".tmp"
        try:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, source_sha256=np.array(digest),
                                    **{k: v.packed for k, v in self.arrays.items()},
                                    **{k + "_n": np.array(v.n) for k, v in self.arrays.items()})
            os.replace(tmp, self.sidecar)
        except OSError as e:
            warnings.warn(f"could not write the packed mask cache {self.sidecar}: {e}", RuntimeWarning, stacklevel=2)
            with contextlib.suppress(OSError):
                os.remove(tmp)

    def __enter__(self):
        digest = _sha256(self.path) if self.packed_cache else None
        if self.packed_cache and os.path.exists(self.sidecar):
            self.arrays = self._read_sidecar(digest)
            if self.arrays is not None:
                return self
        self.workers = self._parallel_workers()
        with ThreadPoolExecutor(self.workers) as ex:  # zlib releases the GIL while it inflates
            self.arrays = dict(zip(_LOAD_KEYS, ex.map(self._load, _LOAD_KEYS)))
        if self.packed_cache and all(isinstance(v, Packed) for v in self.arrays.values()):
            self._write_sidecar(digest)
        return self

    def __getitem__(self, key: str):
        return self.arrays[key]

    def __exit__(self, *exc):
        self.arrays = None
        return False


def compact_mask(arr, device, mode: str):
    """Convert one mask from the SVF cache; falls back to float32 if it is not purely 0/1."""
    if isinstance(arr, Packed):  # already checked and packed by MaskFile
        return PackedMask(torch.from_numpy(arr.packed).to(device), arr.n)
    if not _is_binary(arr):
        warnings.warn("cached sky-patch mask holds values other than 0 and 1; loading it as float32 "
                      "(original representation, more memory)", RuntimeWarning, stacklevel=2)
        return torch.tensor(arr.astype(np.float32), device=device)
    if mode == "bits":
        return PackedMask.from_numpy(arr, device)
    if mode == "uint8":
        return torch.from_numpy(arr.astype(np.uint8)).to(device)
    raise ValueError(f"unknown mode {mode!r}")


def _check(name: str) -> None:
    import solweig_gpu.utci_process as up

    digest = hashlib.sha256(inspect.getsource(getattr(up, name)).encode()).hexdigest()
    if digest != FINGERPRINTS[name]:
        raise RuntimeError(f"solweig_gpu.utci_process.{name} differs from the verified version; re-verify first")


def patched_sources(mode: str, packed_cache: bool = False) -> dict:
    import solweig_gpu.utci_process as up

    for name in FINGERPRINTS:
        _check(name)
    load = textwrap.dedent(inspect.getsource(up.load_cached_svf_outputs))
    for key in _LOAD_KEYS:
        old = f'torch.tensor(npz["{key}"].astype(np.float32), device=device)'
        if load.count(old) != 1:
            raise RuntimeError("cannot patch load_cached_svf_outputs")
        load = load.replace(old, f'_compact_mask(npz["{key}"], device, "{mode}")')
    if load.count(_LOAD_OPEN) != 1:
        raise RuntimeError("cannot patch load_cached_svf_outputs")
    load = load.replace(_LOAD_OPEN, f'_MaskFile(npz_path, "{mode}", {bool(packed_cache)})')
    comp = textwrap.dedent(inspect.getsource(up.compute_utci))
    if comp.count(_DIFFSH_OLD) != 1:
        raise RuntimeError("cannot patch compute_utci")
    comp = comp.replace(_DIFFSH_OLD, _DIFFSH_NEW)
    return {"load_cached_svf_outputs": load, "compute_utci": comp}


def patched_svf_source(mode: str) -> str:
    import solweig_gpu.shadow as sh

    src = textwrap.dedent(inspect.getsource(sh.svf_calculator))
    if hashlib.sha256(inspect.getsource(sh.svf_calculator).encode()).hexdigest() != SVF_FINGERPRINT:
        raise RuntimeError("solweig_gpu.shadow.svf_calculator differs from the verified version; re-verify first")
    for old, new in _SVF_EDITS:
        if src.count(old) != 1:
            raise RuntimeError(f"cannot patch svf_calculator: {old!r}")
        src = src.replace(old, new.format(mode=mode))
    return src


@contextlib.contextmanager
def compact_masks(mode: str = "bits", svf: bool = True, packed_cache: bool = False) -> Iterator[None]:
    """Run SOLWEIG-GPU with compact masks (``"uint8"`` or ``"bits"``) and on-the-fly diffsh.

    With ``svf=True`` (default) the sky view factor computation also builds the masks compactly,
    so computing a new tile's geometry no longer needs the float32 masks either. The cache file
    it writes is unchanged. With ``packed_cache=True`` (bits only) loading keeps a bit-packed copy
    of the mask cache next to it and reads that on later runs (see the module docstring).
    """
    import solweig_gpu.shadow as shadow_mod
    import solweig_gpu.utci_process as up

    sources = patched_sources(mode, packed_cache)
    svf_src = patched_svf_source(mode) if svf else None
    ns = vars(up)  # compile into the live namespace so the patched functions see current globals
    originals = {name: getattr(up, name) for name in sources}
    original_svf = shadow_mod.svf_calculator
    original_up_svf = up.svf_calculator
    ns["_compact_mask"], ns["_LazyDiffsh"], ns["_MaskFile"] = compact_mask, LazyDiffsh, MaskFile
    sns = vars(shadow_mod)
    try:
        for src in sources.values():
            exec(compile(src, up.__file__, "exec"), ns)
        if svf_src is not None:
            sns["_MaskBuilder"] = MaskBuilder
            exec(compile(svf_src, shadow_mod.__file__, "exec"), sns)
            up.svf_calculator = shadow_mod.svf_calculator  # utci_process imported the name at load time
        yield
    finally:
        for name, f in originals.items():
            setattr(up, name, f)
        shadow_mod.svf_calculator = original_svf
        up.svf_calculator = original_up_svf
        ns.pop("_compact_mask", None)
        ns.pop("_LazyDiffsh", None)
        ns.pop("_MaskFile", None)
        sns.pop("_MaskBuilder", None)
