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

Applied by compiling edited copies of ``load_cached_svf_outputs`` and ``compute_utci`` into
the live ``solweig_gpu.utci_process`` namespace and restoring the originals afterwards.
Both are fingerprint-pinned. Verification: tests/test_compact_masks.py and
scripts/verify_compact.py.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import textwrap
import warnings
from typing import Iterator

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


def compact_mask(arr: np.ndarray, device, mode: str):
    """Convert one mask from the SVF cache; falls back to float32 if it is not purely 0/1."""
    if not np.isin(arr, (0.0, 1.0)).all():
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


def patched_sources(mode: str) -> dict:
    import solweig_gpu.utci_process as up

    for name in FINGERPRINTS:
        _check(name)
    load = textwrap.dedent(inspect.getsource(up.load_cached_svf_outputs))
    for key in _LOAD_KEYS:
        old = f'torch.tensor(npz["{key}"].astype(np.float32), device=device)'
        if load.count(old) != 1:
            raise RuntimeError("cannot patch load_cached_svf_outputs")
        load = load.replace(old, f'_compact_mask(npz["{key}"], device, "{mode}")')
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
def compact_masks(mode: str = "bits", svf: bool = True) -> Iterator[None]:
    """Run SOLWEIG-GPU with compact masks (``"uint8"`` or ``"bits"``) and on-the-fly diffsh.

    With ``svf=True`` (default) the sky view factor computation also builds the masks compactly,
    so computing a new tile's geometry no longer needs the float32 masks either. The cache file
    it writes is unchanged.
    """
    import solweig_gpu.shadow as shadow_mod
    import solweig_gpu.utci_process as up

    sources = patched_sources(mode)
    svf_src = patched_svf_source(mode) if svf else None
    ns = vars(up)  # compile into the live namespace so the patched functions see current globals
    originals = {name: getattr(up, name) for name in sources}
    original_svf = shadow_mod.svf_calculator
    original_up_svf = up.svf_calculator
    ns["_compact_mask"], ns["_LazyDiffsh"] = compact_mask, LazyDiffsh
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
        sns.pop("_MaskBuilder", None)
