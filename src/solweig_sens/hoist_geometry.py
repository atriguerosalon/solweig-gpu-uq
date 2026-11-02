"""Compute the sky-patch geometry once per run instead of every hour (C8).

Three pieces of SOLWEIG-GPU v2.0.0 rebuild the same sky-patch geometry every hour:

- ``shadow.create_patches(patch_option)`` builds the patch altitudes and azimuths with two
  ``torch.cat`` per patch (306 small GPU operations and as many copies);
- ``solweig.Lcyl_v2022a`` and ``solweig.Kside_veg_v2022a`` compute each patch's solid angle
  (``steradian``) in a loop of 153 iterations, each reading a GPU value back to the CPU
  (``if skyalt_c[skyalt == patch_altitude[i]] > 1``).

All three depend only on the patch layout, which is the same every hour. Here
``create_patches`` keeps its first result per ``patch_option``, and the two steradian loops keep
theirs per patch-altitude vector; later calls get copies. The first call runs the original code,
so every later value is the same bits. In ``Kside_veg_v2022a`` the loop also accumulates
``radTot``, which depends on the hour: that accumulation stays, in the same order, in a loop of
its own.

Measured cost removed (scripts/where_time_goes.py, 2 Nov): ``create_patches`` ~0.7 s and most of
``Lcyl_v2022a``'s ~1.5 s per day, at every tile size.

Applied like compact_masks: edited copies of the two functions are compiled into the live
``solweig_gpu.solweig`` namespace (so they see the current globals, including other patches),
``create_patches`` is wrapped where its name is bound, and everything is restored on exit. The
three functions are pinned to the SHA-256 of the source verified against. Verification:
tests/test_hoist_geometry.py.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import textwrap
from typing import Iterator

FINGERPRINTS = {
    "Lcyl_v2022a": "b04719be58197cf115166adceda47cda377bac461bfc970f560c883b0bdb30ef",
    "Kside_veg_v2022a": "27d554a94a79fb31f6e139298b84922dd4ba406082edf043290cb306b4ea8269",
    "create_patches": "61016f2e8125115d52222e71401b1387bf88f64c9735729e7d58481a132fdec2",  # shadow.py
}

_LCYL_START = "steradian = torch.zeros(patch_altitude.shape[0], device=device)"
_LCYL_END = "# True = anisotropic sky, False = isotropic sky"
_KSIDE_START = "steradian = torch.zeros((patch_altitude.shape[0]), device=device)"
_KSIDE_RADTOT = "radTot += (patch_luminance[i] * steradian[i] * torch.sin(patch_altitude[i] * deg2rad))"
_LOOP = "for i in range(patch_altitude.shape[0]):"


def _source(name: str) -> str:
    import solweig_gpu.shadow as sh
    import solweig_gpu.solweig as sw

    return inspect.getsource(getattr(sh if name == "create_patches" else sw, name))


def check_fingerprints() -> None:
    for name, digest in FINGERPRINTS.items():
        if hashlib.sha256(_source(name).encode()).hexdigest() != digest:
            raise RuntimeError(f"solweig_gpu {name} differs from the verified version; re-verify first")


def _indent(lines: list[str], n: int = 4) -> list[str]:
    return [" " * n + l if l.strip() else l for l in lines]


def _find(lines: list[str], text: str, lo: int = 0) -> int:
    hits = [i for i in range(lo, len(lines)) if lines[i].strip() == text]
    if len(hits) != 1:
        raise RuntimeError(f"SOLWEIG source changed; cannot find {text!r}")
    return hits[0]


def _cached(block: list[str], tag: str) -> list[str]:
    """Wrap the steradian block: run it on the first call per patch layout, reuse a copy after."""
    ind = block[0][: len(block[0]) - len(block[0].lstrip())]
    return ([f'{ind}_key = ("{tag}", str(device), patch_altitude.cpu().numpy().tobytes())',
             f"{ind}if _key in _GEOMETRY_CACHE:",
             f"{ind}    steradian = _GEOMETRY_CACHE[_key].clone()",
             f"{ind}else:"]
            + _indent(block)
            + [f"{ind}    _GEOMETRY_CACHE[_key] = steradian.clone()"])


def patched_lcyl() -> str:
    lines = textwrap.dedent(_source("Lcyl_v2022a")).splitlines()
    a = _find(lines, _LCYL_START)
    b = _find(lines, _LCYL_END, a)
    while not lines[b - 1].strip():
        b -= 1
    return "\n".join(lines[:a] + _cached(lines[a:b], "Lcyl") + lines[b:]) + "\n"


def patched_kside() -> str:
    lines = textwrap.dedent(_source("Kside_veg_v2022a")).splitlines()
    a = _find(lines, _KSIDE_START)
    loop = _find(lines, _LOOP, a)
    r = _find(lines, _KSIDE_RADTOT, loop)
    if loop != a + 1:
        raise RuntimeError("SOLWEIG source changed; unexpected lines in Kside's steradian block")
    b = r
    while not lines[b - 1].strip():
        b -= 1
    # radTot's accumulation leaves the steradian loop: same terms, same order, own loop.
    radtot = [lines[loop], lines[r]]
    return "\n".join(lines[:a] + _cached(lines[a:b], "Kside") + radtot + lines[r + 1:]) + "\n"


def _memo_create_patches(original):
    import torch

    cache = {}

    @functools.wraps(original)
    def create_patches(patch_option):
        if patch_option not in cache:
            cache[patch_option] = original(patch_option)
        return tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in cache[patch_option])

    return create_patches


@contextlib.contextmanager
def hoist_geometry() -> Iterator[None]:
    """Run SOLWEIG-GPU with the sky-patch geometry computed once per run."""
    import solweig_gpu.shadow as sh
    import solweig_gpu.solweig as sw
    import solweig_gpu.utci_process as up

    check_fingerprints()
    sources = {"Lcyl_v2022a": patched_lcyl(), "Kside_veg_v2022a": patched_kside()}
    ns = vars(sw)
    originals = {name: getattr(sw, name) for name in sources}
    bindings = [(m, m.create_patches) for m in (sh, sw, up)]
    memo = _memo_create_patches(sh.create_patches)
    ns["_GEOMETRY_CACHE"] = {}
    try:
        for src in sources.values():
            exec(compile(src, sw.__file__, "exec"), ns)
        for m, _ in bindings:
            m.create_patches = memo
        yield
    finally:
        for name, f in originals.items():
            setattr(sw, name, f)
        for m, f in bindings:
            m.create_patches = f
        ns.pop("_GEOMETRY_CACHE", None)
