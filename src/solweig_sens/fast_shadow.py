"""Remove per-step CPU-GPU syncs from SOLWEIG-GPU's shadow casting with vegetation.

``shadowingfunction_wallheight_23`` (``solweig_gpu/solweig.py``, v2.0.0) steps along the sun
direction until the shadow height exceeds the tallest object or the step leaves the tile.
The loop test and the eight ``int()`` slice bounds per step each read a GPU value back to
the CPU. The step offsets, shadow heights and the stopping step depend only on the sun
position, so here they are computed for all steps at once, with the same float32
operations and comparison types, and copied to the CPU once. The array work is unchanged,
so results should be bit-identical (checked by ``scripts/verify_fast.py``).

Applied by compiling an edited copy of the function into a copy of the module namespace;
the installed package is not modified. If the source differs from what this edit expects,
it refuses rather than guessing.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import textwrap
from typing import Iterator

import torch

WHILE = "while (amaxvalue >= dz) and (torch.abs(dx) < sizex) and (torch.abs(dy) < sizey):"
DZ = "dz = (ds * index) * tanaltitudebyscale"


def _shadow_steps(amaxvalue, azimuth, signsin, signcos, tanaz, dssin, dscos, tanaltbyscale,
                  sizex, sizey, p4, p34, p54, p74):
    """Slice bounds and shadow heights for every step the original loop would run."""
    device = tanaz.device
    cond = bool((((p4 <= azimuth) & (azimuth < p34)) | ((p54 <= azimuth) & (azimuth < p74))).item())
    ds = dssin if cond else dscos

    # The original tests the condition before each step with the previous step's values,
    # starting from dx = dy = dz = 0, so it runs steps 0..K where K is the first step that
    # fails the test. An upper bound on K: the main axis advances one pixel per step.
    kmax = max(sizex, sizey) + 2
    t, d, amax = float(tanaltbyscale), float(ds), float(amaxvalue)
    if t > 0 and d > 0 and math.isfinite(amax):
        kmax = min(kmax, int(math.ceil(max(amax, 0.0) / (d * t))) + 3)
    index = torch.arange(kmax, device=device, dtype=torch.float32)
    if cond:
        dy = signsin * index
        dx = -1 * signcos * torch.abs(torch.round(index / tanaz))
    else:
        dy = signsin * torch.abs(torch.round(index * tanaz))
        dx = -1 * signcos * index
    dz = (ds * index) * tanaltbyscale

    # Same comparison types as the original: a 0-d tensor against a 0-d tensor promotes,
    # a Python number is cast to the tensor dtype.
    if isinstance(amaxvalue, torch.Tensor):
        wide = torch.promote_types(amaxvalue.dtype, dz.dtype)
        height_ok = amaxvalue.to(wide) >= dz.to(wide)
    else:
        height_ok = amaxvalue >= dz
    ok = height_ok & (torch.abs(dx) < sizex) & (torch.abs(dy) < sizey)

    absdx, absdy = torch.abs(dx), torch.abs(dy)
    bounds = torch.stack([
        ((dx + absdx) / 2).int(), (sizex + (dx - absdx) / 2).int(),
        ((dy + absdy) / 2).int(), (sizey + (dy - absdy) / 2).int(),
        -(((dx - absdx) / 2).int()), (sizex - (dx + absdx) / 2).int(),
        -(((dy - absdy) / 2).int()), (sizey - (dy + absdy) / 2).int(),
    ], dim=1)
    ok_cpu, bounds_cpu = ok.cpu().tolist(), bounds.cpu().tolist()

    if not (amaxvalue >= 0):  # loop test fails before the first step
        return [], dz[:0]
    n = len(ok_cpu)
    for k, good in enumerate(ok_cpu):
        if not good:
            n = k + 1  # the failing step itself still runs
            break
    else:
        raise RuntimeError("shadow step bound too small; please report")
    return [tuple(b) for b in bounds_cpu[:n]], dz[:n]


def patched_source() -> str:
    import solweig_gpu.solweig as sw

    lines = textwrap.dedent(inspect.getsource(sw.shadowingfunction_wallheight_23)).splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip() == WHILE]
    dzs = [i for i, l in enumerate(lines) if l.strip() == DZ]
    if len(starts) != 1 or len(dzs) != 1 or not starts[0] < dzs[0]:
        raise RuntimeError("SOLWEIG source changed; cannot apply the shadow patch")
    w, z = starts[0], dzs[0]
    indent = lines[w][: len(lines[w]) - len(lines[w].lstrip())]
    body = indent + "    "

    # Block 1: the loop header and the per-step dx, dy, ds, dz computation (lines w..z).
    header_block = [l.strip() for l in lines[w + 1:z] if l.strip()]
    if not all(s.startswith(("if ((pibyfour", "dy = ", "dx = ", "ds = ", "else:")) for s in header_block):
        raise RuntimeError("SOLWEIG source changed; unexpected lines in the shadow loop header")

    # Block 2: the slice-bound computation, absdx = ... through yp2 = ..., after block 1.
    a0 = [i for i in range(z, len(lines)) if lines[i].strip().startswith("absdx = ")]
    a1 = [i for i in range(z, len(lines)) if lines[i].strip().startswith("yp2 = ")]
    if len(a0) != 1 or len(a1) != 1 or not a0[0] < a1[0]:
        raise RuntimeError("SOLWEIG source changed; cannot find the slice-bound block")
    bounds_block = [l.strip() for l in lines[a0[0]:a1[0] + 1] if l.strip()]
    expected = ("absdx", "absdy", "xc1", "xc2", "yc1", "yc2", "xp1", "xp2", "yp1", "yp2")
    if [s.split(" =")[0] for s in bounds_block] != list(expected):
        raise RuntimeError("SOLWEIG source changed; unexpected slice-bound lines")

    new_header = [
        f"{indent}_steps, _dzs = _shadow_steps(amaxvalue, azimuth, signsinazimuth, signcosazimuth, tanazimuth, dssin, dscos,",
        f"{indent}                             tanaltitudebyscale, sizex, sizey, pibyfour, threetimespibyfour,",
        f"{indent}                             fivetimespibyfour, seventimespibyfour)",
        f"{indent}for _k, (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in enumerate(_steps):",
        f"{body}dz = _dzs[_k]",
    ]
    out = lines[:w] + new_header + lines[z + 1:a0[0]] + lines[a1[0] + 1:]
    return "\n".join(out) + "\n"


def build():
    """The patched function, compiled against a copy of the module namespace."""
    import solweig_gpu.solweig as sw

    namespace = dict(vars(sw))
    namespace["_shadow_steps"] = _shadow_steps
    exec(compile(patched_source(), sw.__file__, "exec"), namespace)
    return namespace["shadowingfunction_wallheight_23"]


@contextlib.contextmanager
def fast_shadow() -> Iterator[None]:
    import solweig_gpu.solweig as sw

    old = sw.shadowingfunction_wallheight_23
    sw.shadowingfunction_wallheight_23 = build()
    try:
        yield
    finally:
        sw.shadowingfunction_wallheight_23 = old
