"""Remove per-step CPU-GPU syncs from SOLWEIG-GPU's ground view factor loop.

``sunonsurface_2018a`` (``solweig_gpu/solweig.py``, v2.0.0) steps along one direction, and at
every step converts GPU scalars to Python integers for slicing (eight ``.int()`` calls) and
compares GPU scalars in ``if`` statements: each is a sync. The step offsets depend only on
the azimuth, so here they are computed for all steps at once, with the same float32
operations on the same device, and copied to the CPU once. The array work is unchanged, so
results should be bit-identical.

Applied by compiling an edited copy of the function into a copy of the module namespace;
the installed package is not modified.
"""

from __future__ import annotations

import contextlib
import inspect
import textwrap
from typing import Iterator

import torch

OLD_LOOP = """    for n in torch.arange(0, second, device=device):
        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index

        absdx = torch.abs(dx)
        absdy = torch.abs(dy)

        xc1 = ((dx + absdx) / 2).int()
        xc2 = (sizex + (dx - absdx) / 2).int()
        yc1 = ((dy + absdy) / 2).int()
        yc2 = (sizey + (dy - absdy) / 2).int()

        xp1 = -((dx - absdx) / 2).int()
        xp2 = (sizex - (dx + absdx) / 2).int()
        yp1 = -((dy - absdy) / 2).int()
        yp2 = (sizey - (dy + absdy) / 2).int()
"""

NEW_LOOP = """    _offsets = _gvf_offsets(azimuth, signsinazimuth, signcosazimuth, tanazimuth, second, sizex, sizey,
                            pibyfour, threetimespibyfour, fivetimespibyfour, seventimespibyfour)
    _first = float(first)
    for n, (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in enumerate(_offsets):
"""

OLD_FIRST = "        if (n + 1) <= first:"
NEW_FIRST = "        if (n + 1) <= _first:"


def _gvf_offsets(azimuth, signsin, signcos, tanaz, second, sizex, sizey, p4, p34, p54, p74):
    """Slice bounds for every step, computed exactly as the original loop does, in one go."""
    n = int(second)
    index = torch.arange(n, device=tanaz.device, dtype=torch.float32)
    cond = ((p4 <= azimuth) & (azimuth < p34)) | ((p54 <= azimuth) & (azimuth < p74))
    dy = torch.where(cond, signsin * index, signsin * torch.abs(torch.round(index * tanaz)))
    dx = torch.where(cond, -1 * signcos * torch.abs(torch.round(index / tanaz)), -1 * signcos * index)
    absdx, absdy = torch.abs(dx), torch.abs(dy)
    b = torch.stack([
        ((dx + absdx) / 2).int(), (sizex + (dx - absdx) / 2).int(),
        ((dy + absdy) / 2).int(), (sizey + (dy - absdy) / 2).int(),
        -((dx - absdx) / 2).int(), (sizex - (dx + absdx) / 2).int(),
        -((dy - absdy) / 2).int(), (sizey - (dy + absdy) / 2).int(),
    ], dim=1)
    return [tuple(row) for row in b.cpu().tolist()]


def patched_source() -> str:
    import solweig_gpu.solweig as sw

    src = textwrap.dedent(inspect.getsource(sw.sunonsurface_2018a))
    for needle in (OLD_LOOP, OLD_FIRST):
        if src.count(needle) != 1:
            raise RuntimeError("SOLWEIG source changed; cannot apply the ground-view-factor patch")
    return src.replace(OLD_LOOP, NEW_LOOP).replace(OLD_FIRST, NEW_FIRST)


def build():
    """The patched function, compiled against a copy of the module namespace."""
    import solweig_gpu.solweig as sw

    namespace = dict(vars(sw))
    namespace["_gvf_offsets"] = _gvf_offsets
    exec(compile(patched_source(), sw.__file__, "exec"), namespace)
    return namespace["sunonsurface_2018a"]


@contextlib.contextmanager
def fast_gvf() -> Iterator[None]:
    import solweig_gpu.solweig as sw

    old = sw.sunonsurface_2018a
    sw.sunonsurface_2018a = build()  # gvf_2018a looks it up at call time
    try:
        yield
    finally:
        sw.sunonsurface_2018a = old
