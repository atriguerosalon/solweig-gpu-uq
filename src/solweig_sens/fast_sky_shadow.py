"""Remove per-step CPU-GPU syncs from the sky-patch shadow loop that builds the sky view factor (C7).

``solweig_gpu.shadow.shadow`` (v2.0.0) casts the shadows of one of the 153 sky patches by
stepping away from it, like the hourly shadow function C3 treats. Each step reads GPU values
back to the CPU about twelve times: the loop test on ``dz``, ``dx`` and ``dy``, eight ``int()``
slice bounds, and the bush test ``bush.max() > 0. and torch.max(fabovea * bush) > 0.``. Here
the step offsets and heights come from C3's ``_shadow_steps`` (same float32 operations, one
copy to the CPU; this loop starts at index 1), ``bush.max()`` is read once, and the per-step
bush test stays on the GPU as ``torch.where``. The array work is unchanged, so the masks are
bit-identical (tests/test_fast_sky_shadow.py).

Prior work: Isma A. Di Carlo's fork of SOLWEIG-GPU (https://github.com/iadicarlo/SOLWEIG-GPU,
commits 5932ef9 and 96e718e, 8 Oct 2026) already moves this loop's per-step bookkeeping to the
CPU, as part of changes that also alter results. This patch reaches the same end through C3's
step precomputation and leaves the results bit-identical.

Applied like C3, by compiling an edited copy of the function into a copy of the module
namespace, pinned to the SHA-256 of the source it was verified against. Enter it before
tight_bound() (which must stay innermost):

    with compact_masks("bits", svf=True), fast_sky_shadow(), tight_bound():
        prepare(setup)
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import textwrap
from typing import Iterator

from .fast_shadow import _shadow_steps

FINGERPRINT = "c0b9cb3745257f77482e225849de6f4b035177c7374c7e3d516c1226e8071690"
WHILE = "while (amaxvalue >= dz and torch.abs(dx) < sizex and torch.abs(dy) < sizey):"
DZ = "dz = ds * index * tanaltitudebyscale"
BUSH = "if bush.max() > 0. and torch.max(fabovea * bush) > 0.:"
BUSH_BODY = ["tempbush.zero_()",
             "tempbush[int(xp1):int(xp2), int(yp1):int(yp2)] = bush[int(xc1):int(xc2), int(yc1):int(yc2)] - dz",
             "g = torch.max(g, tempbush)",
             "g *= bushplant"]


def check_fingerprint() -> None:
    import solweig_gpu.shadow as shm

    digest = hashlib.sha256(inspect.getsource(shm.shadow).encode()).hexdigest()
    if digest != FINGERPRINT:
        raise RuntimeError("solweig_gpu.shadow.shadow differs from the verified version; "
                           "re-verify the fast replacement before using it")


def patched_source() -> str:
    import solweig_gpu.shadow as shm

    lines = textwrap.dedent(inspect.getsource(shm.shadow)).splitlines()
    find = lambda s, lo=0: [i for i in range(lo, len(lines)) if lines[i].strip() == s]  # noqa: E731
    w, z, b = find(WHILE), find(DZ), find(BUSH)
    if len(w) != 1 or len(z) != 1 or len(b) != 1 or not w[0] < z[0] < b[0]:
        raise RuntimeError("SOLWEIG source changed; cannot apply the sky-patch shadow patch")
    w, z, b = w[0], z[0], b[0]
    indent = lines[w][: len(lines[w]) - len(lines[w].lstrip())]
    body = indent + "    "

    # Loop header and the per-step dx, dy, ds, dz (lines w..z) become a precomputed step list.
    header = [l.strip() for l in lines[w + 1:z] if l.strip()]
    if not all(s.startswith(("if (pibyfour", "dy = ", "dx = ", "ds = ", "else:")) for s in header):
        raise RuntimeError("SOLWEIG source changed; unexpected lines in the sky-patch loop header")
    # The slice-bound block absdx ... yp2 goes; its values come with each step.
    a0 = [i for i in range(z, b) if lines[i].strip().startswith("absdx = ")]
    a1 = [i for i in range(z, b) if lines[i].strip().startswith("yp2 = ")]
    if len(a0) != 1 or len(a1) != 1:
        raise RuntimeError("SOLWEIG source changed; cannot find the slice-bound block")
    names = [l.strip().split(" =")[0] for l in lines[a0[0]:a1[0] + 1] if l.strip()]
    if names != ["absdx", "absdy", "xc1", "xc2", "yc1", "yc2", "xp1", "xp2", "yp1", "yp2"]:
        raise RuntimeError("SOLWEIG source changed; unexpected slice-bound lines")
    if [l.strip() for l in lines[b + 1:b + 1 + len(BUSH_BODY)]] != BUSH_BODY:
        raise RuntimeError("SOLWEIG source changed; unexpected bush block")
    bi = lines[b][: len(lines[b]) - len(lines[b].lstrip())]

    new_header = [
        f"{indent}_bush_any = bool(bush.max() > 0.)",
        f"{indent}_steps, _dzs = _shadow_steps(amaxvalue, azimuth, signsinazimuth, signcosazimuth, tanazimuth, dssin, dscos,",
        f"{indent}                             tanaltitudebyscale, sizex, sizey, pibyfour, threetimespibyfour,",
        f"{indent}                             fivetimespibyfour, seventimespibyfour, start=1)",
        f"{indent}for _k, (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in enumerate(_steps):",
        f"{body}dz = _dzs[_k]",
    ]
    new_bush = [
        f"{bi}if _bush_any:  # same update, chosen on the GPU instead of read back every step",
        f"{bi}    _hit = torch.max(fabovea * bush) > 0.",
        f"{bi}    tempbush.zero_()",
        f"{bi}    tempbush[xp1:xp2, yp1:yp2] = bush[xc1:xc2, yc1:yc2] - dz",
        f"{bi}    g = torch.where(_hit, torch.max(g, tempbush) * bushplant, g)",
    ]
    out = (lines[:w] + new_header + lines[z + 1:a0[0]] + lines[a1[0] + 1:b] + new_bush
           + lines[b + 1 + len(BUSH_BODY):])
    return "\n".join(out) + "\n"


def build():
    """The patched function, compiled against a copy of the module namespace."""
    import solweig_gpu.shadow as shm

    check_fingerprint()
    namespace = dict(vars(shm))
    namespace["_shadow_steps"] = _shadow_steps
    exec(compile(patched_source(), shm.__file__, "exec"), namespace)
    return namespace["shadow"]


@contextlib.contextmanager
def fast_sky_shadow() -> Iterator[None]:
    import solweig_gpu.shadow as shm

    old = shm.shadow
    shm.shadow = build()  # svf_calculator looks it up in the module at call time
    try:
        yield
    finally:
        shm.shadow = old
