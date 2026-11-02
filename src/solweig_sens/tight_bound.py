"""Stop the shadow ray loops at the tile's height range instead of its absolute elevation.

SOLWEIG-GPU v2.0.0 casts shadows by stepping a ray away from the sun until it has dropped
``amaxvalue``. Both callers set it to the highest surface in absolute metres:
``svf_calculator`` (sky-patch masks, ``shadow.shadow``) and ``compute_utci`` (hourly shadows,
``shadowingfunction_wallheight_23``). In Pacoima the ground is at 304-374 m, so the loop runs to
380 m although nothing can shade a cell from more than (highest surface - lowest surface) =
76 m above it. Beyond that drop every comparison in the loop is false, so the extra steps
change nothing.

``tight_bound()`` passes min(amaxvalue, max(DSM, canopy) - min(DSM)) instead. Results are
bit-identical (tests/test_tight_bound.py; on the Pacoima tile, all SVF rasters, sky-patch
masks and hourly Shadow, Tmrt and UTCI, three repeats). Sky view factor 106 -> 43 s, day
21.9 -> 19.8 s on an RTX 4070 laptop, with fast functions and compact masks. The saving grows
with the city's elevation and is nil at sea level.

Why it is exact, in more detail. A skipped step has dropped more than the height range, so for
every source cell q and target cell p, ``a[q] - dz`` and ``vegdem[q] - dz`` are below ``a[p]``:
``f = max(f, temp)`` keeps f (f starts at a), and the strict tests ``> a`` (sh, fabovea,
gabovea and the "last" variants) are all false, so vegsh and sh do not change. ``shvoveg`` is a
maximum, not a test: it only matters where vegsh > 0, and there it already holds a value no
more than one step's drop below ``a[p]``; the loop always runs the first step past the bound,
so every skipped step's value is lower still. ``vbshvegsh`` keeps counting, but is turned into
0/1 at the end, and a cell can only be non-zero there if it was before the skipped steps. Ties
cannot flip a strict comparison, and the margin, one step's drop (about 1 cm even with the sun
0.5 degrees above the horizon on 1 m pixels), is far above float32 rounding at these heights
(0.00003 m at 400 m).

Credit: the bound comes from Isma A. Di Carlo's fork of SOLWEIG-GPU
(https://github.com/iadicarlo/SOLWEIG-GPU, commit 5932ef9, 8 Oct 2026), where it is part of a
change that also alters results (sky-patch azimuths and vegetation shadows). Here it is applied
alone to v2.0.0, where it changes nothing.

It wraps the functions as they are installed when the block is entered, so enter it inside
``fast_solweig()`` (innermost), or the fast shadow function replaces the wrapper.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch


def tight(amaxvalue, a: torch.Tensor, vegdem: torch.Tensor) -> torch.Tensor:
    """The loop bound capped at the height range of buildings, ground and canopy."""
    h = torch.maximum(a.max(), vegdem.max()) - a.min()
    return torch.minimum(torch.as_tensor(amaxvalue, device=a.device, dtype=h.dtype), h)


@contextlib.contextmanager
def tight_bound() -> Iterator[None]:
    import solweig_gpu.shadow as shm
    import solweig_gpu.solweig as sw

    sky, hourly = shm.shadow, sw.shadowingfunction_wallheight_23

    def shadow(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale):
        return sky(tight(amaxvalue, a, vegdem), a, vegdem, vegdem2, bush, azimuth, altitude, scale)

    def shadowingfunction_wallheight_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls,
                                        aspect):
        return hourly(a, vegdem, vegdem2, azimuth, altitude, scale, tight(amaxvalue, a, vegdem), bush, walls,
                      aspect)

    # callers look both names up in their module at call time (svf_calculator, Solweig_2022a_calc)
    shm.shadow, sw.shadowingfunction_wallheight_23 = shadow, shadowingfunction_wallheight_23
    try:
        yield
    finally:
        shm.shadow, sw.shadowingfunction_wallheight_23 = sky, hourly
