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
