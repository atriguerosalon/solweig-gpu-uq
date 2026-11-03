"""Vectorised replacement for SOLWEIG-GPU's ``define_patch_characteristics``.

The original (``solweig_gpu/solweig.py``, v2.0.0) loops over the 153 sky patches in Python,
running about 40 full-grid operations per patch and branching on GPU values, every hour.
Every term it accumulates is a sum over patches of a visibility mask times a per-patch
weight, sometimes also times a per-pixel sunlit/shaded flag. Here the same sums are done as
matrix-vector products over chunks of patches.

- Masks use the original's boolean expressions (``(S == 1) & (V == 1)`` etc.), so nothing
  assumes the shadow matrices hold only 0 and 1.
- Patches are processed in chunks (``chunk`` patches at a time) so temporary memory stays
  bounded on large tiles: about rows x cols x chunk x 4 bytes per temporary. Default 8,
  chosen from the memory/speed sweep (see CHUNK).
- Comparisons that decide sunlit or shaded walls reproduce the original's dtype promotion.
- Only the summation order differs from the original, so results agree to float32 rounding
  (checked by ``scripts/bench_fast_sky.py``).
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import numpy as np
import torch

# Patches per chunk. Measured 30 Oct on a 1000 x 1000 tile (scripts/profile_memory.py): chunk 8 adds only
# 0.10 GiB to the original's 2.90 GiB peak, against 0.62 GiB for 64 and 1.47 GiB for all 153 at once.
# It is not quite as fast: single runs on 30 Oct (25.9 against 24.5 s per tile-day) looked like noise, but
# repeated, interleaved runs on 2 Nov (scripts/bench_steps.py) put chunk 8 2-3 s per tile-day slower in
# both sessions. A deliberate trade: about 10% more time for 0.9 GiB less reserved memory.
CHUNK = 8


def _wsum(mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Sum over the last axis of mask (rows, cols, p) times weights (p,) or (p, k)."""
    rows, cols, p = mask.shape
    m = mask.reshape(rows * cols, p).to(weights.dtype)  # a no-op in normal float32 runs
    return (m @ weights).reshape(rows, cols, *weights.shape[1:])


def define_patch_characteristics(solar_altitude, solar_azimuth,
                                 patch_altitude, patch_azimuth, steradian,
                                 asvf,
                                 shmat, vegshmat, vbshvegshmat,
                                 Lsky_down, Lsky_side, Lsky, Lup,
                                 Ta, Tgwall, ewall,
                                 rows, cols, chunk=CHUNK):
    device = shmat.device
    SBC = torch.tensor(5.67051e-8, device=device)
    deg2rad = torch.tensor(np.pi / 180, device=device)
    pi = torch.tensor(np.pi, device=device)
    ewall = torch.as_tensor(ewall, device=device, dtype=torch.float32)

    alt, azi, ster = patch_altitude, patch_azimuth, steradian
    side_w = ster * torch.cos(alt * deg2rad)
    down_w = ster * torch.sin(alt * deg2rad)

    # Cardinal-direction weights, with the original's (asymmetric) azimuth conditions.
    zero = torch.zeros_like(azi)
    dirs = torch.stack([
        torch.where((azi > 360) | (azi < 180), torch.cos((90 - azi) * deg2rad), zero),   # east
        torch.where((azi > 90) & (azi < 270), torch.cos((180 - azi) * deg2rad), zero),   # south
        torch.where((azi > 180) & (azi < 360), torch.cos((270 - azi) * deg2rad), zero),  # west
        torch.where((azi > 270) | (azi < 90), torch.cos((0 - azi) * deg2rad), zero),     # north
    ], dim=1)  # (P, 4)
    bw = torch.cat([side_w[:, None], down_w[:, None], side_w[:, None] * dirs], dim=1)  # side, down, E, S, W, N

    vegetation_surface = (ewall * SBC * ((Ta + 273.15) ** 4)) / pi
    sunlit_surface = (ewall * SBC * ((Ta + Tgwall + 273.15) ** 4)) / pi
    shaded_surface = (ewall * SBC * ((Ta + 273.15) ** 4)) / pi

    # Which patches face the sun. The original subtracts two 0-d tensors per patch, which
    # promotes to the wider dtype; reproduce that so the boundary comparisons agree.
    sa = torch.as_tensor(solar_azimuth, device=device)
    salt = torch.as_tensor(solar_altitude, device=device)
    wide = torch.promote_types(sa.dtype, azi.dtype)
    azimuth_difference = torch.abs(sa.to(wide) - azi.to(wide))
    facing = (azimuth_difference > 90) & (azimuth_difference < 270) & (salt > 0)
    xi = torch.cos(azimuth_difference * (torch.pi / 180.0))
    yi = 2 * xi * torch.tan(salt.to(wide) * (torch.pi / 180.0))
    yi_ = torch.where(yi > 0, 0.0, yi).to(asvf.dtype)  # the original adds a 0-d value to a float32 grid
    hsvf = torch.tan(asvf)

    sky = torch.zeros((rows, cols, 6), device=device)   # Ldown_sky, Lside_sky, E, S, W, N
    veg = torch.zeros((rows, cols, 6), device=device)   # Lside, Ldown, E, S, W, N (times vegetation_surface)
    sun = torch.zeros((rows, cols, 6), device=device)   # times sunlit_surface
    shd = torch.zeros((rows, cols, 6), device=device)   # times shaded_surface
    ref = torch.zeros((rows, cols, 6), device=device)   # times reflected_on_surfaces

    sky_w = torch.cat([Lsky_down[:, 2:3], Lsky_side[:, 2:3], Lsky_side[:, 2:3] * dirs], dim=1)

    for start in range(0, alt.shape[0], chunk):
        sl = slice(start, start + chunk)
        S, V, B = shmat[:, :, sl], vegshmat[:, :, sl], vbshvegshmat[:, :, sl]

        sky += _wsum(((S == 1) & (V == 1)).float(), sky_w[sl])
        veg += _wsum(((V == 0) | (B == 0)).float(), bw[sl])
        ref += _wsum(((S == 0) | (V == 0) | (B == 0)).float(), bw[sl])

        temp_sh = (((1 - S) * B) == 1).float()
        f = facing[sl]
        off = torch.nonzero(~f).flatten()
        on = torch.nonzero(f).flatten()
        if len(off):
            shd += _wsum(temp_sh[:, :, off], bw[sl][off])
        if len(on):
            sunlit_degrees = torch.atan(hsvf[:, :, None] + yi_[sl][on][None, None, :]) * (180.0 / torch.pi)
            pa = alt[sl][on][None, None, :]
            ts = temp_sh[:, :, on]
            sun += _wsum(ts * (sunlit_degrees < pa), bw[sl][on])
            shd += _wsum(ts * (sunlit_degrees > pa), bw[sl][on])
            del sunlit_degrees, ts
        del temp_sh, S, V, B

    Ldown_sky, Lside_sky = sky[..., 0], sky[..., 1]
    Lside_veg, Ldown_veg = vegetation_surface * veg[..., 0], vegetation_surface * veg[..., 1]
    Lside_sun, Ldown_sun = sunlit_surface * sun[..., 0], sunlit_surface * sun[..., 1]
    Lside_sh, Ldown_sh = shaded_surface * shd[..., 0], shaded_surface * shd[..., 1]

    reflected_on_surfaces = ((Ldown_sky + Lup) * (1 - ewall) * 0.5) / pi
    Lside_ref, Ldown_ref = reflected_on_surfaces * ref[..., 0], reflected_on_surfaces * ref[..., 1]

    d = (sky[..., 2:] + vegetation_surface * veg[..., 2:] + sunlit_surface * sun[..., 2:]
         + shaded_surface * shd[..., 2:] + reflected_on_surfaces[..., None] * ref[..., 2:])
    Least, Lsouth, Lwest, Lnorth = d[..., 0], d[..., 1], d[..., 2], d[..., 3]

    Lside = Lside_sky + Lside_veg + Lside_sh + Lside_sun + Lside_ref
    Ldown = Ldown_sky + Ldown_veg + Ldown_sh + Ldown_sun + Ldown_ref
    return Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, Least, Lwest, Lnorth, Lsouth


@contextlib.contextmanager
def fast_sky() -> Iterator[None]:
    """Swap the vectorised version in for the duration of a run."""
    import solweig_gpu.solweig as sw

    old = sw.define_patch_characteristics
    sw.define_patch_characteristics = define_patch_characteristics
    try:
        yield
    finally:
        sw.define_patch_characteristics = old
