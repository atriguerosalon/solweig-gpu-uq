"""The sky-patch shadow loop without per-step syncs (C7) must give the same masks, bit for bit.

CPU, synthetic tile on raised terrain with buildings, trees and bushes (the per-step bush test
is one of the removed syncs), all sun directions of interest including the zenith patch, with
the absolute loop bound of v2.0.0 and with the tight bound.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
shm = pytest.importorskip("solweig_gpu.shadow")

from solweig_sens.fast_sky_shadow import fast_sky_shadow  # noqa: E402
from solweig_sens.tight_bound import tight_bound  # noqa: E402

N = 48
AZIMUTHS = [0.0, 1e-4, 30.0, 44.99999, 45.0, 90.0, 135.0, 180.0, 200.0, 270.0, 315.0, 359.99]


def _scene(with_bush: bool):
    g = torch.Generator().manual_seed(1)
    ground = 300.0 + 5.0 * torch.rand((N, N), generator=g)
    bld = torch.zeros((N, N))
    bld[8:16, 10:20] = 9.0
    bld[30:40, 25:31] = 21.0
    dsm = ground + bld
    canopy = torch.zeros((N, N))
    canopy[5:9, 30:36] = 7.0
    canopy[35:44, 5:12] = 12.0
    vegdem = torch.where(canopy > 0, dsm + canopy, torch.zeros_like(dsm))
    vegdem2 = torch.where(canopy > 0, dsm + 0.25 * canopy, torch.zeros_like(dsm))
    bush = torch.zeros((N, N))
    if with_bush:  # canopy without a trunk zone, as SOLWEIG-GPU derives it: vegdem where vegdem2 is 0
        bush[20:24, 38:44] = dsm[20:24, 38:44] + 2.0
        vegdem = torch.where(bush > 0, bush, vegdem)
    amax = torch.maximum(dsm.max(), vegdem.max())  # absolute, as v2.0.0 passes it
    return amax, dsm, vegdem, vegdem2, bush


@pytest.mark.parametrize("with_bush", [False, True])
@pytest.mark.parametrize("altitude", [3.0, 10.0, 35.0, 80.0, 90.0])
def test_masks_bit_identical(altitude, with_bush):
    amax, dsm, vegdem, vegdem2, bush = _scene(with_bush)
    for az in AZIMUTHS:
        args = (amax, dsm, vegdem, vegdem2, bush, az, altitude, 0.5)
        original = shm.shadow(*args)
        with fast_sky_shadow():
            fast = shm.shadow(*args)
        with fast_sky_shadow(), tight_bound():
            fast_tight = shm.shadow(*args)
        for x, y, z in zip(original, fast, fast_tight):
            assert torch.equal(x, y) and torch.equal(x, z)


def test_bush_scene_exercises_the_bush_branch():
    amax, dsm, vegdem, vegdem2, bush = _scene(True)
    assert bool((bush > 1.).any())  # bushplant non-empty, so the replaced test matters
    _, vegsh_bush, _ = shm.shadow(amax, dsm, vegdem, vegdem2, bush, 135.0, 20.0, 0.5)
    _, vegsh_none, _ = shm.shadow(amax, dsm, vegdem, vegdem2, torch.zeros_like(bush), 135.0, 20.0, 0.5)
    assert not torch.equal(vegsh_bush, vegsh_none)


def test_restores_the_function():
    original = shm.shadow
    with fast_sky_shadow():
        assert shm.shadow is not original
    assert shm.shadow is original
