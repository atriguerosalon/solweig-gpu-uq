"""The tight loop bound must not change any shadow.

CPU, synthetic tile on raised terrain (ground near 300 m, as in Pacoima), with the absolute
bound v2.0.0 passes. The GPU check on a real tile is in docs/PERFORMANCE_LOG.md.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
sw = pytest.importorskip("solweig_gpu.solweig")
shm = pytest.importorskip("solweig_gpu.shadow")

from solweig_sens import fast  # noqa: E402
from solweig_sens.tight_bound import tight, tight_bound  # noqa: E402

N = 48
AZIMUTHS = [0.0, 30.0, 45.0, 90.0, 135.0, 200.0, 270.0, 315.0]


def _scene(seed=0):
    g = torch.Generator().manual_seed(seed)
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
    walls = torch.zeros((N, N))
    walls[8, 10:20] = 9.0
    walls[30:40, 25] = 21.0
    aspect = torch.rand((N, N), generator=g) * 6.283 * (walls > 0)
    amax = torch.maximum(dsm.max(), vegdem.max())  # absolute, as v2.0.0 passes it
    return dsm, vegdem, vegdem2, walls, aspect, torch.zeros((N, N)), amax


def _same(r1, r2):
    for x, y in zip(r1, r2):
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y


def test_bound_is_the_height_range():
    dsm, vegdem, *_, amax = _scene()
    assert float(tight(amax, dsm, vegdem)) == pytest.approx(float(amax - dsm.min()))  # about 25 m, not 326
    tall = vegdem + 30.0 * (vegdem > 0)  # canopy above the tallest building
    assert float(tight(amax, dsm, tall)) == pytest.approx(float(tall.max() - dsm.min()))
    assert float(tight(1.0, dsm, vegdem)) == 1.0  # never raises a smaller bound


@pytest.mark.parametrize("altitude", [3.0, 10.0, 35.0, 80.0, 90.0])
def test_sky_patch_shadows_bit_identical(altitude):
    dsm, vegdem, vegdem2, _, _, bush, amax = _scene()
    for az in AZIMUTHS:
        args = (amax, dsm, vegdem, vegdem2, bush, az, altitude, 0.5)
        original = shm.shadow(*args)
        with tight_bound():
            capped = shm.shadow(*args)
        _same(original, capped)


@pytest.mark.parametrize("altitude", [0.5, 3.0, 10.0, 35.0, 80.0])
def test_hourly_shadows_bit_identical_with_fast_functions(altitude):
    dsm, vegdem, vegdem2, walls, aspect, bush, amax = _scene()
    for az in AZIMUTHS:
        args = (dsm, vegdem, vegdem2, az, altitude, 0.5, amax, bush, walls, aspect)
        original = sw.shadowingfunction_wallheight_23(*args)
        with fast.fast_solweig(), tight_bound():
            capped = sw.shadowingfunction_wallheight_23(*args)
        _same(original, capped)


def test_restores_the_functions():
    sky, hourly = shm.shadow, sw.shadowingfunction_wallheight_23
    with fast.fast_solweig():
        fast_hourly = sw.shadowingfunction_wallheight_23
        with tight_bound():
            assert shm.shadow is not sky and sw.shadowingfunction_wallheight_23 is not fast_hourly
        assert sw.shadowingfunction_wallheight_23 is fast_hourly
    assert shm.shadow is sky and sw.shadowingfunction_wallheight_23 is hourly
