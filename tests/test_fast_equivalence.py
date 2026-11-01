"""The fast replacements must match the original SOLWEIG-GPU functions.

Runs on the CPU with small synthetic tiles, so it needs no GPU and no data. Shadow casting
and ground view factors must be bit-identical; the sky may differ only by float32 rounding
(summation order). The full GPU check on real tiles is scripts/verify_fast.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
sw = pytest.importorskip("solweig_gpu.solweig")

from solweig_sens import fast  # noqa: E402

N = 48
AZIMUTHS = [0.0, 1e-4, 30.0, 44.99999, 45.0, 45.00001, 90.0, 135.0, 180.0, 200.0, 225.0, 270.0, 315.0, 359.99]


def _scene(seed=0):
    g = torch.Generator().manual_seed(seed)
    ground = 100.0 + torch.rand((N, N), generator=g)
    bld = torch.zeros((N, N))
    bld[8:16, 10:20] = 9.0
    bld[30:40, 25:31] = 21.0
    bld[20:24, 5:8] = 4.0
    dsm = ground + bld
    canopy = torch.zeros((N, N))
    canopy[5:9, 30:36] = 7.0
    canopy[35:44, 5:12] = 12.0
    vegdem = torch.where(canopy > 0, ground + canopy, torch.zeros_like(ground))
    vegdem2 = torch.where(canopy > 0, ground + 0.25 * canopy, torch.zeros_like(ground))
    walls = torch.zeros((N, N))
    walls[8, 10:20] = 9.0
    walls[30:40, 25] = 21.0
    aspect = torch.rand((N, N), generator=g) * 2 * np.pi * (walls > 0)
    bush = torch.zeros((N, N))
    return dsm, vegdem, vegdem2, walls, aspect, bush, float(bld.max())


def _assert_same(r1, r2, rtol=0.0):
    for x, y in zip(r1, r2):
        if isinstance(x, torch.Tensor):
            x, y = x.double(), y.double()
            if rtol == 0.0:
                assert torch.equal(x, y)
            else:
                assert torch.allclose(x, y, rtol=rtol, atol=rtol * float(x.abs().max().clamp(min=1.0)))
        else:
            assert x == y


def test_fingerprints_match_installed_solweig():
    for name in fast.FINGERPRINTS:
        fast.check_fingerprint(name)


@pytest.mark.parametrize("altitude", [0.5, 3.0, 10.0, 35.0, 80.0])
def test_shadow_casting_bit_identical(altitude):
    dsm, vegdem, vegdem2, walls, aspect, bush, amax = _scene()
    fast_fn = fast.build_fast_functions()["shadowingfunction_wallheight_23"]
    for az in AZIMUTHS:
        args = (dsm, vegdem, vegdem2, az, altitude, 0.5, amax, bush, walls, aspect)
        _assert_same(sw.shadowingfunction_wallheight_23(*args), fast_fn(*args))


def test_shadow_casting_flat_tile():
    dsm, vegdem, vegdem2, walls, aspect, bush, _ = _scene()
    fast_fn = fast.build_fast_functions()["shadowingfunction_wallheight_23"]
    for amax in (0.0, -1.0):  # no buildings; negative maximum skips the loop entirely
        args = (dsm, vegdem, vegdem2, 120.0, 30.0, 0.5, amax, bush, walls, aspect)
        _assert_same(sw.shadowingfunction_wallheight_23(*args), fast_fn(*args))


def test_ground_view_factor_bit_identical():
    dsm, vegdem, vegdem2, walls, aspect, bush, amax = _scene()
    buildings = (dsm - dsm.min() < 2).float()  # 1 = ground, 0 = building, as in SOLWEIG
    shadow = (torch.rand((N, N), generator=torch.Generator().manual_seed(1)) > 0.4).float()
    fast_fn = fast.build_fast_functions()["sunonsurface_2018a"]
    for az in AZIMUTHS + [5.0, 25.0, 345.0]:
        def args():  # the function modifies some inputs in place, so give each call its own copies
            return (torch.tensor(az, dtype=torch.float32), 0.5, buildings.clone(), shadow.clone(),
                    (walls > 0).float(), torch.tensor(1.0), torch.tensor(22.0), aspect.clone(), walls.clone(),
                    torch.full((N, N), 12.0), torch.tensor(6.0), torch.tensor(30.0), torch.full((N, N), 0.95),
                    0.9, torch.full((N, N), 0.15), 5.67051e-8, 0.2, 25.0, torch.ones((N, N)), 1)
        _assert_same(sw.sunonsurface_2018a(*args()), fast_fn(*args()))


@pytest.mark.parametrize("sun", [(-5.0, 10.0), (2.0, 80.0), (20.0, 135.0), (45.0, 180.0), (60.0, 225.0), (85.0, 315.0)])
def test_sky_longwave_within_rounding(sun):
    from solweig_gpu.shadow import create_patches

    alt, azi = sun
    g = torch.Generator().manual_seed(2)
    skyalt, skyazi, *_ = create_patches(2)
    P = skyalt.shape[0]
    patches = torch.stack([torch.as_tensor(skyalt).float().flatten(), torch.as_tensor(skyazi).float().flatten(),
                           torch.zeros(P)], dim=1)
    shmat = (torch.rand((N, N, P), generator=g) > 0.3).float()
    vegshmat = (torch.rand((N, N, P), generator=g) > 0.2).float()
    vbshvegshmat = (torch.rand((N, N, P), generator=g) > 0.2).float()
    asvf = torch.acos(torch.sqrt(0.2 + 0.8 * torch.rand((N, N), generator=g)))
    args = (torch.tensor(0.85), patches, torch.tensor(28.0), torch.tensor(7.0), 0.9,
            400 + 50 * torch.rand((N, N), generator=g), shmat, vegshmat, vbshvegshmat,
            torch.tensor(alt, dtype=torch.float64), torch.tensor(azi, dtype=torch.float64), N, N, asvf)
    original = sw.Lcyl_v2022a(*args)
    old = sw.define_patch_characteristics
    sw.define_patch_characteristics = fast.build_fast_functions()["define_patch_characteristics"]
    try:
        replaced = sw.Lcyl_v2022a(*args)
    finally:
        sw.define_patch_characteristics = old
    _assert_same(original, replaced, rtol=1e-5)


def test_fast_solweig_restores_and_heat_storage_sees_fast_functions():
    import solweig_gpu.utci_process as up
    from solweig_sens.heat_storage import ground_heat_storage

    originals = {name: getattr(sw, name) for name in fast.FINGERPRINTS}
    original_calc = up.Solweig_2022a_calc
    with fast.fast_solweig():
        with ground_heat_storage(2.0):
            # The storage patch is compiled into the live module namespace, so at call time it
            # resolves the shadow function to whatever is installed: here the fast one.
            assert up.Solweig_2022a_calc.__globals__ is vars(sw)
            assert sw.shadowingfunction_wallheight_23 is not originals["shadowingfunction_wallheight_23"]
            assert sw.Solweig_2022a_calc is not up.Solweig_2022a_calc  # module keeps the original
    assert up.Solweig_2022a_calc is original_calc
    assert "_HEAT_STORAGE" not in vars(sw)
    for name, f in originals.items():
        assert getattr(sw, name) is f
