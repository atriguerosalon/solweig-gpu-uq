"""Sky-patch geometry computed once per run (C8) must give the same values, bit for bit.

CPU, small synthetic tile. Each function is called for several hours in a row, so the first call
fills the cache and the later ones reuse it; every call must match the original exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
sh = pytest.importorskip("solweig_gpu.shadow")
sw = pytest.importorskip("solweig_gpu.solweig")
up = pytest.importorskip("solweig_gpu.utci_process")

from solweig_sens import hoist_geometry  # noqa: E402
from solweig_sens.hoist_geometry import hoist_geometry as hoist  # noqa: E402

R, C = 9, 11
HOURS = [  # (solar altitude, solar azimuth, air temperature, sky emissivity, direct, diffuse)
    (15.0, 95.0, 18.0, 0.78, 420.0, 90.0),
    (48.0, 140.0, 27.0, 0.82, 760.0, 120.0),
    (62.0, 205.0, 31.0, 0.85, 810.0, 140.0),
    (25.0, 260.0, 29.0, 0.80, 380.0, 110.0),
]


def _maps(seed):
    g = torch.Generator().manual_seed(seed)
    return lambda lo=0.0, hi=1.0: lo + (hi - lo) * torch.rand((R, C), generator=g)


def _masks(seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.rand((R, C, 153), generator=g) > p).float() for p in (0.4, 0.3, 0.2)]


def _same(a, b):
    a, b = (a,) if isinstance(a, torch.Tensor) else a, (b,) if isinstance(b, torch.Tensor) else b
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and torch.equal(x, y)
        else:
            assert x == y


def _lcyl_args(hour):
    alt, azi, ta, esky, _, _ = hour
    m = _maps(1)
    sky_alt, sky_azi = sh.create_patches(2)[:2]
    emis = torch.linspace(0.6, 0.95, sky_alt.shape[0])
    sky_patches = torch.stack([sky_alt.float(), sky_azi.float(), emis], dim=1)
    shmat, vegshmat, vbshvegshmat = _masks(2)
    return (esky, sky_patches, ta, m(280.0, 320.0), 0.9, m(380.0, 520.0), shmat, vegshmat, vbshvegshmat,
            torch.tensor(alt), torch.tensor(azi), R, C, m(0.2, 1.0))  # Lcyl gets the angles as tensors


def _kside_args(hour):
    alt, azi, _, _, rad_i, rad_d = hour
    m = _maps(3)
    shmat, vegshmat, vbshvegshmat = _masks(4)
    lv = sw.Perez_v3(90.0 - alt, azi, rad_d, rad_i, 211, 1, 2)[0]
    diffsh = shmat - (1 - vegshmat) * (1 - 0.03)
    rad_g = rad_i * np.sin(np.radians(alt)) + rad_d
    return (rad_i, rad_d, rad_g, (m() > 0.5).float(), m(), m(), m(), m(), m(), m(), m(), m(),
            azi, alt, 0.03, 0.0, 0.2, m(), m(0, 50), m(0, 50), m(0, 50), m(0, 50), 1, lv, 1,
            diffsh, R, C, m(0.2, 1.0), shmat, vegshmat, vbshvegshmat)


def test_fingerprints_match_installed_solweig():
    hoist_geometry.check_fingerprints()


def test_create_patches_same_values_and_independent_copies():
    original = sh.create_patches(2)
    with hoist():
        first = sh.create_patches(2)
        first[0].add_(1.0)  # a caller changing its copy must not change the next call's result
        second = sw.create_patches(2)
        third = up.create_patches(2)
    _same(original, second)
    _same(original, third)


@pytest.mark.parametrize("fn, args", [("Lcyl_v2022a", _lcyl_args), ("Kside_veg_v2022a", _kside_args)])
def test_bit_identical_hour_after_hour(fn, args):
    original = [getattr(sw, fn)(*args(h)) for h in HOURS]
    with hoist():
        patched = [getattr(sw, fn)(*args(h)) for h in HOURS]  # first call fills the cache, the rest reuse it
        assert len(sw._GEOMETRY_CACHE) == 1
    for o, p in zip(original, patched):
        _same(o, p)


def test_restores_the_functions():
    before = (sw.Lcyl_v2022a, sw.Kside_veg_v2022a, sh.create_patches, sw.create_patches, up.create_patches)
    with hoist():
        assert sw.Lcyl_v2022a is not before[0] and sh.create_patches is not before[2]
    assert (sw.Lcyl_v2022a, sw.Kside_veg_v2022a, sh.create_patches, sw.create_patches, up.create_patches) == before
    assert not hasattr(sw, "_GEOMETRY_CACHE")
