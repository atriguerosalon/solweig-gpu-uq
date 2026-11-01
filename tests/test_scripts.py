import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import emulate  # noqa: E402
import make_la_met  # noqa: E402
import run_la_pavement  # noqa: E402

ASOS_CSV = (
    "station,valid,tmpc,relh,sknt,drct\n"
    "VNY,2019-07-29 22:51,23.00,66.00,4.00,150.00\n"
    "VNY,2019-07-29 23:51,22.80,69.00,4.00,M\n"
    "VNY,2019-07-30 00:51,21.70,73.00,5.00,160.00\n"
    "VNY,2019-07-30 01:30,M,M,M,M\n"
    "VNY,2019-07-30 13:51,31.70,42.00,11.00,150.00\n"
    "VNY,2019-07-30 23:51,22.00,72.00,4.00,150.00\n"
)


def test_station_hours_maps_reports_to_the_following_hour(monkeypatch):
    monkeypatch.setattr(make_la_met, "fetch", lambda url, *a, **k: ASOS_CSV.encode())
    hours = make_la_met.station_hours("VNY", date(2019, 7, 30))
    assert set(hours) == {0, 1, 14}  # 23:51 the day before -> hour 0; 23:51 on the day belongs to the next day
    assert hours[0]["ta"] == 22.8 and hours[0]["wdir"] == -999.0
    assert hours[14]["ta"] == 31.7
    np.testing.assert_allclose(hours[14]["wind"], 11 * make_la_met.KNOT)


def test_fitted_ground_heating_reproduces_measured_asphalt_peak():
    m = run_la_pavement
    peak = m.TA_PEAK + m.TS_FIT * m.ALTMAX + m.TSTART
    assert abs(peak - m.TS_ASPHALT_PEAK) < 0.1
    assert abs((m.TS_FIT_COATED - m.TS_FIT) * m.ALTMAX - m.TS_COATING) < 0.1


def test_heat_storage_patch_compiles_and_restores():
    import solweig_gpu.utci_process as up
    from solweig_sens import heat_storage

    src = heat_storage.patched_source()
    assert "_HEAT_STORAGE.night()" in src and "_HEAT_STORAGE.day(" in src
    original = up.Solweig_2022a_calc
    with heat_storage.ground_heat_storage(2.0):
        assert up.Solweig_2022a_calc is not original
    assert up.Solweig_2022a_calc is original


def test_heat_storage_decay():
    import torch
    from solweig_sens.heat_storage import _State

    s = _State(tau_hours=2.0)
    assert s.night() == 0.0  # nothing stored before the first day step
    s.day(torch.full((2, 2), 500.0), 5.67e-8, torch.full((2, 2), 0.95), torch.tensor(20.0))
    first = s.night()
    second = s.night()
    np.testing.assert_allclose((second / first).numpy(), np.exp(-0.5), rtol=1e-6)


def test_emulator_cross_validation_is_exact_for_linear_response():
    rng = np.random.default_rng(0)
    k, n_traj = 3, 5
    X = rng.random((n_traj * (k + 1), k))
    coef = rng.normal(size=(k, 4))
    Y = 1.0 + X @ coef  # four "pixels", exactly linear
    resid = emulate.loto_errors(X, Y, k, quadratic=False)
    assert np.abs(resid).max() < 1e-10
    assert emulate.design(X, quadratic=True).shape[1] == 1 + k + k * (k + 1) // 2
