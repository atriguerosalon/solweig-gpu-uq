import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.perturb import (  # noqa: E402
    MET_COLS, patched_metfiles, perturb_met, restore_metfiles, write_landcover_table)


def test_metfiles_recovered_after_a_killed_run(tmp_path):
    metdir = tmp_path / "metfiles"
    metdir.mkdir()
    header = "h " * 24
    np.savetxt(metdir / "m.txt", np.full((2, 24), 20.0), fmt="%.4f", header=header.strip(), comments="")
    # Simulate a run killed mid-case: backup holds the originals, metfiles holds perturbed data.
    import shutil
    shutil.copytree(metdir, tmp_path / "metfiles_orig")
    np.savetxt(metdir / "m.txt", np.full((2, 24), 99.0), fmt="%.4f", header=header.strip(), comments="")
    restore_metfiles(metdir)
    assert not (tmp_path / "metfiles_orig").exists()
    assert np.loadtxt(metdir / "m.txt", skiprows=1)[0, 0] == 20.0
    # And a new perturbation still applies on top of the originals, then restores them.
    with patched_metfiles(metdir, {"ta": {"add": 1.0}}):
        assert np.loadtxt(metdir / "m.txt", skiprows=1)[0, MET_COLS["ta"]] == 21.0
    assert np.loadtxt(metdir / "m.txt", skiprows=1)[0, MET_COLS["ta"]] == 20.0


def _met(n=3):
    data = np.zeros((n, 24))
    data[:, MET_COLS["ta"]] = [25.0, 30.0, 35.0]
    data[:, MET_COLS["rh"]] = [10.0, 50.0, 95.0]
    data[:, MET_COLS["wind"]] = [0.0, 2.0, -999.0]
    return data


def test_shift_air_temperature_leaves_other_columns():
    data = _met()
    out = perturb_met(data, "ta", add=2.0)
    np.testing.assert_allclose(out[:, MET_COLS["ta"]], [27.0, 32.0, 37.0])
    np.testing.assert_array_equal(np.delete(out, MET_COLS["ta"], axis=1), np.delete(data, MET_COLS["ta"], axis=1))


def test_humidity_is_clipped():
    out = perturb_met(_met(), "rh", add=15.0)
    np.testing.assert_allclose(out[:, MET_COLS["rh"]], [25.0, 65.0, 100.0])


def test_missing_values_untouched_and_input_not_modified():
    data = _met()
    out = perturb_met(data, "wind", scale=1.5)
    np.testing.assert_allclose(out[:, MET_COLS["wind"]], [0.0, 3.0, -999.0])
    assert data[1, MET_COLS["wind"]] == 2.0


def test_landcover_table_edit(tmp_path):
    src = tmp_path / "lc.txt"
    src.write_text(
        "Name Code Alb Emis Ts_deg Tstart TmaxLST\n"
        "Dark_asphalt 1 0.18 0.95 0.58 -9.78 15.0\n"
        "Grass_unmanaged 5 0.16 0.94 0.21 -3.38 14.0\n"
    )
    dst = tmp_path / "out.txt"
    write_landcover_table(src, dst, classes=(1,), values={"albedo": 0.35, "emissivity": 0.9})
    rows = [line.split() for line in dst.read_text().splitlines()[1:]]
    assert float(rows[0][2]) == 0.35 and float(rows[0][3]) == 0.9
    assert rows[1][2] == "0.16" and rows[1][3] == "0.94"


def test_landcover_ground_heating_edit(tmp_path):
    src = tmp_path / "lc.txt"
    src.write_text(
        "Name Code Alb Emis Ts_deg Tstart TmaxLST\n"
        "Dark_asphalt 1 0.18 0.95 0.58 -9.78 15.0\n"
    )
    dst = tmp_path / "out.txt"
    write_landcover_table(src, dst, classes=(1,), values={"albedo": 0.25, "ts_deg": 0.40})
    row = dst.read_text().splitlines()[1].split()
    assert float(row[2]) == 0.25 and float(row[4]) == 0.40
    assert row[5] == "-9.78"  # tstart unchanged
