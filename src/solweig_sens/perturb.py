"""Input perturbations for SOLWEIG-GPU.

Three kinds of input are perturbed:

* model constants that SOLWEIG-GPU keeps as module globals in
  ``solweig_gpu.utci_process`` (wall albedo, wall emissivity, body absorption),
* per-class surface properties from the land-cover lookup table,
* the meteorological forcing (UMEP met file columns).

Each perturbation is applied inside a context manager that restores the
original state afterwards, so cases can run back to back in one process.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

# Column indices in the UMEP met file (0-based), as read by solweig_gpu.utci_process.
MET_COLS = {"wind": 9, "rh": 10, "ta": 11, "kdown": 14}

# Land-cover codes in landcoverclasses_2016a.txt.
PAVED_CLASSES = (0, 1)  # cobble stone, dark asphalt
# Column index in each table row. ts_deg is the ground-heating coefficient: daily surface
# temperature amplitude above air = ts_deg * max sun altitude (degrees) + tstart.
LC_COLS = {"albedo": 2, "emissivity": 3, "ts_deg": 4}


@dataclass(frozen=True)
class Perturbation:
    """One perturbed case: a name, the input it changes, and how."""

    name: str
    group: str  # input being varied, e.g. "wall albedo"
    value: float  # perturbed value, for labels
    kind: str  # "global", "landcover" or "met"
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------- model globals

@contextlib.contextmanager
def patched_globals(**values: float) -> Iterator[None]:
    """Temporarily override module-level constants in solweig_gpu.utci_process."""
    import solweig_gpu.utci_process as up

    old = {k: getattr(up, k) for k in values}
    try:
        for k, v in values.items():
            setattr(up, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(up, k, v)


# ------------------------------------------------------------------- land cover

def write_landcover_table(src: Path, dst: Path, classes: tuple[int, ...], values: dict[str, float]) -> None:
    """Copy the land-cover lookup table, setting properties (e.g. albedo) for some classes."""
    lines = src.read_text().splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        parts = line.split()
        if parts and int(parts[1]) in classes:
            for column, value in values.items():
                parts[LC_COLS[column]] = f"{value:.3f}"
        out.append(" ".join(parts))
    dst.write_text("\n".join(out) + "\n")


@contextlib.contextmanager
def patched_landcover(workdir: Path, classes: tuple[int, ...], values: dict[str, float]) -> Iterator[None]:
    """Point SOLWEIG-GPU at an edited copy of its land-cover table."""
    import solweig_gpu.utci_process as up

    src = Path(up.landcover_classes_path)
    tag = "_".join(f"{k}{v:.3f}" for k, v in values.items())
    dst = workdir / f"landcover_{tag}.txt"
    write_landcover_table(src, dst, classes, values)
    old = up.landcover_classes_path
    up.landcover_classes_path = str(dst)
    try:
        yield
    finally:
        up.landcover_classes_path = old


# ----------------------------------------------------------------- met forcing

def perturb_met(data: np.ndarray, column: str, *, add: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Return a copy of a met array with one column shifted and/or scaled.

    Missing values (-999) are left alone. Relative humidity is clipped to 0-100 %,
    wind speed and radiation to non-negative values.
    """
    out = data.copy()
    col = MET_COLS[column]
    x = out[:, col]
    valid = x != -999.0
    x[valid] = x[valid] * scale + add
    if column == "rh":
        x[valid] = np.clip(x[valid], 0.0, 100.0)
    elif column in ("wind", "kdown"):
        x[valid] = np.maximum(x[valid], 0.0)
    out[:, col] = x
    return out


def _read_met(path: Path) -> tuple[str, np.ndarray]:
    header = path.read_text().splitlines()[0]
    return header, np.loadtxt(path, skiprows=1, delimiter=" ")


def _write_met(path: Path, header: str, data: np.ndarray) -> None:
    np.savetxt(path, data, fmt="%.4f", delimiter=" ", header=header, comments="")


def restore_metfiles(metfile_dir: Path) -> None:
    """Put back original met files left in the backup folder by an interrupted run."""
    backup = metfile_dir.parent / (metfile_dir.name + "_orig")
    if backup.exists():
        shutil.rmtree(metfile_dir, ignore_errors=True)
        shutil.copytree(backup, metfile_dir)
        shutil.rmtree(backup)


@contextlib.contextmanager
def patched_metfiles(metfile_dir: Path, changes: dict[str, dict[str, float]]) -> Iterator[None]:
    """Rewrite every per-tile met file in place, restoring the originals afterwards.

    ``changes`` maps a column name to ``{"add": ..., "scale": ...}``.
    """
    backup = metfile_dir.parent / (metfile_dir.name + "_orig")
    if backup.exists():  # a previous run was killed before restoring: the backup holds the originals
        restore_metfiles(metfile_dir)
    shutil.copytree(metfile_dir, backup)
    try:
        for f in sorted(metfile_dir.glob("*.txt")):
            header, data = _read_met(f)
            for column, ch in changes.items():
                data = perturb_met(data, column, add=ch.get("add", 0.0), scale=ch.get("scale", 1.0))
            _write_met(f, header, data)
        yield
    finally:
        shutil.rmtree(metfile_dir)
        shutil.copytree(backup, metfile_dir)
        shutil.rmtree(backup)


# --------------------------------------------------------------- the case list

def default_cases() -> list[Perturbation]:
    """Low and high value for each input. Ranges are plausible, not measured, bounds."""
    P = Perturbation
    return [
        P("wall_albedo_lo", "wall albedo", 0.10, "global", {"albedo_b": 0.10}),
        P("wall_albedo_hi", "wall albedo", 0.40, "global", {"albedo_b": 0.40}),
        P("wall_emis_lo", "wall emissivity", 0.85, "global", {"ewall": 0.85}),
        P("wall_emis_hi", "wall emissivity", 0.95, "global", {"ewall": 0.95}),
        P("abs_k_lo", "body shortwave absorption", 0.60, "global", {"absK": 0.60}),
        P("abs_k_hi", "body shortwave absorption", 0.80, "global", {"absK": 0.80}),
        P("paved_albedo_lo", "paved albedo", 0.10, "landcover", {"column": "albedo", "value": 0.10}),
        P("paved_albedo_hi", "paved albedo", 0.35, "landcover", {"column": "albedo", "value": 0.35}),
        P("paved_emis_lo", "paved emissivity", 0.90, "landcover", {"column": "emissivity", "value": 0.90}),
        P("paved_emis_hi", "paved emissivity", 0.98, "landcover", {"column": "emissivity", "value": 0.98}),
        P("ta_lo", "air temperature", -2.0, "met", {"column": "ta", "add": -2.0}),
        P("ta_hi", "air temperature", +2.0, "met", {"column": "ta", "add": +2.0}),
        P("rh_lo", "relative humidity", -15.0, "met", {"column": "rh", "add": -15.0}),
        P("rh_hi", "relative humidity", +15.0, "met", {"column": "rh", "add": +15.0}),
        P("wind_lo", "wind speed", 0.5, "met", {"column": "wind", "scale": 0.5}),
        P("wind_hi", "wind speed", 1.5, "met", {"column": "wind", "scale": 1.5}),
        P("kdown_lo", "global radiation", 0.9, "met", {"column": "kdown", "scale": 0.9}),
        P("kdown_hi", "global radiation", 1.1, "met", {"column": "kdown", "scale": 1.1}),
    ]


def applied(case: Perturbation, workdir: Path, metfile_dir: Path) -> contextlib.AbstractContextManager:
    """Context manager that applies one case."""
    if case.kind == "global":
        return patched_globals(**case.params)
    if case.kind == "landcover":
        return patched_landcover(workdir, PAVED_CLASSES, {case.params["column"]: case.params["value"]})
    if case.kind == "met":
        p = case.params
        return patched_metfiles(metfile_dir, {p["column"]: {"add": p.get("add", 0.0), "scale": p.get("scale", 1.0)}})
    raise ValueError(f"unknown perturbation kind: {case.kind}")


@contextlib.contextmanager
def applied_many(sample: dict[str, float], workdir: Path, metfile_dir: Path) -> Iterator[None]:
    """Apply several inputs at once, for global designs such as Morris.

    Keys: ``albedo_b``, ``ewall``, ``absK`` (model globals), ``paved_albedo``,
    ``paved_emissivity`` (land-cover table), ``ta_add``, ``rh_add``,
    ``wind_scale``, ``kdown_scale`` (met forcing).
    """
    globals_ = {k: sample[k] for k in ("albedo_b", "ewall", "absK") if k in sample}
    lc = {col: sample[f"paved_{col}"] for col in ("albedo", "emissivity") if f"paved_{col}" in sample}
    met = {}
    for key, col, how in [("ta_add", "ta", "add"), ("rh_add", "rh", "add"),
                          ("wind_scale", "wind", "scale"), ("kdown_scale", "kdown", "scale")]:
        if key in sample:
            met[col] = {how: sample[key]}
    with contextlib.ExitStack() as stack:
        stack.enter_context(patched_globals(**globals_))
        if lc:
            stack.enter_context(patched_landcover(workdir, PAVED_CLASSES, lc))
        if met:
            stack.enter_context(patched_metfiles(metfile_dir, met))
        yield
