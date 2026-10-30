"""Run SOLWEIG-GPU once for the geometry, then cheaply for each perturbed case.

SOLWEIG-GPU's pipeline has four stages: tiling and met files, wall height and
aspect, sky view factor (the expensive, geometry-only step), and the hourly
radiation and UTCI step. Every case here shares the first three stages and
only reruns the last one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .perturb import Perturbation, applied


@dataclass(frozen=True)
class Setup:
    raster_dir: Path  # folder with Building_DSM.tif, DEM.tif, Trees.tif, Landcover.tif
    met_file: Path  # UMEP-format hourly met file
    runs_dir: Path  # where processed inputs and outputs go
    date: str = "2020-08-13"
    tile_size: int = 1000
    overlap: int = 20
    use_landcover: bool = True

    @property
    def shared_dir(self) -> Path:
        return self.runs_dir / "shared"

    @property
    def preprocess_dir(self) -> Path:
        return self.shared_dir / "processed_inputs"


def prepare(setup: Setup) -> None:
    """Tiles, met files, walls and sky view factor, computed once for all cases."""
    from solweig_gpu import calculate_svf, preprocess, run_walls_aspect

    setup.shared_dir.mkdir(parents=True, exist_ok=True)
    preprocess(
        base_path=str(setup.shared_dir),
        selected_date_str=setup.date,
        building_dsm_filename=str(setup.raster_dir / "Building_DSM.tif"),
        dem_filename=str(setup.raster_dir / "DEM.tif"),
        trees_filename=str(setup.raster_dir / "Trees.tif"),
        landcover_filename=str(setup.raster_dir / "Landcover.tif") if setup.use_landcover else None,
        tile_size=setup.tile_size,
        overlap=setup.overlap,
        use_own_met=True,
        own_met_file=str(setup.met_file),
        use_uhi=False,
    )
    run_walls_aspect(str(setup.preprocess_dir))
    calculate_svf(str(setup.preprocess_dir), patch_option=2, overwrite=False)


def run_case(setup: Setup, case: Perturbation | None) -> float:
    """Run the radiation and UTCI step for one case; returns wall time in seconds.

    ``case=None`` is the baseline, which also saves shadows for the sun/shade split.
    """
    from solweig_gpu import run_utci_tiles

    name = "baseline" if case is None else case.name
    out_dir = setup.runs_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metfile_dir = setup.preprocess_dir / "metfiles"

    def _run() -> None:
        run_utci_tiles(
            base_path=str(out_dir),
            preprocess_dir=str(setup.preprocess_dir),
            selected_date_str=setup.date,
            save_tmrt=True,
            save_shadow=case is None,
            save_svf=case is None,
        )

    t0 = time.perf_counter()
    if case is None:
        _run()
    else:
        with applied(case, out_dir, metfile_dir):
            _run()
    return time.perf_counter() - t0
