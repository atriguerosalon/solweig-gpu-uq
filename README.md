# solweig-gpu-uq

Uncertainty in street-scale heat stress (UTCI and mean radiant temperature) from [SOLWEIG-GPU](https://github.com/nvnsudharsan/SOLWEIG-GPU): which inputs move it, how far it is from street measurements, and what forecast errors do to a heat-stress forecast at 2 m. Along the way SOLWEIG-GPU v2.0.0 got about 3.3x faster on a laptop GPU, with the same outputs.

Everything wraps SOLWEIG-GPU v2.0.0 at run time. The installed package is never edited: each change is a context manager that is undone when the block ends, and checks that SOLWEIG-GPU's source is the one it was written for.

## What is here

- **Speed and memory** (C1-C9). One 2 x 2 km tile-day at 2 m: 48.4 s to 14.7 s, peak GPU memory 2.9 to 1.1 GiB, outputs bit-identical except one summation order (within 0.0001 °C). A 4 x 4 km tile now fits an 8 GB GPU. Every change, how it was checked, and its costs: [docs/SOLWEIG_MODIFICATIONS.md](docs/SOLWEIG_MODIFICATIONS.md). Measurements and bounds: [docs/PERFORMANCE_LOG.md](docs/PERFORMANCE_LOG.md).
- **Sensitivity.** One at a time and Morris screening on the SOLWEIG-GPU sample (Austin); Sobol indices on a Los Angeles tile (Pacoima, 5120 runs, with a Gaussian-process surrogate checked out of sample) and a first Sobol pilot on a second tile (Exposition Park).
- **Against measurements.** Pacoima built from public data, compared with the cool-pavement Tmrt measurements of Middel et al. (2020): input uncertainty against model bias, and an experimental patch that keeps ground heat after sunset.
- **Forecast errors at street scale.** Perfect-model ensembles with air-temperature, wind and radiation errors; shortcuts that build a 20-member ensemble from 4-14 physics runs; HRRR's real errors at 19 Los Angeles stations; errors that vary in space; a street-scale forecast driven by real HRRR forecasts through the September 2022 heatwave; and what a forecast cycle costs.
- **Notes on SOLWEIG-GPU v2.0.0.** A bracket in the Perez sky clearness that differs from the published formula (carried over from UMEP), the sky view step generating 149 of 153 patches, and what we measured about draft PR #137. In [docs/SOLWEIG_MODIFICATIONS.md](docs/SOLWEIG_MODIFICATIONS.md), section Upstream. The code is left as it is.

The main figures, their captions and the scripts that draw them: [docs/FIGURES.md](docs/FIGURES.md).

## Setup

Linux or WSL with an NVIDIA GPU. GDAL is installed from conda-forge.

```bash
bash scripts/setup_env.sh            # micromamba env "solweig": Python 3.11, GDAL, PyTorch (CUDA 12.1), solweig-gpu 2.0.0, SALib
micromamba run -n solweig pytest     # 75 tests, CPU only, no data or network needed
```

The HRRR scripts (`hrrr_errors.py`, `forecast_demo.py hrrr`) need a second env with `herbie-data`, `cfgrib`, `eccodes`, `xarray`, `scipy` and `matplotlib` from conda-forge.

## Using the speed-ups

```python
from solweig_gpu import run_utci_tiles
from solweig_sens.compact_masks import compact_masks
from solweig_sens.fast import fast_solweig
from solweig_sens.hoist_geometry import hoist_geometry
from solweig_sens.tight_bound import tight_bound

with fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(), tight_bound():
    run_utci_tiles(base_path=..., preprocess_dir=..., selected_date_str="2019-07-30")
```

For the sky view step, `compact_masks("bits", svf=True)`, `fast_sky_shadow()` and `tight_bound()` around `calculate_svf`. Each can be switched off on its own; `scripts/verify_fast.py`, `verify_compact.py` and `verify_compact_svf.py` compare against the original on a real tile.

## Data

Not in git. The SOLWEIG-GPU v2.0 sample (Austin, Texas, 2 m) is on [Zenodo 21081622](https://zenodo.org/records/21081622); only `Input_rasters.zip` and `ownmet_Forcing_data.txt` are needed, in `data/`. The Los Angeles tiles and forcing are built from public sources by the scripts below (USGS 3DEP, LA County building outlines, Meta/WRI canopy height, ESA WorldCover, the Iowa Environmental Mesonet ASOS archive, ERA5 through Open-Meteo); HRRR comes from the NOAA archive on AWS. No account or key is needed. `data_checksums.sha256` records what was downloaded.

## Running

Austin sample, one at a time and Morris:

```bash
python scripts/crop_domain.py --src data/austin/Input_rasters --dst data/austin_crop --size 1000
python scripts/run_all.py --rasters data/austin_crop --met data/ownmet_Forcing_data.txt
python scripts/run_morris.py --rasters data/austin_crop --met data/ownmet_Forcing_data.txt
python scripts/analyse_morris.py --runs runs_morris --out figures
```

Los Angeles tile and forcing, then the pavement comparison:

```bash
python scripts/build_la_domain.py --lat 34.2735 --lon -118.410 --size 1000 --res 2 --out data/la/pacoima
python scripts/make_la_met.py --date 2019-07-30 --station VNY --lat 34.2735 --lon -118.41 --out data/la/met_pacoima_20190730.txt
python scripts/run_la_pavement.py --rasters data/la/pacoima --met data/la/met_pacoima_20190730.txt
python scripts/analyse_la_pavement.py --runs runs_la --rasters data/la/pacoima --out figures
```

Every other study has its own script, listed with its figure in [docs/FIGURES.md](docs/FIGURES.md); each script's `--help` says what it does, what it needs and what it writes. On an RTX 4070 Laptop GPU a 2 x 2 km tile-day takes about 15 s with the speed-ups, once the sky view factor is cached.

## Licence and citation

GPL-3.0 or later (see [LICENSE](LICENSE)), the licence of SOLWEIG-GPU and its sample data.

If you use this, please cite SOLWEIG-GPU and SOLWEIG:

- Kamath, H. G., Sudharsan, N., Singh, M., Wallenberg, N., Lindberg, F., & Niyogi, D. (2026). SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework for Urban Digital Twins. *Journal of Open Source Software*, 11(118), 9535. https://doi.org/10.21105/joss.09535
- Lindberg, F., Holmer, B., & Thorsson, S. (2008). SOLWEIG 1.0: Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *International Journal of Biometeorology*, 52, 697–713. https://doi.org/10.1007/s00484-008-0162-7
