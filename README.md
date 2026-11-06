# solweig-gpu-uq

Sensitivity of street-scale heat stress (UTCI and mean radiant temperature) in [SOLWEIG-GPU](https://github.com/nvnsudharsan/SOLWEIG-GPU) to its uncertain inputs: surface albedo and emissivity, body absorption, and the meteorological forcing.

SOLWEIG has no uncertainty quantification. This repo asks the first question on the way there: which inputs actually move the result, where, and when? It wraps SOLWEIG-GPU v2.0.0 without modifying it and runs on a laptop GPU.

## What it does

- **One at a time:** nine inputs, each set to a low and a high value (18 runs of 24 hours).
- **Morris screening:** the same nine inputs changed together along 10 random trajectories (100 runs), with indices for domain means, sunlit and shaded ground, the area in very strong heat stress, and per pixel.
- **Emulator:** a per-pixel linear fit on the Morris runs, cross-validated, then used for Monte Carlo uncertainty maps.
- **Spatial air temperature:** SOLWEIG-GPU uses one air temperature for the whole domain. This recomputes UTCI with a spatially varying field to see how much that assumption matters. It needs no extra model runs.
- **Los Angeles, against measurements:** a 2 x 2 km, 2 m domain for Pacoima built from public data (USGS 3DEP, LA County building outlines, Meta/WRI 1 m canopy height, ESA WorldCover), with the 30 July 2019 forcing from the Van Nuys airport station and ERA5 radiation. Regular (albedo 0.07) and coated (0.25) pavement are compared with the Tmrt that Middel et al. (2020, ERL 15, 064016) measured over both that day, with SOLWEIG's default ground heating and with heating set to the measured surface temperatures.
- **Heat kept after sunset (experimental):** SOLWEIG drops the ground to air temperature at sunset. An opt-in patch lets the daytime excess decay instead, to see how much of the evening gap to the measurements that explains.

Perturbations are context managers that patch module constants, the land-cover table or the met file, and always restore them, so all cases run in one process and share the cached sky view factor.

More in `docs/`: [SOLWEIG_MODIFICATIONS.md](docs/SOLWEIG_MODIFICATIONS.md) lists every change we make to SOLWEIG-GPU (including speed-ups that make a tile-day 3.3x faster, outputs equal to within 0.0001 °C) and how each was checked, [PERFORMANCE_LOG.md](docs/PERFORMANCE_LOG.md) has the measurements, and [FIGURES.md](docs/FIGURES.md) the captions of the main figures.

## Setup

Linux or WSL with an NVIDIA GPU. GDAL is installed from conda-forge.

```bash
bash scripts/setup_env.sh            # micromamba env "solweig": Python 3.11, GDAL, PyTorch (CUDA 12.1), solweig-gpu 2.0.0, SALib
micromamba run -n solweig pytest     # 11 tests, CPU only, no data or network needed
```

Data: the SOLWEIG-GPU v2.0 sample (Austin, Texas, 2 m), [Zenodo 21081622](https://zenodo.org/records/21081622). Only `Input_rasters.zip` and `ownmet_Forcing_data.txt` are needed; put them in `data/`.

## Running

```bash
python scripts/crop_domain.py --src data/austin/Input_rasters --dst data/austin_crop --size 1000
python scripts/run_all.py --rasters data/austin_crop --met data/ownmet_Forcing_data.txt
python scripts/make_figures.py --rasters data/austin_crop --hour 14
python scripts/run_morris.py --rasters data/austin_crop --met data/ownmet_Forcing_data.txt   # resumable, about 1 h
python scripts/analyse_morris.py --runs runs_morris --out figures
python scripts/spatial_ta.py --runs runs --out figures
```

For the whole day, run to 23:00 with every hour saved (`run_morris.py --hour 23 --all-hours --runs runs_morris_day`), then `scripts/analyse_morris_day.py`.

Los Angeles (the first two download public data, no account or key needed):

```bash
python scripts/build_la_domain.py --lat 34.2735 --lon -118.410 --size 1000 --res 2 --out data/la/pacoima
python scripts/make_la_met.py --date 2019-07-30 --station VNY --lat 34.2735 --lon -118.41 --out data/la/met_pacoima_20190730.txt
python scripts/run_la_pavement.py --rasters data/la/pacoima --met data/la/met_pacoima_20190730.txt
python scripts/analyse_la_pavement.py --runs runs_la --rasters data/la/pacoima --out figures
python scripts/run_la_storage.py --runs runs_la
python scripts/analyse_la_storage.py --runs runs_la --rasters data/la/pacoima --out figures
```

Each script's `--help` lists its arguments. Cost on an RTX 4070 Laptop GPU: about 4.7 min per 24-hour run of 2 x 2 km, once the sky view factor is cached.

## Licence and citation

GPL-3.0 or later (see [LICENSE](LICENSE)), the licence of SOLWEIG-GPU and its sample data.

If you use this, please cite SOLWEIG-GPU and SOLWEIG:

- Kamath, H. G., Sudharsan, N., Singh, M., Wallenberg, N., Lindberg, F., & Niyogi, D. (2026). SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework for Urban Digital Twins. *Journal of Open Source Software*, 11(118), 9535. https://doi.org/10.21105/joss.09535
- Lindberg, F., Holmer, B., & Thorsson, S. (2008). SOLWEIG 1.0: Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *International Journal of Biometeorology*, 52, 697–713. https://doi.org/10.1007/s00484-008-0162-7
