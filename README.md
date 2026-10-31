# solweig-gpu-uq

Sensitivity of street-scale heat stress (UTCI and mean radiant temperature) in [SOLWEIG-GPU](https://github.com/nvnsudharsan/SOLWEIG-GPU) to its uncertain inputs: surface albedo and emissivity, body absorption, and the meteorological forcing.

SOLWEIG has no uncertainty quantification. This repo asks the first question on the way there: which inputs actually move the result, where, and when? It wraps SOLWEIG-GPU v2.0.0 without modifying it and runs on a laptop GPU.

## What it does

- **One at a time:** nine inputs, each set to a low and a high value (18 runs of 24 hours).

Perturbations are context managers that patch module constants, the land-cover table or the met file, and always restore them, so all cases run in one process and share the cached sky view factor.

## Setup

Linux or WSL with an NVIDIA GPU. GDAL is installed from conda-forge.

```bash
bash scripts/setup_env.sh            # micromamba env "solweig": Python 3.11, GDAL, PyTorch (CUDA 12.1), solweig-gpu 2.0.0, SALib
micromamba run -n solweig pytest     # 6 tests, CPU only, no data or network needed
```

Data: the SOLWEIG-GPU v2.0 sample (Austin, Texas, 2 m), [Zenodo 21081622](https://zenodo.org/records/21081622). Only `Input_rasters.zip` and `ownmet_Forcing_data.txt` are needed; put them in `data/`.

## Running

```bash
python scripts/crop_domain.py --src data/austin/Input_rasters --dst data/austin_crop --size 1000
python scripts/run_all.py --rasters data/austin_crop --met data/ownmet_Forcing_data.txt
python scripts/make_figures.py --rasters data/austin_crop --hour 14
```

Each script's `--help` lists its arguments. Cost on an RTX 4070 Laptop GPU: about 4.7 min per 24-hour run of 2 x 2 km, once the sky view factor is cached.

## Licence and citation

GPL-3.0 or later (see [LICENSE](LICENSE)), the licence of SOLWEIG-GPU and its sample data.

If you use this, please cite SOLWEIG-GPU and SOLWEIG:

- Kamath, H. G., Sudharsan, N., Singh, M., Wallenberg, N., Lindberg, F., & Niyogi, D. (2026). SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework for Urban Digital Twins. *Journal of Open Source Software*, 11(118), 9535. https://doi.org/10.21105/joss.09535
- Lindberg, F., Holmer, B., & Thorsson, S. (2008). SOLWEIG 1.0: Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *International Journal of Biometeorology*, 52, 697–713. https://doi.org/10.1007/s00484-008-0162-7
