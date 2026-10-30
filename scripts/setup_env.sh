#!/usr/bin/env bash
# One-time environment setup inside WSL Ubuntu (user-level, no sudo).
set -euo pipefail
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
BIN="$HOME/.local/bin"
mkdir -p "$BIN"
if [ ! -x "$BIN/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$HOME/.local" bin/micromamba
fi
MM="$BIN/micromamba"
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/solweig" ]; then
  "$MM" create -y -n solweig -c conda-forge python=3.11 gdal numpy scipy pandas matplotlib xarray netcdf4 pip
fi
"$MM" run -n solweig pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
"$MM" run -n solweig pip install --quiet solweig-gpu==2.0.0 SALib
"$MM" run -n solweig python -c "import torch, solweig_gpu; from osgeo import gdal; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('gdal', gdal.__version__)"
