"""Does SOLWEIG-GPU v2.0.0's svf_calculator drop sky patches through float32 rounding?

Reproduces its azimuth loop: the number of patches per altitude ring is computed as
int(360 / skyvaultaziint[j]) with skyvaultaziint a float32 tensor of 360 / patches. If that
truncates below the intended count, patches are dropped and later azimuths shift (reported in
the iadicarlo fork, commit 5932ef9, and related to draft PR #137).

Usage:
    python scripts/check_patch_count.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import torch
from solweig_gpu.shadow import create_patches


def main() -> None:
    _, _, _, skyvaultaltint, aziinterval, _, azistart = create_patches(2)
    step32 = torch.tensor([360 / p for p in aziinterval])  # float32, as in svf_calculator
    intended = [int(p) for p in aziinterval]
    used = [int(360 / step32[j]) for j in range(len(intended))]
    print("ring  altitude  intended  used  dropped")
    for j, (a, b) in enumerate(zip(intended, used)):
        print(f"{j:4d}  {float(skyvaultaltint[j]):8.1f}  {a:8d}  {b:4d}  {a - b:7d}")
    print(f"total patches: intended {sum(intended)}, azimuths actually generated {sum(used)}")


if __name__ == "__main__":
    main()
