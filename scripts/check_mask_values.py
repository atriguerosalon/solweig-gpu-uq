"""Are the sky-patch masks really only 0 and 1 on real tiles? Checks every value of every cached mask.

Usage:
    python scripts/check_mask_values.py npz [npz ...]
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    for path in sys.argv[1:]:
        with np.load(path) as npz:
            for key in npz.files:
                a = npz[key]
                vals, counts = np.unique(a, return_counts=True)
                other = {float(v): int(c) for v, c in zip(vals, counts) if v not in (0.0, 1.0)}
                print(f"{path.split('/')[-3]:>22s} {key:13s} shape {a.shape} dtype {a.dtype} "
                      f"values {[float(v) for v in vals]} non-0/1 entries {other or 'none'}")
                if other:
                    idx = np.argwhere((a != 0) & (a != 1))
                    patches, pc = np.unique(idx[:, 2], return_counts=True)
                    print(f"{'':>22s}   in patches {dict(zip(patches.tolist(), pc.tolist()))}")


if __name__ == "__main__":
    main()
