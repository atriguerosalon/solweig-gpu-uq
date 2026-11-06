"""Which is closer to the exact sky longwave sums: the original sequential float32 loop or the fast chunked one?

Captures the float32 inputs that SOLWEIG's Lcyl_v2022a passes to define_patch_characteristics on a
synthetic tile, then evaluates that one function four ways on those same inputs:
  - original and fast, in float32 (as in production)
  - original and fast, in float64 (inputs cast up; every internal array float64), as the reference
The two float64 results must agree closely; each float32 result is scored against them.
CPU only.

Usage:
    python scripts/sky_rounding_check.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NAMES = ["Ldown", "Lside", "Lside_sky", "Lside_veg", "Lside_sh", "Lside_sun", "Lside_ref",
         "Least", "Lwest", "Lnorth", "Lsouth"]


def lcyl_inputs(n=64, seed=2):
    from solweig_gpu.shadow import create_patches

    g = torch.Generator().manual_seed(seed)
    skyalt, skyazi, *_ = create_patches(2)
    p = len(skyalt)
    patches = torch.stack([torch.as_tensor(skyalt).float().flatten(), torch.as_tensor(skyazi).float().flatten(),
                           torch.zeros(p)], 1)
    masks = [(torch.rand((n, n, p), generator=g) > q).float() for q in (0.3, 0.2, 0.2)]
    asvf = torch.acos(torch.sqrt(0.2 + 0.8 * torch.rand((n, n), generator=g)))
    lup = 400 + 50 * torch.rand((n, n), generator=g)
    return (torch.tensor(0.85), patches, torch.tensor(28.0), torch.tensor(7.0), 0.9, lup, *masks,
            torch.tensor(35.0, dtype=torch.float64), torch.tensor(200.0, dtype=torch.float64), n, n, asvf)


def capture_define_args():
    import solweig_gpu.solweig as sw

    captured = {}
    original = sw.define_patch_characteristics

    def grab(*args, **kw):
        captured["args"], captured["kw"] = args, kw
        return original(*args, **kw)

    sw.define_patch_characteristics = grab
    try:
        sw.Lcyl_v2022a(*lcyl_inputs())
    finally:
        sw.define_patch_characteristics = original
    return captured["args"], captured["kw"]


def up64(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.double()
    return x


def evaluate(fn, args, kw, dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        a = [up64(x) for x in args] if dtype == torch.float64 else list(args)
        return [x.double() for x in fn(*a, **kw)]
    finally:
        torch.set_default_dtype(old)


def main() -> None:
    import solweig_gpu.solweig as sw
    from solweig_sens.fast import build_fast_functions

    args, kw = capture_define_args()
    original, fast = sw.define_patch_characteristics, build_fast_functions()["define_patch_characteristics"]
    ref_o = evaluate(original, args, kw, torch.float64)
    ref_f = evaluate(fast, args, kw, torch.float64)
    print(f"float64 reference: original and fast agree to {max(float((a - b).abs().max()) for a, b in zip(ref_o, ref_f)):.1e} W/m2")
    o32 = evaluate(original, args, kw, torch.float32)
    f32 = evaluate(fast, args, kw, torch.float32)
    print(f"float32 original vs float32 fast: {max(float((a - b).abs().max()) for a, b in zip(o32, f32)):.1e} W/m2")
    for label, out in (("original float32 (sequential loop)", o32), ("fast float32 (chunked products)", f32)):
        errs = [float((x - r).abs().max()) for x, r in zip(out, ref_o)]
        mean = [float((x - r).abs().mean()) for x, r in zip(out, ref_o)]
        print(f"{label}: largest error vs float64 {max(errs):.1e} W/m2, mean {sum(mean) / len(mean):.1e} W/m2")
        print("   " + ", ".join(f"{n} {e:.1e}" for n, e in zip(NAMES, errs)))


if __name__ == "__main__":
    main()
