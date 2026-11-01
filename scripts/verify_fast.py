"""Verify the fast replacements (src/solweig_sens/fast.py) against the original SOLWEIG-GPU code.

Three levels, per configuration:
1. Shadow mode: the original pipeline runs as usual, and every call to the three replaced
   functions is repeated with the fast version on cloned inputs. Every returned array is
   compared, including intermediate fluxes that never reach the output files.
2. End to end: a separate run with all fast versions; every saved output (UTCI, Tmrt,
   shadow, Kup, Kdown, Lup, Ldown) compared with the original run, all pixels and hours.
3. Sun sweep (once): the replaced functions at sun positions the test days never reach:
   low sun, azimuths exactly on the branch boundaries (0, 45, 90, ... degrees), night.

Configurations: Pacoima summer, Pacoima winter (low sun, standard time), Austin (more
trees, other UTM zone and time zone), Pacoima cut into non-square 600-pixel tiles.

Writes figures/profile/verify_fast.json. GPU to itself recommended.

Usage:
    python scripts/verify_fast.py [--configs pacoima_summer pacoima_winter austin tiles600] [--no-sweep]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solweig_sens.analysis import read_stack  # noqa: E402
from solweig_sens.fast import build_fast_functions, fast_solweig, no_tf32  # noqa: E402

LIN = Path.home() / "solweig_verify"
VARS = ("UTCI", "TMRT", "Shadow", "Kup", "Kdown", "Lup", "Ldown")
SAVE = dict(save_tmrt=True, save_shadow=True, save_kup=True, save_kdown=True, save_lup=True, save_ldown=True)
CLONE_LIMIT = 200 * 2**20  # bytes; larger tensors (sky-patch masks) are shared, none of the functions modify them


# ------------------------------------------------------------------ comparisons

def _flat(x):
    if isinstance(x, (tuple, list)):
        for y in x:
            yield from _flat(y)
    else:
        yield x


def compare_results(r1, r2) -> dict:
    import torch

    out = {"identical": True, "max_abs": 0.0, "max_rel": 0.0, "n_arrays": 0}
    a, b = list(_flat(r1)), list(_flat(r2))
    if len(a) != len(b):
        return {"identical": False, "error": f"{len(a)} vs {len(b)} outputs"}
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            out["n_arrays"] += 1
            x, y = x.detach().double(), y.detach().double()
            if x.shape != y.shape:
                return {"identical": False, "error": f"shape {tuple(x.shape)} vs {tuple(y.shape)}"}
            if not torch.equal(torch.isnan(x), torch.isnan(y)):
                return {"identical": False, "error": "NaN pattern differs"}
            m = ~torch.isnan(x)
            if m.any():
                d = (x[m] - y[m]).abs().max().item()
                scale = max(x[m].abs().max().item(), 1e-12)
                out["max_abs"] = max(out["max_abs"], d)
                out["max_rel"] = max(out["max_rel"], d / scale)
                out["identical"] &= d == 0.0
        elif not (isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor)):
            if x != y and not (x is None and y is None):
                out["identical"] = False
        else:
            return {"identical": False, "error": "tensor vs non-tensor output"}
    return out


def merge(acc: dict, r: dict) -> None:
    acc["calls"] += 1
    if "error" in r:
        acc["errors"].append(r["error"])
        acc["identical_calls_all"] = False
        return
    acc["identical_calls"] += int(r["identical"])
    acc["max_abs"] = max(acc["max_abs"], r["max_abs"])
    acc["max_rel"] = max(acc["max_rel"], r["max_rel"])


def new_acc() -> dict:
    return {"calls": 0, "identical_calls": 0, "max_abs": 0.0, "max_rel": 0.0, "errors": []}


def clone_args(args):
    import torch

    def c(x):
        if isinstance(x, torch.Tensor) and x.numel() * x.element_size() < CLONE_LIMIT:
            return x.clone()
        return x
    return [c(x) for x in args], c


# ------------------------------------------------------------------ shadow mode

def shadow_mode_run(out_dir: Path, pre: Path, date: str, fast: dict, captured: dict) -> dict:
    import solweig_gpu.solweig as sw
    from solweig_gpu import run_utci_tiles

    records = defaultdict(new_acc)
    originals = {name: getattr(sw, name) for name in fast}

    def make(name):
        orig, fst = originals[name], fast[name]

        def wrapped(*args, **kw):
            a2, c = clone_args(args)
            k2 = {k: c(v) for k, v in kw.items()}
            r1 = orig(*args, **kw)
            r2 = fst(*a2, **k2)
            merge(records[name], compare_results(r1, r2))
            if name not in captured:  # keep one real call's inputs for the sun sweep
                captured[name] = (clone_args(args)[0], k2)
            return r1
        return wrapped

    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        for name in fast:
            setattr(sw, name, make(name))
        with no_tf32():
            run_utci_tiles(base_path=str(out_dir), preprocess_dir=str(pre), selected_date_str=date, **SAVE)
    finally:
        for name, f in originals.items():
            setattr(sw, name, f)
    return dict(records)


def compare_runs(a: Path, b: Path) -> dict:
    res = {}
    for var in VARS:
        x, y = read_stack(a, var), read_stack(b, var)
        nan_same = bool(np.array_equal(np.isnan(x), np.isnan(y)))
        d = np.abs(x - y)[np.isfinite(x) & np.isfinite(y)]
        res[var] = {"identical": bool(np.array_equal(x, y, equal_nan=True)), "nan_pattern_same": nan_same,
                    "max_abs": float(d.max()) if d.size else 0.0}
    return res


# ------------------------------------------------------------------ sun sweep

def sun_sweep(captured: dict, fast: dict) -> dict:
    import solweig_gpu.solweig as sw
    import torch

    azimuths = [0.0, 1e-4, 44.99999, 45.0, 45.00001, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 359.99]
    out = {}

    def like(template, value):
        if isinstance(template, torch.Tensor):
            return torch.tensor(value, dtype=template.dtype, device=template.device)
        return type(template)(value)

    # Shadow casting: arguments (a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect).
    if "shadowingfunction_wallheight_23" in captured:
        args, kw = captured["shadowingfunction_wallheight_23"]
        acc = new_acc()
        for az in azimuths:
            for alt in (0.5, 2.0, 5.0, 15.0, 45.0, 80.0):
                a = list(args)
                a[3], a[4] = like(args[3], az), like(args[4], alt)
                r1 = sw.shadowingfunction_wallheight_23(*clone_args(a)[0], **kw)
                r2 = fast["shadowingfunction_wallheight_23"](*clone_args(a)[0], **kw)
                merge(acc, compare_results(r1, r2))
        out["shadowingfunction_wallheight_23"] = acc

    # Sky: arguments (solar_altitude, solar_azimuth, ...).
    if "define_patch_characteristics" in captured:
        args, kw = captured["define_patch_characteristics"]
        acc = new_acc()
        for az in azimuths:
            for alt in (-5.0, 0.5, 5.0, 20.0, 45.0, 70.0, 89.0):
                a = list(args)
                a[0], a[1] = like(args[0], alt), like(args[1], az)
                r1 = sw.define_patch_characteristics(*a, **kw)
                r2 = fast["define_patch_characteristics"](*a, **kw)
                merge(acc, compare_results(r1, r2))
        out["define_patch_characteristics"] = acc

    # Ground view factors: first argument is the search azimuth in degrees.
    if "sunonsurface_2018a" in captured:
        args, kw = captured["sunonsurface_2018a"]
        acc = new_acc()
        for az in azimuths:
            a = list(args)
            a[0] = like(args[0], az)
            r1 = sw.sunonsurface_2018a(*clone_args(a)[0], **kw)
            r2 = fast["sunonsurface_2018a"](*clone_args(a)[0], **kw)
            merge(acc, compare_results(r1, r2))
        out["sunonsurface_2018a"] = acc
    return out


# ------------------------------------------------------------------ configurations

def prepare_configs(names: list[str]) -> dict:
    from solweig_sens.run import Setup, prepare

    LIN.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if "pacoima_summer" in names or "pacoima_winter" in names:
        src = ROOT / "runs_la" / "shared" / "processed_inputs"
        dst = LIN / "pacoima_summer"
        if not dst.exists():
            shutil.copytree(src, dst)
        cfg["pacoima_summer"] = (dst, "2019-07-30")
        wdst = LIN / "pacoima_winter"
        if not wdst.exists():
            shutil.copytree(src, wdst)
            for f in (wdst / "metfiles").glob("*.txt"):  # same weather, moved to 21 December
                lines = f.read_text().splitlines()
                rows = [l.split() for l in lines[1:]]
                for r in rows:
                    r[1] = "355"
                f.write_text("\n".join([lines[0]] + [" ".join(r) for r in rows]) + "\n")
        cfg["pacoima_winter"] = (wdst, "2019-12-21")
    if "austin" in names:
        src = ROOT / "runs" / "shared" / "processed_inputs"
        dst = LIN / "austin"
        if not dst.exists():
            shutil.copytree(src, dst)
        cfg["austin"] = (dst, "2020-08-13")
    if "tiles600" in names:
        setup = Setup(raster_dir=(ROOT / "data/la/pacoima").resolve(),
                      met_file=(ROOT / "data/la/met_pacoima_20190730.txt").resolve(),
                      runs_dir=LIN / "tiles600", date="2019-07-30", tile_size=600)
        if not (setup.preprocess_dir / "SVF").exists() or not any((setup.preprocess_dir / "SVF").glob("SkyViewFactor_*")):
            prepare(setup)
        cfg["tiles600"] = (setup.preprocess_dir, "2019-07-30")
    return {k: v for k, v in cfg.items() if k in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=["pacoima_summer", "pacoima_winter", "austin", "tiles600"])
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    import torch
    from solweig_gpu import run_utci_tiles

    torch.zeros(1, device="cuda")
    fast = build_fast_functions()
    report = {"configs": {}}
    captured: dict = {}
    for name, (pre, date) in prepare_configs(args.configs).items():
        print(f"[verify] {name}: shadow mode", flush=True)
        ref_dir, fast_dir = LIN / "out" / name / "original", LIN / "out" / name / "fast"
        t0 = time.perf_counter()
        calls = shadow_mode_run(ref_dir, pre, date, fast, captured if name == "pacoima_summer" else {})
        t1 = time.perf_counter()
        print(f"[verify] {name}: fast run", flush=True)
        shutil.rmtree(fast_dir, ignore_errors=True)
        with fast_solweig():
            run_utci_tiles(base_path=str(fast_dir), preprocess_dir=str(pre), selected_date_str=date, **SAVE)
        t2 = time.perf_counter()
        entry = {"function_calls": calls, "end_to_end": compare_runs(ref_dir, fast_dir),
                 "seconds": {"shadow_mode": t1 - t0, "fast": t2 - t1}}
        if name == "pacoima_summer":  # determinism of the fast path
            again = LIN / "out" / name / "fast_again"
            shutil.rmtree(again, ignore_errors=True)
            with fast_solweig():
                run_utci_tiles(base_path=str(again), preprocess_dir=str(pre), selected_date_str=date, **SAVE)
            entry["fast_repeat_identical"] = all(v["identical"] for v in compare_runs(fast_dir, again).values())
            shutil.rmtree(again, ignore_errors=True)
        report["configs"][name] = entry
        print(json.dumps(entry, indent=1), flush=True)
        shutil.rmtree(LIN / "out" / name, ignore_errors=True)  # outputs are large; the report keeps the numbers

    if not args.no_sweep and captured:
        print("[verify] sun sweep", flush=True)
        report["sun_sweep"] = sun_sweep(captured, fast)
        print(json.dumps(report["sun_sweep"], indent=1), flush=True)

    out = ROOT / "figures" / "profile" / "verify_fast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[verify] report: {out}")


if __name__ == "__main__":
    main()
