"""What a street-scale UTCI forecast costs per forecast cycle, from our measured run times.

A forecast cycle here: one HRRR 48 h forecast (HRRR v4 issues them at 00, 06, 12 and 18 UTC), turned into 48 h of
hourly UTCI at 2 m on every tile of a domain. Each tile is 2 x 2 km (1000 x 1000 px); its sky view factor is
computed once and reused. Three ways to make a 20-member ensemble:

    full       every member run with full physics: 20 runs per tile
    shortcut4  base run, air temperature + 1 K, radiation x 1.1 and x 0.9; members from per-pixel slopes
               (shortcut_suite.py: within 1-15% of the ensemble's own sampling noise): 4 runs per tile
    shortcut14 the same with a radiation table of 13 scales: 14 runs per tile
    single     one deterministic run per tile

Measured (RTX 4070 Laptop GPU, our code C1-C9): a 24 h day on one 1000 x 1000 tile 14.7 s (median of 10 runs,
bench_steps.json); a 48 h forecast counted as two days. Members from slopes: UTCI recomputed per member, measured
here on the GPU when available (else the figure below). Other GPUs are an estimate: the code is memory-bound
(bounds.png), so time is scaled by memory bandwidth, measured 227 GB/s on the laptop against the vendors'
peak figures for the others, times 0.8 for what code reaches in practice. These are not measured.

Domains: 1 venue tile; 30 venue tiles (an Olympic-size set of venues; LA28 lists about 40 venues, some sharing
tiles); the City of Los Angeles (1,302 km2, about 330 tiles); the urbanised part of Los Angeles County (about
4,000 km2, about 1,000 tiles).

Writes figures/forecast_cost.json and prints a table.

Usage:
    python scripts/forecast_cost.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY_S = None  # filled from bench_steps
LAPTOP_BW = 227e9
GPUS = {"RTX 4070 Laptop (measured)": None, "A100 80 GB (est.)": 2.0e12, "H100 SXM (est.)": 3.35e12}
EFF = 0.8
DOMAINS = {"1 venue tile": 1, "30 venue tiles": 30, "City of Los Angeles, ~330 tiles": 330,
           "urban LA County, ~1,000 tiles": 1000}
METHODS = {"single": 1, "shortcut4": 4, "shortcut14": 14, "full": 20}
MEMBERS = 20
UTCI_PER_MEMBER_DAY_S = 0.5  # 24 UTCI evaluations of 1 M pixels on the GPU, a generous figure; measured below if possible


def measure_utci() -> float | None:
    try:
        import numpy as np
        import torch
        from solweig_gpu.calculate_utci import utci_calculator

        if not torch.cuda.is_available():
            return None
        x = torch.full((1000, 1000), 30.0, device="cuda")
        for _ in range(3):
            utci_calculator(x, x + 10, x + 20, x * 0 + 2)
        torch.cuda.synchronize()
        import time

        t = time.perf_counter()
        for _ in range(48):
            utci_calculator(x, x + 10, x + 20, x * 0 + 2)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / 2  # per 24 h
    except Exception:
        return None


def main() -> None:
    runs = []
    for name in ("bench_steps.json", "bench_steps_b.json"):
        rep = json.loads((ROOT / "figures" / "profile" / name).read_text())
        runs += [r["seconds"] for r in rep["runs"] if r["step"] == "I" and r["repeat"] > 0]
    day_s = statistics.median(runs)
    u = measure_utci()
    utci_s = u if u is not None else UTCI_PER_MEMBER_DAY_S
    rows = []
    for gpu, bw in GPUS.items():
        speed = 1.0 if bw is None else bw * EFF / LAPTOP_BW
        for dom, tiles in DOMAINS.items():
            for m, n_runs in METHODS.items():
                physics = tiles * n_runs * 2 * day_s / speed
                members = 0.0 if m in ("single", "full") else tiles * MEMBERS * 2 * utci_s / speed
                total = physics + members
                rows.append({"gpu": gpu, "domain": dom, "tiles": tiles, "method": m, "physics_runs_per_tile": n_runs,
                             "seconds_per_cycle_one_gpu": round(total, 1), "gpus_to_finish_in_1h": max(1, -(-int(total) // 3600)),
                             "fits_6h_cycle_on_one_gpu": total <= 6 * 3600})
    rep = {"day_seconds_1000px": day_s, "day_runs": len(runs), "utci_member_day_seconds": utci_s,
           "utci_measured": u is not None, "bandwidth_efficiency": EFF, "laptop_bandwidth": LAPTOP_BW,
           "gpus_peak_bandwidth": {k: v for k, v in GPUS.items() if v}, "domains": DOMAINS, "rows": rows}
    (ROOT / "figures" / "forecast_cost.json").write_text(json.dumps(rep, indent=1))
    print(f"day on one tile {day_s:.1f} s (n = {len(runs)}); UTCI per member-day {utci_s:.2f} s "
          f"({'measured' if u is not None else 'assumed'})")
    for gpu in GPUS:
        print(f"\n{gpu}: time per 48 h cycle on one GPU (h:mm), and GPUs to finish within 1 h")
        print(f"{'domain':34s}" + "".join(f"{m:>18s}" for m in METHODS))
        for dom in DOMAINS:
            cells = []
            for m in METHODS:
                r = next(x for x in rows if x["gpu"] == gpu and x["domain"] == dom and x["method"] == m)
                s = r["seconds_per_cycle_one_gpu"]
                cells.append(f"{int(s // 3600)}:{int(s % 3600 // 60):02d} ({r['gpus_to_finish_in_1h']} GPU)")
            print(f"{dom:34s}" + "".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
