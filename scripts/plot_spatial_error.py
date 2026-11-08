"""Figure for spatial_error_fss.py: how street-scale skill depends on how the air-temperature error varies in space.

    a  fractions skill score at 2 m (single forecast) against the error's correlation length, at sigma = 2 K, for
       four events: UTCI above 32 C for 4 h or more, the hottest 10% and coolest 10% of outdoor pixels at 15:00, and
       the hotter half; solid the venue, dashed Pacoima. "uniform" is one error for the whole tile.
    b  correlation between members' UTCI anomaly maps at 15:00 (tile mean removed) against the correlation length,
       for sigma = 1, 2 and 3 K, the venue
    c  with errors like HRRR's (figures/hrrr/hrrr_errors.json): at each lead the measured size, a share c0 uniform
       over the tile and the rest local; line: local length 1 km, band: 250 m to 3 km (stations cannot tell)

No title or caption in the image; the caption is in docs/FIGURES.md.

Usage:
    python scripts/plot_spatial_error.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "figures" / "spatial_error"
EVENTS = {"event_32_4h": ("UTCI > 32 °C for 4 h or more", "#7a5aa6"), "event_top10": ("hottest 10% at 15:00", "#c4501b"),
          "event_cool10": ("coolest 10% at 15:00", "#0e6a78"), "event_top50": ("hotter half at 15:00", "#4c78a8")}


def load(tile):
    f = D / f"spatial_error_fss_{tile}.json"
    return json.loads(f.read_text())["results"] if f.exists() else []


def xpos(L, lengths):
    return lengths.index(L)


def main() -> None:
    from solweig_sens import plotstyle

    plotstyle.use()
    res = {t: load(t) for t in ("expo", "pacoima")}
    lengths = [250.0, 1000.0, 3000.0, 10000.0, "uniform"]
    labels = ["250 m", "1 km", "3 km", "10 km", "uniform"]
    fig, (a, b, cp) = plt.subplots(1, 3, figsize=(16, 4))
    for tile, ls, mk in (("expo", "-", "o"), ("pacoima", "--", "s")):
        for ev, (lab, c) in EVENTS.items():
            pts = sorted((xpos(r["L_m"], lengths), r[ev]["fss_member"][0]) for r in res[tile]
                         if r["sigma"] == 2.0 and ev in r and r["L_m"] in lengths)
            if pts:
                x, y = zip(*pts)
                a.plot(x, y, ls, marker=mk, color=c, mfc="white" if tile == "pacoima" else c,
                       label=lab if tile == "expo" else None)
    a.plot([], [], "k-", marker="o", label="venue")
    a.plot([], [], "k--", marker="s", mfc="white", label="Pacoima")
    a.set_xticks(range(len(lengths)), labels)
    a.set_xlabel("correlation length of the air-temperature error")
    a.set_ylabel("fractions skill score at 2 m")
    a.set_ylim(0, 1.02)
    a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    for s, c in ((1.0, "#9ecae1"), (2.0, "#4c78a8"), (3.0, "#08306b")):
        pts = sorted((xpos(r["L_m"], lengths), r["anomaly_correlation_mean"]) for r in res["expo"]
                     if r["sigma"] == s and "anomaly_correlation_mean" in r and r["L_m"] in lengths)
        if pts:
            x, y = zip(*pts)
            b.plot(x, y, "o-", color=c, label=f"{s:g} K")
    b.set_xticks(range(len(lengths)), labels)
    b.set_xlabel("correlation length of the air-temperature error")
    b.set_ylabel("anomaly-map correlation")
    b.set_ylim(0, 1.02)
    b.legend(frameon=False, fontsize=8, title="error, standard deviation", title_fontsize=8, loc="lower right")
    leads = (("0", 0), ("4", 4), ("22", 22), ("46", 46))
    for tile, ls, mk in (("expo", "-", "o"), ("pacoima", "--", "s")):
        for ev in ("event_top10", "event_cool10", "event_top50"):
            lab, col = EVENTS[ev]
            xs, mid, lo, hi = [], [], [], []
            for tag, L in leads:
                f = D / f"spatial_error_fss_{tile}_hrrr_lead{tag}.json"
                if not f.exists():
                    continue
                v = {r["Ls_m"]: r[ev]["fss_member"][0] for r in json.loads(f.read_text())["results"]}
                xs.append(L)
                mid.append(v.get(1000.0, np.nan))
                lo.append(min(v.values()))
                hi.append(max(v.values()))
            if xs:
                cp.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
                cp.plot(xs, mid, ls, marker=mk, color=col, mfc="white" if tile == "pacoima" else col,
                       label=lab if tile == "expo" else None)
    cp.set_xlabel("HRRR lead time (h)")
    cp.set_ylabel("fractions skill score at 2 m")
    cp.set_ylim(0, 1.02)
    cp.legend(frameon=False, fontsize=8, loc="lower left")
    for ax, letter in ((a, "a"), (b, "b"), (cp, "c")):
        ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = D / "spatial_error_fss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
