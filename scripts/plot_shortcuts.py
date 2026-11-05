"""How many physics runs does a forecast ensemble need? Shortcut error against sampling noise, all settings.

Reads figures/shortcuts/shortcut_suite_*.json (shortcut_suite.py). For each setting (venue, day,
error model) and each ensemble product that the day actually reaches, the shortcut's mean |dp|
against full physics divided by the mean |dp| between two independent full-physics ensembles
(the sampling noise of 20 members), against the number of physics runs the shortcut needs. Below 1
the shortcut's error is smaller than the ensemble's own noise. Products with no noise (never or
always reached: mean |dp| between ensembles below 0.002) are left out. Second panel: the largest
single-member UTCI error, the failure mode the ratios hide.

Usage:
    python scripts/plot_shortcuts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "figures" / "shortcuts"
LABELS = {"expo_0730_independent": "venue, 30 Jul 2019 (clear)", "expo_0730_correlated": "venue, 30 Jul, correlated errors",
          "expo_0615_independent": "venue, 15 Jun 2019 (morning cloud)", "expo_0906_independent": "venue, 6 Sep 2020 (heatwave)",
          "pacoima_0730_independent": "Pacoima, 30 Jul 2019 (clear)"}
PRODUCTS = {"p_4h_above_32": ("o", "P(>32 C for 4 h)"), "p_4h_above_38": ("s", "P(>38 C for 4 h)"),
            "p_above_38_at_15": ("^", "P(>38 C at 15 h)"), "p_above_46_at_15": ("D", "P(>46 C at 15 h)")}
SHORT = {"b": "base", "c": "+Ta", "d": "+rad.", "e": "2-sided", "f": "table"}
MIN_NOISE = 0.002


def main() -> None:
    from solweig_sens import plotstyle

    files = sorted(D.glob("shortcut_suite_*.json"))
    data = {f.stem.replace("shortcut_suite_", ""): json.loads(f.read_text()) for f in files}
    data = {k: v for k, v in data.items() if v.get("members", 0) >= 20}
    plotstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    cmap = plt.get_cmap("tab10")
    summary = {}
    for i, (key, d) in enumerate(sorted(data.items())):
        c = cmap(i)
        runs = [d["shortcuts"][n]["physics_runs"] for n in "bcdef"]
        summary[key] = {}
        for q, (mk, _) in PRODUCTS.items():
            noise = d["noise_A_vs_B"][q]["mean_abs_dp"]
            if noise < MIN_NOISE:
                continue
            r = [d["shortcuts"][n][q]["mean_abs_dp"] / noise for n in "bcdef"]
            summary[key][q] = r
            axes[0].plot(runs, r, mk + "-", color=c, ms=5, lw=1.2, alpha=0.9)
        axes[1].plot(runs, [d["shortcuts"][n]["member_utci_error"]["max_abs"] for n in "bcdef"], "o-", color=c, lw=1.4,
                     label=LABELS.get(key, key))
        axes[1].plot(runs, [d["shortcuts"][n]["member_utci_error"]["p99_abs"] for n in "bcdef"], "o--", color=c, lw=1,
                     alpha=0.7)
    axes[0].axhline(1, color="0.3", lw=1, ls=":")
    axes[0].text(14.5, 1.05, "as large as the ensemble's own sampling noise", ha="right", va="bottom", fontsize=8, color="0.3")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    xt = [1, 2, 3, 4, 14]
    axes[0].set_xticks(xt, [f"{x}\n{SHORT[n]}" for x, n in zip(xt, "bcdef")], fontsize=8)
    axes[0].minorticks_off()
    axes[0].set_xlabel("physics runs per ensemble (any number of members)")
    axes[0].set_ylabel("shortcut error / sampling noise\n(mean |dp|, 20 members)")
    axes[0].set_title("Ensemble products: radiation shortcuts reach 1-30% of the sampling noise", loc="left",
                      fontsize=10, fontweight="bold")
    axes[0].grid(alpha=0.3, which="both")
    for q, (mk, lab) in PRODUCTS.items():
        axes[0].plot([], [], mk, color="0.4", label=lab)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", title="product (colour: setting)", title_fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xticks(xt, [str(x) for x in xt])
    axes[1].minorticks_off()
    axes[1].set_xlabel("physics runs per ensemble")
    axes[1].set_ylabel("member UTCI error vs full physics (C)")
    axes[1].set_title("Single members: largest error (solid), 99th percentile (dashed)", loc="left", fontsize=10,
                      fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.text(0.01, 0.005, "SOLWEIG-GPU's own physics, our code C1-C9, 2 x 2 km at 2 m, 20 members with AR(1) errors in Ta "
             "(1.5 K), wind (30%), global radiation (10%). Noise: two independent full-physics ensembles. Products never or "
             "always reached on a day are left out. shortcut_suite.py; figures/shortcuts/*.json.", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(D / "shortcuts_vs_noise.png", dpi=150)
    plt.close(fig)
    (D / "shortcuts_vs_noise.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {D / 'shortcuts_vs_noise.png'} ({len(data)} settings)")


if __name__ == "__main__":
    main()
