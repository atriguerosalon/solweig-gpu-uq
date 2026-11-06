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
D = ROOT / "figures" / "shortcuts"
LABELS = {"expo_0730_independent": "venue, 30 Jul 2019", "expo_0730_correlated": "venue, 30 Jul 2019, correlated errors",
          "expo_0615_independent": "venue, 15 Jun 2019, morning cloud", "expo_0906_independent": "venue, 6 Sep 2020, heatwave",
          "expo_0906_independent_perez_published": "venue, 6 Sep 2020, heatwave, Perez clearness as published",
          "pacoima_0730_independent": "Pacoima, 30 Jul 2019",
          "pacoima_0730_independent_fitted": "Pacoima, 30 Jul 2019, fitted ground model"}  # only these are plotted
PRODUCTS = {"p_4h_above_32": ("o", "P(UTCI > 32 °C for 4 h or more)"), "p_4h_above_38": ("s", "P(UTCI > 38 °C for 4 h or more)"),
            "p_above_38_at_15": ("^", "P(UTCI > 38 °C at 15:00)"), "p_above_46_at_15": ("D", "P(UTCI > 46 °C at 15:00)")}
SHORT = {"b": "base run\nonly", "c": "+ air temperature\nslope", "d": "+ radiation\nslope", "e": "+ radiation slope\nboth sides",
         "f": "radiation\ntable"}
MIN_NOISE = 0.002


def main() -> None:
    from solweig_sens import plotstyle

    files = sorted(D.glob("shortcut_suite_*.json"))
    data = {f.stem.replace("shortcut_suite_", ""): json.loads(f.read_text()) for f in files}
    data = {k: v for k, v in data.items() if v.get("members", 0) >= 20 and k in LABELS}
    plotstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.4, 1]})
    cmap = plt.get_cmap("tab10")
    keys = [k for k in LABELS if k in data]
    summary, ratios = {}, {n: [] for n in "bcdef"}
    for i, key in enumerate(keys):  # left: every (setting, product) pair as a dot, one column per shortcut
        d, c = data[key], cmap(i)
        summary[key] = {}
        for q, (mk, _) in PRODUCTS.items():
            noise = d["noise_A_vs_B"][q]["mean_abs_dp"]
            if noise < MIN_NOISE:
                continue
            r = [d["shortcuts"][n][q]["mean_abs_dp"] / noise for n in "bcdef"]
            summary[key][q] = r
            for j, (n, v) in enumerate(zip("bcdef", r)):
                ratios[n].append(v)
                axes[0].scatter(j + (i - (len(keys) - 1) / 2) * 0.07, v, marker=mk, s=26, color=c, edgecolor="white",
                                linewidth=0.4, zorder=3)
        runs = [d["shortcuts"][n]["physics_runs"] for n in "bcdef"]
        axes[1].plot(range(5), [d["shortcuts"][n]["member_utci_error"]["max_abs"] for n in "bcdef"], "o-", color=c, lw=1.4,
                     ms=4, label=LABELS.get(key, key))
        axes[1].plot(range(5), [d["shortcuts"][n]["member_utci_error"]["p99_abs"] for n in "bcdef"], "o--", color=c, lw=1,
                     ms=3, alpha=0.7)
    for j, n in enumerate("bcdef"):  # median over all pairs
        m = float(np.median(ratios[n]))
        axes[0].plot([j - 0.32, j + 0.32], [m, m], color="k", lw=2, zorder=4)
        axes[0].text(j + 0.36, m, f"{m:.2g}", va="center", fontsize=8,
                     bbox=dict(fc="white", ec="none", pad=0.5))
    lo = {n: min(ratios[n]) for n in "bcdef"}
    hi = {n: max(ratios[n]) for n in "bcdef"}
    axes[0].axhline(1, color="0.3", lw=1, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ticks = [f"{runs[j]} run{'s' if runs[j] > 1 else ''}\n{SHORT[n]}" for j, n in enumerate("bcdef")]
    axes[0].set_xticks(range(5), ticks, fontsize=8.5)
    axes[0].set_xlim(-0.5, 4.6)
    axes[0].set_ylabel("error in ensemble probabilities\nrelative to the noise of 20 members")
    axes[0].grid(alpha=0.3, axis="y", which="both")
    for q, (mk, lab) in PRODUCTS.items():
        axes[0].scatter([], [], marker=mk, color="0.4", label=lab)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].set_xticks(range(5), [f"{runs[j]} run{'s' if runs[j] > 1 else ''}\n{SHORT[n]}" for j, n in enumerate("bcdef")],
                       fontsize=8)
    axes[1].set_ylabel("single-member UTCI error (°C)")
    styles = [plt.Line2D([], [], color="0.3", lw=1.4, marker="o", ms=4, label="largest"),
              plt.Line2D([], [], color="0.3", lw=1, ls="--", marker="o", ms=3, label="99th percentile")]
    axes[1].add_artist(axes[1].legend(handles=styles, frameon=False, fontsize=8, loc="lower left"))
    axes[1].set_yscale("log")
    axes[1].set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    axes[1].grid(alpha=0.3, which="both")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.1, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 1))  # caption: docs/FIGURES.md
    fig.savefig(D / "shortcuts_vs_noise.png", dpi=150)
    plt.close(fig)
    (D / "shortcuts_vs_noise.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {D / 'shortcuts_vs_noise.png'} ({len(data)} settings)")


if __name__ == "__main__":
    main()
