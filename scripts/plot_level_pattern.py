"""Level against pattern: what an air-temperature forecast error does to a street-scale heat-stress map.

From the perfect-model ensembles of run_la_forecast_error.py (20 members per error size, AR(1) errors
in time, one error per hour for the whole tile, as SOLWEIG takes air temperature), analysed by
analyse_la_forecast_error.py, for Pacoima and the Exposition Park venue tile:

    a  UTCI change at 15:00 against the unperturbed run, split into its tile mean (the level) and
       its standard deviation across outdoor pixels (the pattern), mean over members
    b  share of outdoor area with UTCI above 32 C at 15:00, mean and standard deviation over members

No title or caption in the image; the caption is in docs/FIGURES.md.

Usage:
    python scripts/plot_level_pattern.py
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
SITES = (("Pacoima", ROOT / "figures" / "la_forecast_error.json", "#0e6a78"),
         ("Exposition Park", ROOT / "figures" / "expo" / "la_forecast_error.json", "#c4501b"))


def main() -> None:
    from solweig_sens import plotstyle

    plotstyle.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    for name, path, c in SITES:
        d = json.loads(path.read_text())
        sig = np.array(sorted(float(s) for s in d["sigma"]))
        r = [d["sigma"][f"{s:g}" if f"{s:g}" in d["sigma"] else str(s)] for s in sig]
        level = [x["utci_15h_level_shift_abs_mean"] for x in r]
        spread = [x["utci_15h_pattern_spread_mean"] for x in r]
        a.plot(sig, level, "o-", color=c, label=f"{name}, tile mean")
        a.plot(sig, spread, "s--", color=c, mfc="white", label=f"{name}, across pixels")
        m, sd = np.array([x["area_above_32_pct_15h_mean_sd"] for x in r]).T
        b.fill_between(sig, np.clip(m - sd, 0, 100), np.clip(m + sd, 0, 100), color=c, alpha=0.15, lw=0)
        b.plot(sig, m, "o-", color=c, label=name)
    a.set_yscale("log")
    a.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    a.minorticks_off()
    a.set_ylim(0.015, 5)
    a.set_xlabel("air-temperature error, standard deviation (K)")
    a.set_ylabel("UTCI change at 15:00 (°C)")
    a.legend(frameon=False, fontsize=8)
    b.set_ylim(0, 100)
    b.set_xlabel("air-temperature error, standard deviation (K)")
    b.set_ylabel("outdoor area with UTCI > 32 °C at 15:00 (%)")
    b.legend(frameon=False, fontsize=8, loc="lower left")
    for ax, letter in ((a, "a"), (b, "b")):
        ax.text(-0.13, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = ROOT / "figures" / "la_level_vs_pattern.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
