"""UMEP met file for one LA day from an airport station (IEM ASOS archive) and ERA5 radiation.

Air temperature, humidity and wind come from the station; global radiation and surface
pressure from ERA5 through the Open-Meteo archive API (no key needed). The observation at
hh:5x is used for hour hh+1. Diffuse and direct radiation are left for SOLWEIG to estimate.

Usage:
    python scripts/make_la_met.py --date 2019-07-30 --station VNY --lat 34.2735 --lon -118.41 \
        --out data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_la_domain import fetch  # noqa: E402

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
HEADER = ("%iy  id  it imin   Q*      QH      QE      Qs      Qf    Wind    RH     Td     press   rain    Kdn    "
          "snow    ldown   fcld    wuh     xsmd    lai_hr  Kdiff   Kdir    Wd")
KNOT = 0.514444


def station_hours(station: str, day: date) -> dict[int, dict]:
    """Last valid report in each clock hour, keyed by the following hour (0-23)."""
    prev, end = day - timedelta(days=1), day + timedelta(days=1)  # IEM's end date is exclusive
    params = {"station": station, "data": ["tmpc", "relh", "sknt", "drct"], "tz": "America/Los_Angeles",
              "format": "onlycomma", "latlon": "no", "missing": "M", "trace": "T", "direct": "no",
              "report_type": "3", "year1": prev.year, "month1": prev.month, "day1": prev.day,
              "year2": end.year, "month2": end.month, "day2": end.day}
    query = "&".join(f"{k}={v}" for k, vs in params.items() for v in (vs if isinstance(vs, list) else [vs]))
    rows = csv.DictReader(io.StringIO(fetch(f"{IEM}?{query}").decode()))
    out = {}
    for r in rows:
        t = r["valid"]  # "YYYY-MM-DD HH:MM", local time
        if "M" in (r["tmpc"], r["relh"], r["sknt"]):
            continue
        d, hh = date.fromisoformat(t[:10]), int(t[11:13])
        target = hh + 1 if d == prev else hh + 1 + 24  # hour index relative to the previous day
        if 24 <= target <= 47:
            out[target - 24] = {"ta": float(r["tmpc"]), "rh": float(r["relh"]),
                                "wind": float(r["sknt"]) * KNOT, "wdir": float(r["drct"]) if r["drct"] != "M" else -999.0}
    return out


def era5_hours(lat: float, lon: float, day: date) -> dict:
    d = json.loads(fetch(OPEN_METEO, {
        "latitude": lat, "longitude": lon, "start_date": day.isoformat(), "end_date": day.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,shortwave_radiation",
        "timezone": "America/Los_Angeles", "wind_speed_unit": "ms"}))
    return d["hourly"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--station", default="VNY")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    obs = station_hours(args.station, args.date)
    era = era5_hours(args.lat, args.lon, args.date)
    missing = [h for h in range(24) if h not in obs]
    if missing:  # fill short gaps by linear interpolation between neighbouring hours
        have = sorted(obs)
        for h in missing:
            for k in ("ta", "rh", "wind"):
                obs.setdefault(h, {})[k] = float(np.interp(h, have, [obs[i][k] for i in have]))
            obs[h]["wdir"] = -999.0
        print(f"[met] interpolated station hours: {missing}")

    doy = args.date.timetuple().tm_yday
    rows = []
    for h in range(24):
        o = obs[h]
        row = [args.date.year, doy, h, 0] + [-999.0] * 5 + [
            max(o["wind"], 0.1), o["rh"], o["ta"], era["surface_pressure"][h] / 10.0, 0.0,
            era["shortwave_radiation"][h], -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, o["wdir"]]
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(" ".join(f"{v:d}" if isinstance(v, int) else f"{v:.2f}" for v in r) + "\n")

    print("hour  Ta_stn  Ta_ERA5  RH_stn  RH_ERA5  wind_stn  wind_ERA5  Kdown")
    for h in range(24):
        print(f"{h:4d} {obs[h]['ta']:7.1f} {era['temperature_2m'][h]:8.1f} {obs[h]['rh']:7.0f} "
              f"{era['relative_humidity_2m'][h]:8.0f} {obs[h]['wind']:9.1f} {era['wind_speed_10m'][h]:10.1f} "
              f"{era['shortwave_radiation'][h]:6.0f}")
    ta_s = np.array([obs[h]["ta"] for h in range(24)])
    ta_e = np.array(era["temperature_2m"])
    print(f"ERA5 minus station Ta: mean {np.mean(ta_e - ta_s):+.2f} K, max {ta_s.max():.1f} (stn) "
          f"vs {ta_e.max():.1f} (ERA5), min {ta_s.min():.1f} vs {ta_e.min():.1f}")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
