"""Validate the NOAA solar engine against pvlib's NREL SPA (BRD O3: within ~1 deg).

Checks every 20 minutes across a full year at points along the corridor and
writes ``solar_validation.json`` plus Kotlin unit-test fixtures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .solar import solar_position

SITES = {
    "Manali": (32.2396, 77.1887, 2050),
    "Keylong": (32.5711, 77.0322, 3080),
    "Sarchu": (32.9100, 77.5800, 4290),
    "Tanglang La": (33.5080, 77.7700, 5328),
    "Leh": (34.1526, 77.5771, 3500),
}


def main(argv: list[str] | None = None) -> None:
    import pandas as pd
    import pvlib

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    times = pd.date_range(f"{args.year}-01-01", f"{args.year}-12-31 23:40", freq="20min", tz="UTC")
    unix = times.as_unit("s").asi8.astype(np.float64)
    report = {"reference": f"pvlib {pvlib.__version__} NREL SPA (apparent elevation)", "sites": {}}
    worst_el = worst_az = 0.0
    for name, (lat, lon, alt) in SITES.items():
        ref = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=alt, method="nrel_numpy")
        el, az = solar_position(lat, lon, unix)
        up = ref["apparent_elevation"].to_numpy() > 0
        d_el = np.abs(el - ref["apparent_elevation"].to_numpy())[up]
        d_az = np.abs((az - ref["azimuth"].to_numpy() + 180.0) % 360.0 - 180.0)[up]
        report["sites"][name] = {
            "daylight_samples": int(up.sum()),
            "max_elevation_error_deg": float(d_el.max()),
            "mean_elevation_error_deg": float(d_el.mean()),
            "max_azimuth_error_deg": float(d_az.max()),
            "mean_azimuth_error_deg": float(d_az.mean()),
        }
        worst_el, worst_az = max(worst_el, d_el.max()), max(worst_az, d_az.max())

    report["max_elevation_error_deg"] = float(worst_el)
    report["max_azimuth_error_deg"] = float(worst_az)
    report["within_1_deg"] = bool(worst_el < 1.0 and worst_az < 1.0)
    (out / "solar_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Fixtures for SolarPositionTest.kt (reference = pvlib, so Kotlin is checked against SPA too).
    fixture_times = pd.to_datetime([
        "2026-01-15 04:30", "2026-01-15 06:30", "2026-03-20 07:00",
        "2026-06-21 02:00", "2026-06-21 06:30", "2026-10-08 10:15", "2026-12-21 05:00",
    ]).tz_localize("UTC")
    rows = []
    for name, (lat, lon, alt) in SITES.items():
        ref = pvlib.solarposition.get_solarposition(fixture_times, lat, lon, altitude=alt, method="nrel_numpy")
        for ts, (_, r) in zip(fixture_times, ref.iterrows()):
            rows.append(f"Fixture({lat}, {lon}, {int(ts.timestamp())}L, "
                        f"{r['apparent_elevation']:.4f}, {r['azimuth']:.4f}), // {name} {ts:%Y-%m-%d %H:%M}Z")
    (out / "solar_fixtures.kt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
