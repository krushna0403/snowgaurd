"""Build the offline Manali-Leh route database bundled in the Android app.

Pipeline: OSM route -> resample every ``--spacing`` m -> Copernicus DEM
elevation, slope/aspect and 72-direction horizon angles -> SQLite.

Outputs (in ``--out``):
    route_manali_leh.db       SQLite route database (also copied to the app assets)
    route_manali_leh.gpx      whole route, for emulator GPS playback
    route_demo_high.gpx       ~35 km high-altitude stretch, quicker to replay
    route_report.json         build summary and sanity statistics
    route_profile.png         elevation profile with shade on a winter morning
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .dem import Dem, download_tiles, tile_names
from .route import build_route, landmarks
from .solar import solar_position

DB_NAME = "route_manali_leh.db"
SCHEMA_VERSION = 1
N_AZIMUTH = 72
HORIZON_SCALE = 10  # stored as int16 tenths of a degree

# FR-T3 altitude-month freeze thresholds.
FREEZE_THRESHOLDS = [
    {"season": "winter", "months": [11, 12, 1, 2, 3], "min_elev_m": 3000},
    {"season": "shoulder", "months": [4, 5, 10], "min_elev_m": 4000},
    {"season": "summer", "months": [6, 7, 8, 9], "min_elev_m": 4800},
]


def fill_tunnels(elev: np.ndarray, tunnel: np.ndarray) -> np.ndarray:
    """Inside tunnels the DEM is the mountain above; interpolate road level between portals."""
    elev = elev.copy()
    idx = np.arange(len(elev))
    if tunnel.any() and (~tunnel).any():
        elev[tunnel] = np.interp(idx[tunnel], idx[~tunnel], elev[~tunnel])
    return elev


def write_db(path: Path, route, elev, slope, aspect, horizon, marks, meta: dict) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY,
            lat REAL NOT NULL, lon REAL NOT NULL,
            dist_m REAL NOT NULL, elev_m REAL NOT NULL,
            heading_deg REAL NOT NULL, curvature_deg_per_100m REAL NOT NULL,
            slope_deg REAL NOT NULL, aspect_deg REAL NOT NULL,
            tunnel INTEGER NOT NULL,
            horizon BLOB NOT NULL
        );
        CREATE TABLE landmarks (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
            sample_id INTEGER NOT NULL REFERENCES samples(id),
            lat REAL NOT NULL, lon REAL NOT NULL, elev_m REAL NOT NULL
        );
    """)
    hz = np.clip(np.round(horizon * HORIZON_SCALE), -900, 900).astype("<i2")
    con.executemany(
        "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (i, float(route.lat[i]), float(route.lon[i]), float(route.dist_m[i]), float(elev[i]),
             float(route.heading_deg[i]), float(route.curvature_deg_per_100m[i]),
             float(slope[i]), float(aspect[i]), int(route.tunnel[i]), hz[i].tobytes())
            for i in range(len(route.lat))
        ],
    )
    con.executemany(
        "INSERT INTO landmarks (name, kind, sample_id, lat, lon, elev_m) VALUES (?,?,?,?,?,?)",
        [(m["name"], m["kind"], m["sample"], m["lat"], m["lon"], float(elev[m["sample"]])) for m in marks],
    )
    con.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.commit()
    con.execute("VACUUM")
    con.close()


def write_gpx(path: Path, lat, lon, elev, start: datetime, speed_kmh: float, name: str) -> None:
    dist = np.concatenate([[0.0], np.cumsum(np.full(len(lat) - 1, 0.0))])
    from .route import haversine

    dist[1:] = np.cumsum(haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))
    secs = dist / (speed_kmh / 3.6)
    pts = "\n".join(
        f'      <trkpt lat="{a:.6f}" lon="{b:.6f}"><ele>{e:.1f}</ele>'
        f"<time>{(start + timedelta(seconds=float(s))).strftime('%Y-%m-%dT%H:%M:%SZ')}</time></trkpt>"
        for a, b, e, s in zip(lat, lon, elev, secs)
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="snowguard-terrain" xmlns="http://www.topografix.com/GPX/1/1">\n'
        f"  <trk><name>{name}</name><trkseg>\n{pts}\n  </trkseg></trk>\n</gpx>\n",
        encoding="utf-8",
    )


def shaded(horizon: np.ndarray, sun_el: np.ndarray, sun_az: np.ndarray) -> np.ndarray:
    """Linear interpolation of the horizon at the sun's azimuth; True = in shadow."""
    step = 360.0 / horizon.shape[1]
    f = sun_az / step
    i0 = np.floor(f).astype(int) % horizon.shape[1]
    i1 = (i0 + 1) % horizon.shape[1]
    w = f - np.floor(f)
    rows = np.arange(len(horizon))
    h = horizon[rows, i0] * (1 - w) + horizon[rows, i1] * w
    return sun_el <= h


def profile_plot(path: Path, route, elev, horizon, when_utc: list[datetime]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    km = route.dist_m / 1000
    fig, axes = plt.subplots(len(when_utc), 1, figsize=(12, 2.4 * len(when_utc)), sharex=True)
    stats = {}
    for ax, t in zip(np.atleast_1d(axes), when_utc):
        el, az = solar_position(route.lat, route.lon, np.full(len(km), t.timestamp()))
        sh = shaded(horizon, el, az) | route.tunnel
        ist = (t + timedelta(hours=5, minutes=30)).strftime("%d %b %H:%M IST")
        stats[ist] = {"shaded_fraction": round(float(sh.mean()), 3),
                      "shaded_and_above_3000m_km": round(float((sh & (elev >= 3000)).sum() * np.diff(route.dist_m).mean() / 1000), 1)}
        ax.fill_between(km, elev.min() - 100, elev, color="#cfd8dc")
        ax.scatter(km[sh], elev[sh], s=2, c="#1a237e", label="shaded")
        ax.scatter(km[~sh], elev[~sh], s=2, c="#ffb300", label="sunlit")
        ax.axhline(3000, color="#d50000", lw=0.8, ls="--")
        ax.set_ylabel("m")
        ax.set_title(f"{ist}: {sh.mean() * 100:.0f}% of route in shade", fontsize=9)
    np.atleast_1d(axes)[0].legend(loc="upper left", markerscale=4, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("km from Manali")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--spacing", type=float, default=75.0)
    ap.add_argument("--horizon-km", type=float, default=20.0)
    ap.add_argument("--android-assets", help="copy the database here")
    args = ap.parse_args(argv)
    # OSM place names include Devanagari/Tibetan; the Windows console defaults to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)

    print("[route] fetching OSM route")
    route = build_route(cache, args.spacing)
    print(f"[route] {route.dist_m[-1] / 1000:.1f} km, {len(route.lat)} samples, "
          f"{route.tunnel.sum() * args.spacing / 1000:.1f} km in tunnels")

    pad_deg = args.horizon_km / 111.0 + 0.02
    bounds = (route.lat.min() - pad_deg, route.lon.min() - pad_deg / np.cos(np.radians(route.lat.max())),
              route.lat.max() + pad_deg, route.lon.max() + pad_deg / np.cos(np.radians(route.lat.max())))
    dem = Dem(download_tiles(tile_names(*bounds), cache / "dem"), bounds)

    raw_elev = dem.sample(route.lat, route.lon)
    elev = fill_tunnels(raw_elev, route.tunnel)
    slope, aspect = dem.slope_aspect(route.lat, route.lon)
    slope[route.tunnel], aspect[route.tunnel] = 0.0, 0.0
    # Horizon tracing is the slow step (~10 min); cache it per route geometry.
    key = f"{len(route.lat)}_{args.spacing:g}_{args.horizon_km:g}_{route.lat.sum():.4f}_{route.lon.sum():.4f}"
    horizon_cache = cache / f"horizons_{key}.npy"
    if horizon_cache.exists():
        horizon = np.load(horizon_cache)
        print(f"[dem] horizons loaded from {horizon_cache}")
    else:
        horizon = dem.horizons(route.lat, route.lon, elev, N_AZIMUTH, args.horizon_km * 1000)
        np.save(horizon_cache, horizon)
    horizon[route.tunnel] = 90.0

    marks = landmarks(route, cache)
    print(f"[route] {len(marks)} landmarks: " + ", ".join(m["name"] for m in marks))

    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "route": "Manali-Leh (NH3 via Atal Tunnel)",
        "route_source": "OpenStreetMap via OSRM",
        "dem_source": "Copernicus GLO-30 (AWS Open Data)",
        "spacing_m": args.spacing,
        "n_samples": len(route.lat),
        "length_m": float(route.dist_m[-1]),
        "n_azimuth": N_AZIMUTH,
        "horizon_scale": HORIZON_SCALE,
        "horizon_radius_m": args.horizon_km * 1000,
        "freeze_thresholds": FREEZE_THRESHOLDS,
    }
    db = out / DB_NAME
    write_db(db, route, elev, slope, aspect, horizon, marks, meta)
    print(f"[db] {db} ({db.stat().st_size / 1024:.0f} KB)")

    start = datetime(2026, 1, 15, 2, 30, tzinfo=timezone.utc)  # 08:00 IST
    write_gpx(out / "route_manali_leh.gpx", route.lat, route.lon, elev, start, 40, "Manali-Leh")
    hi = int(np.argmax(elev >= 4300))
    a = max(0, hi - int(10_000 / args.spacing))
    b = min(len(elev), a + int(35_000 / args.spacing))
    write_gpx(out / "route_demo_high.gpx", route.lat[a:b], route.lon[a:b], elev[a:b], start, 40,
              f"SnowGuard demo km {route.dist_m[a] / 1000:.0f}-{route.dist_m[b - 1] / 1000:.0f}")

    shade_stats = profile_plot(out / "route_profile.png", route, elev, horizon, [
        datetime(2026, 1, 15, 3, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 15, 6, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
    ])
    open_sky = horizon[~route.tunnel]
    report = {
        "length_km": round(route.dist_m[-1] / 1000, 1),
        "samples": len(route.lat),
        "tunnel_km": round(route.tunnel.sum() * args.spacing / 1000, 2),
        "elev_min_m": round(float(elev.min())), "elev_max_m": round(float(elev.max())),
        "km_above_3000m": round(float((elev >= 3000).sum() * args.spacing / 1000), 1),
        "km_above_4000m": round(float((elev >= 4000).sum() * args.spacing / 1000), 1),
        "horizon_deg_median": round(float(np.median(open_sky)), 1),
        "horizon_deg_p95": round(float(np.percentile(open_sky, 95)), 1),
        "landmarks": [{k: m[k] for k in ("name", "kind", "offset_m")} | {"km": round(route.dist_m[m["sample"]] / 1000, 1)}
                      for m in marks],
        "demo_gpx_km": [round(route.dist_m[a] / 1000, 1), round(route.dist_m[b - 1] / 1000, 1)],
        "winter_shade": shade_stats,
        "db_kb": round(db.stat().st_size / 1024),
    }
    (out / "route_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "landmarks"}, indent=2))

    if args.android_assets:
        assets = Path(args.android_assets)
        assets.mkdir(parents=True, exist_ok=True)
        shutil.copy(db, assets / DB_NAME)
        print(f"[db] copied to {assets / DB_NAME}")


if __name__ == "__main__":
    main()
