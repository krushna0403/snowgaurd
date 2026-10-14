"""Route geometry from OpenStreetMap: road polyline, resampling, heading, curvature,
tunnels and named landmarks.

The polyline comes from the public OSRM router (OSM road graph), which follows
the current alignment through the Atal Tunnel rather than over Rohtang Pass.
Downloads are cached under ``data/`` so rebuilds are offline and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

EARTH_R = 6_371_008.8
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Start, via points that pin the route to the Manali-Leh highway, end. (lat, lon)
MANALI_LEH_VIA = [
    (32.2396, 77.1887),  # Manali
    (32.5711, 77.0322),  # Keylong
    (33.1400, 77.7880),  # Pang
    (34.1526, 77.5771),  # Leh
]


@dataclass
class Route:
    lat: np.ndarray
    lon: np.ndarray
    dist_m: np.ndarray
    heading_deg: np.ndarray
    curvature_deg_per_100m: np.ndarray
    tunnel: np.ndarray


HEADERS = {"User-Agent": "SnowGuard-terrain/0.1 (academic research prototype; offline route builder)"}


def _get_json(url: str, cache: Path, **kwargs) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if "data" in kwargs:
        resp = requests.post(url, timeout=180, headers=HEADERS, **kwargs)
    else:
        resp = requests.get(url, timeout=180, headers=HEADERS)
    resp.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text(resp.text, encoding="utf-8")
    tmp.replace(cache)
    return resp.json()


def fetch_polyline(via: list[tuple[float, float]], cache_dir: Path) -> np.ndarray:
    coords = ";".join(f"{lon},{lat}" for lat, lon in via)
    data = _get_json(OSRM_URL.format(coords=coords), cache_dir / "osrm_route.json")
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data.get('code')}")
    lonlat = np.array(data["routes"][0]["geometry"]["coordinates"], dtype=np.float64)
    return lonlat[:, ::-1]  # -> (lat, lon)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.mod(np.degrees(np.arctan2(y, x)), 360.0)


def resample(latlon: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg = haversine(latlon[:-1, 0], latlon[:-1, 1], latlon[1:, 0], latlon[1:, 1])
    keep = np.concatenate([[True], seg > 0.01])
    latlon = latlon[keep]
    cum = np.concatenate([[0.0], np.cumsum(seg[seg > 0.01])])
    targets = np.arange(0.0, cum[-1], spacing_m)
    targets = np.append(targets, cum[-1]) if cum[-1] - targets[-1] > spacing_m / 2 else targets
    return np.interp(targets, cum, latlon[:, 0]), np.interp(targets, cum, latlon[:, 1]), targets


def heading_and_curvature(lat, lon, spacing_m: float, window_m: float = 50.0):
    n = len(lat)
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    prv = np.maximum(np.arange(n) - 1, 0)
    heading = bearing(lat[prv], lon[prv], lat[nxt], lon[nxt])
    k = max(1, int(round(window_m / spacing_m)))
    ahead = bearing(lat, lon, lat[np.minimum(np.arange(n) + k, n - 1)], lon[np.minimum(np.arange(n) + k, n - 1)])
    behind = bearing(lat[np.maximum(np.arange(n) - k, 0)], lon[np.maximum(np.arange(n) - k, 0)], lat, lon)
    turn = (ahead - behind + 180.0) % 360.0 - 180.0  # signed, + = right
    span = np.maximum(spacing_m * (np.minimum(np.arange(n) + k, n - 1) - np.maximum(np.arange(n) - k, 0)), 1.0)
    return heading, turn / span * 100.0


def _overpass(query: str, cache: Path) -> dict:
    return _get_json(OVERPASS_URL, cache, data={"data": query})


def _bbox(lat, lon, pad: float = 0.03) -> str:
    return f"{lat.min() - pad},{lon.min() - pad},{lat.max() + pad},{lon.max() + pad}"


def _nearest(lat, lon, plat, plon):
    """Index of nearest route sample and distance in metres, for each query point."""
    idx, dist = [], []
    for a, b in zip(np.atleast_1d(plat), np.atleast_1d(plon)):
        d = haversine(lat, lon, a, b)
        i = int(np.argmin(d))
        idx.append(i)
        dist.append(float(d[i]))
    return np.array(idx), np.array(dist)


def mark_tunnels(lat, lon, cache_dir: Path, max_offset_m: float = 40.0) -> np.ndarray:
    q = f"[out:json][timeout:120];way[highway][tunnel=yes]({_bbox(lat, lon)});out geom;"
    data = _overpass(q, cache_dir / "overpass_tunnels.json")
    tunnel = np.zeros(len(lat), dtype=bool)
    for way in data.get("elements", []):
        pts = np.array([(g["lat"], g["lon"]) for g in way.get("geometry", [])])
        if len(pts) < 2:
            continue
        idx, dist = _nearest(lat, lon, pts[:, 0], pts[:, 1])
        on = idx[dist < max_offset_m]
        if on.size < 2:
            continue
        way_len = float(haversine(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1]).sum())
        spacing = float(haversine(lat[0], lon[0], lat[1], lon[1]))
        # Reject ways that merely touch the route at two distant places.
        if (on.max() - on.min()) * spacing <= way_len * 1.3 + 2 * spacing:
            tunnel[on.min(): on.max() + 1] = True
    return tunnel


def build_route(cache_dir: Path, spacing_m: float = 75.0) -> Route:
    latlon = fetch_polyline(MANALI_LEH_VIA, cache_dir)
    lat, lon, dist = resample(latlon, spacing_m)
    heading, curvature = heading_and_curvature(lat, lon, spacing_m)
    tunnel = mark_tunnels(lat, lon, cache_dir)
    return Route(lat, lon, dist, heading, curvature, tunnel)


def landmarks(route: Route, cache_dir: Path) -> list[dict]:
    """Passes and settlements close to the road, ordered along the route."""
    q = (f"[out:json][timeout:120];("
         f"node[mountain_pass=yes]({_bbox(route.lat, route.lon)});"
         f"node[place~\"^(town|village)$\"]({_bbox(route.lat, route.lon)});"
         f");out body;")
    data = _overpass(q, cache_dir / "overpass_landmarks.json")
    found: dict[str, dict] = {}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name:en") or tags.get("name")
        if not name:
            continue
        kind = "pass" if tags.get("mountain_pass") == "yes" else tags.get("place", "place")
        limit = 500.0 if kind == "pass" else 1500.0
        (i,), (d,) = _nearest(route.lat, route.lon, el["lat"], el["lon"])
        if d > limit:
            continue
        if name not in found or d < found[name]["offset_m"]:
            found[name] = {"name": name, "kind": kind, "sample": int(i), "lat": el["lat"], "lon": el["lon"],
                           "offset_m": round(d, 1)}
    return sorted(found.values(), key=lambda x: x["sample"])
