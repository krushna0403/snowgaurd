"""Copernicus GLO-30 DEM: tile download, mosaic, sampling, slope/aspect and horizon angles.

Tiles come from the public AWS Open Data bucket (no account or key needed).
A point is in terrain shadow when the sun's elevation is below the horizon
angle in the sun's azimuth direction.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import map_coordinates

TILE_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
            "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
EARTH_R = 6_371_008.8
REFRACTION_K = 0.13  # standard terrestrial refraction coefficient


def tile_names(south: float, west: float, north: float, east: float) -> list[tuple[int, int]]:
    return [(la, lo) for la in range(math.floor(south), math.floor(north) + 1)
            for lo in range(math.floor(west), math.floor(east) + 1)]


def download_tiles(tiles: list[tuple[int, int]], dem_dir: Path) -> list[Path]:
    dem_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for la, lo in tiles:
        url = TILE_URL.format(ns="N" if la >= 0 else "S", lat=abs(la), ew="E" if lo >= 0 else "W", lon=abs(lo))
        path = dem_dir / url.rsplit("/", 1)[1]
        if not path.exists():
            print(f"[dem] downloading {path.name}")
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = path.with_suffix(".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(path)
        paths.append(path)
    return paths


class Dem:
    """In-memory mosaic with bilinear sampling in (lat, lon)."""

    def __init__(self, paths: list[Path], bounds: tuple[float, float, float, float]):
        import rasterio
        from rasterio.merge import merge

        south, west, north, east = bounds
        srcs = [rasterio.open(p) for p in paths]
        try:
            arr, transform = merge(srcs, bounds=(west, south, east, north), nodata=np.nan)
        finally:
            for s in srcs:
                s.close()
        self.z = arr[0].astype(np.float32)
        self.west, self.north = transform.c, transform.f
        self.dlon, self.dlat = transform.a, -transform.e
        print(f"[dem] mosaic {self.z.shape[1]}x{self.z.shape[0]} px, "
              f"{np.nanmin(self.z):.0f}-{np.nanmax(self.z):.0f} m")

    def _rc(self, lat, lon):
        # Pixel-centre convention: value at (row + 0.5, col + 0.5) in transform space.
        return (self.north - np.asarray(lat)) / self.dlat - 0.5, (np.asarray(lon) - self.west) / self.dlon - 0.5

    def sample(self, lat, lon) -> np.ndarray:
        r, c = self._rc(lat, lon)
        return map_coordinates(self.z, [np.ravel(r), np.ravel(c)], order=1, mode="nearest").reshape(np.shape(r))

    def slope_aspect(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Terrain slope (deg) and downslope aspect (deg clockwise from north) from a 3-pixel stencil."""
        lat = np.asarray(lat)
        dy = self.dlat * 111_320.0
        dx = self.dlon * 111_320.0 * np.cos(np.radians(lat))
        step = 1.5
        zn = self.sample(lat + step * self.dlat, lon)
        zs = self.sample(lat - step * self.dlat, lon)
        ze = self.sample(lat, lon + step * self.dlon)
        zw = self.sample(lat, lon - step * self.dlon)
        dzdx = (ze - zw) / (2 * step * dx)
        dzdy = (zn - zs) / (2 * step * dy)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        aspect = np.mod(np.degrees(np.arctan2(-dzdx, -dzdy)), 360.0)
        return slope, aspect

    def horizons(self, lat, lon, z0, n_azimuth: int = 72, max_dist_m: float = 20_000.0,
                 min_dist_m: float = 60.0, n_steps: int = 160, observer_h_m: float = 2.0,
                 chunk: int = 150) -> np.ndarray:
        """Horizon elevation angle (deg) per point and azimuth, shape (n_points, n_azimuth)."""
        lat, lon, z0 = map(np.asarray, (lat, lon, z0))
        az = np.radians(np.arange(n_azimuth) * 360.0 / n_azimuth)
        d = np.geomspace(min_dist_m, max_dist_m, n_steps)
        drop = d**2 / (2 * EARTH_R) * (1 - REFRACTION_K)
        out = np.empty((len(lat), n_azimuth), dtype=np.float32)
        for s in range(0, len(lat), chunk):
            la, lo, zz = lat[s:s + chunk, None, None], lon[s:s + chunk, None, None], z0[s:s + chunk, None, None]
            dn = d[None, None, :] * np.cos(az)[None, :, None]
            de = d[None, None, :] * np.sin(az)[None, :, None]
            plat = la + dn / 111_320.0
            plon = lo + de / (111_320.0 * np.cos(np.radians(la)))
            z = self.sample(plat, plon)
            ang = np.degrees(np.arctan2(z - drop[None, None, :] - (zz + observer_h_m), d[None, None, :]))
            out[s:s + chunk] = np.nanmax(ang, axis=2)
            print(f"\r[dem] horizons {min(s + chunk, len(lat))}/{len(lat)}", end="", flush=True)
        print()
        return out
