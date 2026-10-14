"""NOAA solar-position equations (the NOAA Solar Calculator spreadsheet algorithm).

Accurate to roughly 0.01 deg for dates within a few centuries of 2000, far
inside the BRD's 1 deg budget. ``android/.../terrain/SolarPosition.kt`` is a
line-for-line port; keep the two in sync.

All angles in degrees. Azimuth is clockwise from true north. Elevation
includes atmospheric refraction (apparent elevation).
"""

from __future__ import annotations

import numpy as np


def solar_position(lat, lon, unix_seconds):
    """Return (apparent_elevation, azimuth) for arrays/scalars of lat, lon and UTC epoch seconds."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    t = np.asarray(unix_seconds, dtype=np.float64)

    jd = t / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0

    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = np.radians(m)
    center = (np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + np.sin(2 * mr) * (0.019993 - 0.000101 * jc)
              + np.sin(3 * mr) * 0.000289)
    true_long = l0 + center
    omega = np.radians(125.04 - 1934.136 * jc)
    app_long = true_long - 0.00569 - 0.00478 * np.sin(omega)
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * np.cos(omega)
    decl = np.degrees(np.arcsin(np.sin(np.radians(obliq)) * np.sin(np.radians(app_long))))

    y = np.tan(np.radians(obliq / 2.0)) ** 2
    l0r = np.radians(l0)
    eq_time = 4.0 * np.degrees(
        y * np.sin(2 * l0r)
        - 2 * ecc * np.sin(mr)
        + 4 * ecc * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r)
        - 1.25 * ecc * ecc * np.sin(2 * mr)
    )

    minutes_utc = np.mod(t, 86400.0) / 60.0
    true_solar = np.mod(minutes_utc + eq_time + 4.0 * lon, 1440.0)
    hour_angle = np.where(true_solar / 4.0 < 0, true_solar / 4.0 + 180.0, true_solar / 4.0 - 180.0)

    latr, declr, har = np.radians(lat), np.radians(decl), np.radians(hour_angle)
    cos_zen = np.clip(np.sin(latr) * np.sin(declr) + np.cos(latr) * np.cos(declr) * np.cos(har), -1.0, 1.0)
    zen = np.degrees(np.arccos(cos_zen))
    zenr = np.radians(zen)

    denom = np.cos(latr) * np.sin(zenr)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = np.clip((np.sin(latr) * np.cos(zenr) - np.sin(declr)) / denom, -1.0, 1.0)
    acos_az = np.degrees(np.arccos(np.nan_to_num(cos_az)))
    azimuth = np.where(hour_angle > 0, np.mod(acos_az + 180.0, 360.0), np.mod(540.0 - acos_az, 360.0))

    elevation = 90.0 - zen
    return elevation + refraction(elevation), azimuth


def refraction(elevation):
    """NOAA atmospheric refraction correction in degrees for a geometric elevation."""
    e = np.asarray(elevation, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        te = np.tan(np.radians(e))
        arcsec = np.select(
            [e > 85.0, e > 5.0, e > -0.575],
            [
                0.0,
                58.1 / te - 0.07 / te**3 + 0.000086 / te**5,
                1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711))),
            ],
            default=-20.772 / te,
        )
    return arcsec / 3600.0
