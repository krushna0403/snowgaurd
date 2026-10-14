package com.snowguard.app.terrain

import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.tan

data class SunPosition(val elevationDeg: Double, val azimuthDeg: Double)

/**
 * NOAA solar-position equations; a port of `terrain/snowguard_terrain/solar.py`
 * (validated against pvlib's NREL SPA to within 0.3 deg). Elevation includes
 * refraction; azimuth is clockwise from true north. Works fully offline.
 */
object SolarPosition {

    fun compute(latDeg: Double, lonDeg: Double, epochMillis: Long): SunPosition {
        val t = epochMillis / 1000.0
        val jd = t / 86400.0 + 2440587.5
        val jc = (jd - 2451545.0) / 36525.0

        val l0 = mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
        val m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
        val ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
        val mr = rad(m)
        val center = sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) +
            sin(2 * mr) * (0.019993 - 0.000101 * jc) +
            sin(3 * mr) * 0.000289
        val trueLong = l0 + center
        val omega = rad(125.04 - 1934.136 * jc)
        val appLong = trueLong - 0.00569 - 0.00478 * sin(omega)
        val meanObliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
        val obliq = meanObliq + 0.00256 * cos(omega)
        val decl = deg(asin(sin(rad(obliq)) * sin(rad(appLong))))

        val y = tan(rad(obliq / 2.0)).let { it * it }
        val l0r = rad(l0)
        val eqTime = 4.0 * deg(
            y * sin(2 * l0r) - 2 * ecc * sin(mr) + 4 * ecc * y * sin(mr) * cos(2 * l0r) -
                0.5 * y * y * sin(4 * l0r) - 1.25 * ecc * ecc * sin(2 * mr)
        )

        val minutesUtc = mod(t, 86400.0) / 60.0
        val trueSolar = mod(minutesUtc + eqTime + 4.0 * lonDeg, 1440.0)
        val hourAngle = if (trueSolar / 4.0 < 0) trueSolar / 4.0 + 180.0 else trueSolar / 4.0 - 180.0

        val latr = rad(latDeg)
        val declr = rad(decl)
        val cosZen = (sin(latr) * sin(declr) + cos(latr) * cos(declr) * cos(rad(hourAngle))).coerceIn(-1.0, 1.0)
        val zen = deg(acos(cosZen))
        val zenr = rad(zen)
        val denom = cos(latr) * sin(zenr)
        val cosAz = if (denom == 0.0) 1.0 else ((sin(latr) * cos(zenr) - sin(declr)) / denom).coerceIn(-1.0, 1.0)
        val acosAz = deg(acos(cosAz))
        val azimuth = if (hourAngle > 0) mod(acosAz + 180.0, 360.0) else mod(540.0 - acosAz, 360.0)

        val elevation = 90.0 - zen
        return SunPosition(elevation + refraction(elevation), azimuth)
    }

    /** NOAA atmospheric refraction correction (degrees). */
    fun refraction(e: Double): Double {
        val te = tan(rad(e))
        val arcsec = when {
            e > 85.0 -> 0.0
            e > 5.0 -> 58.1 / te - 0.07 / (te * te * te) + 0.000086 / (te * te * te * te * te)
            e > -0.575 -> 1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
            else -> -20.772 / te
        }
        return arcsec / 3600.0
    }

    private fun rad(d: Double) = Math.toRadians(d)
    private fun deg(r: Double) = Math.toDegrees(r)
    private fun mod(a: Double, b: Double) = a - b * floor(a / b)
}
