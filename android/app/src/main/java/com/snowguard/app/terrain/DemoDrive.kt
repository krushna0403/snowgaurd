package com.snowguard.app.terrain

import java.util.Calendar
import java.util.TimeZone

/**
 * Synthetic drive along the route for demos without being on the Manali-Leh
 * road. Fixes carry a simulated clock (default 15 Jan 08:30 IST) so the sun,
 * shade history and freeze thresholds behave as on a winter morning, and the
 * clock runs [timeScale] times faster than real time to keep the demo short.
 */
class DemoDrive(
    private val route: RouteData,
    val startM: Double = DEFAULT_START_KM * 1000,
    val endM: Double = DEFAULT_END_KM * 1000,
    val speedMps: Double = DEFAULT_SPEED_MPS,
    val timeScale: Double = DEFAULT_TIME_SCALE,
    val simStartMillis: Long = defaultStartMillis(),
) {
    /** Simulated seconds since the start; advanced by the caller. */
    var elapsedS = 0.0
        private set

    val distanceM get() = startM + speedMps * elapsedS
    val simTimeMillis get() = simStartMillis + (elapsedS * 1000).toLong()
    val finished get() = distanceM > endM

    /** Advances by one real-time tick and returns the fix there, or null once past [endM]. */
    fun next(realTickS: Double = 1.0): RouteFix? {
        val fix = fixAt(elapsedS)
        elapsedS += realTickS * timeScale
        return fix
    }

    fun fixAt(simElapsedS: Double): RouteFix? {
        val d = startM + speedMps * simElapsedS
        if (d > endM || d > route.distM.last()) return null
        val i = indexBefore(d)
        val j = minOf(i + 1, route.size - 1)
        val span = route.distM[j] - route.distM[i]
        val w = if (span > 0) ((d - route.distM[i]) / span).coerceIn(0.0, 1.0) else 0.0
        return RouteFix(
            lat = route.lat[i] + (route.lat[j] - route.lat[i]) * w,
            lon = route.lon[i] + (route.lon[j] - route.lon[i]) * w,
            timeMillis = simStartMillis + (simElapsedS * 1000).toLong(),
            bearingDeg = route.headingDeg[i],
            speedMps = speedMps,
        )
    }

    private fun indexBefore(d: Double): Int {
        var lo = 0
        var hi = route.size - 1
        while (lo < hi) {
            val mid = (lo + hi + 1) ushr 1
            if (route.distM[mid] <= d) lo = mid else hi = mid - 1
        }
        return lo
    }

    companion object {
        /** The high, deeply shaded stretch also exported as route_demo_high.gpx. */
        const val DEFAULT_START_KM = 119.6
        const val DEFAULT_END_KM = 154.4
        const val DEFAULT_SPEED_MPS = 40 / 3.6
        const val DEFAULT_TIME_SCALE = 4.0

        fun defaultStartMillis(): Long = Calendar.getInstance(TimeZone.getTimeZone("Asia/Kolkata")).run {
            clear()
            set(Calendar.getInstance().get(Calendar.YEAR), Calendar.JANUARY, 15, 8, 30, 0)
            timeInMillis
        }
    }
}
