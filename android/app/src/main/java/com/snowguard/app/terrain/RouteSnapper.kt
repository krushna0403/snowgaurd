package com.snowguard.app.terrain

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sqrt

data class RouteFix(
    val lat: Double,
    val lon: Double,
    val timeMillis: Long,
    val bearingDeg: Double? = null,
    val speedMps: Double? = null,
)

data class Snap(
    val sample: Int,
    val offsetM: Double,
    val distAlongM: Double,
    /** +1 towards the route end (Leh), -1 towards the start (Manali), 0 unknown. */
    val direction: Int,
)

/**
 * Snaps GPS fixes onto the corridor polyline (FR-T8). Besides raw distance it
 * scores candidates by heading agreement and continuity with the previous
 * snap, so a fix between two legs of a hairpin lands on the leg actually
 * being driven. Returns null when the fix is off the corridor.
 */
class RouteSnapper(
    private val route: RouteData,
    private val maxOffsetM: Double = 150.0,
) {
    private var last: Snap? = null
    private var lastTime = 0L
    private var progressAccum = 0.0

    fun reset() {
        last = null
        progressAccum = 0.0
    }

    fun snap(fix: RouteFix): Snap? {
        val prev = last?.takeIf { fix.timeMillis - lastTime in 0..CONTINUITY_WINDOW_MS }
        val cosLat = cos(Math.toRadians(fix.lat))
        val moving = (fix.speedMps ?: 0.0) >= MIN_SPEED_FOR_HEADING && fix.bearingDeg != null
        val direction = prev?.direction ?: 0
        val allowance = prev?.let {
            (fix.speedMps ?: MAX_SPEED_MPS) * (fix.timeMillis - lastTime) / 1000.0 + CONTINUITY_SLACK_M
        } ?: 0.0

        var best = -1
        var bestScore = Double.MAX_VALUE
        var bestOffset = 0.0
        for (i in 0 until route.size) {
            val dy = (route.lat[i] - fix.lat) * M_PER_DEG
            val dx = (route.lon[i] - fix.lon) * M_PER_DEG * cosLat
            val d2 = dx * dx + dy * dy
            if (d2 > SEARCH_RADIUS_M * SEARCH_RADIUS_M) continue
            val d = sqrt(d2)
            var score = d
            if (moving && direction != 0) {
                val roadHeading = if (direction > 0) route.headingDeg[i] else route.headingDeg[i] + 180.0
                score += HEADING_WEIGHT_M * (1 - cos(Math.toRadians(fix.bearingDeg!! - roadHeading))) / 2
            }
            if (prev != null) {
                val jump = abs(route.distM[i] - prev.distAlongM) - allowance
                if (jump > 0) score += CONTINUITY_WEIGHT * jump
            }
            if (score < bestScore) {
                bestScore = score
                best = i
                bestOffset = d
            }
        }
        if (best < 0 || bestOffset > maxOffsetM) {
            if (best < 0) reset()
            return null
        }

        val along = projectAlong(best, fix.lat, fix.lon, cosLat)
        val newDirection = updateDirection(prev, along, direction)
        val result = Snap(best, bestOffset, along, newDirection)
        last = result
        lastTime = fix.timeMillis
        return result
    }

    /** Distance along the route of the fix projected onto the segment adjacent to [i]. */
    private fun projectAlong(i: Int, lat: Double, lon: Double, cosLat: Double): Double {
        var bestAlong = route.distM[i]
        var bestD = Double.MAX_VALUE
        for (j in intArrayOf(i - 1, i)) {
            if (j < 0 || j + 1 >= route.size) continue
            val ax = 0.0; val ay = 0.0
            val bx = (route.lon[j + 1] - route.lon[j]) * M_PER_DEG * cosLat
            val by = (route.lat[j + 1] - route.lat[j]) * M_PER_DEG
            val px = (lon - route.lon[j]) * M_PER_DEG * cosLat
            val py = (lat - route.lat[j]) * M_PER_DEG
            val len2 = bx * bx + by * by
            val t = if (len2 == 0.0) 0.0 else ((px - ax) * bx + (py - ay) * by) / len2
            val tc = t.coerceIn(0.0, 1.0)
            val ex = px - tc * bx; val ey = py - tc * by
            val d = ex * ex + ey * ey
            if (d < bestD) {
                bestD = d
                bestAlong = route.distM[j] + tc * (route.distM[j + 1] - route.distM[j])
            }
        }
        return bestAlong
    }

    private fun updateDirection(prev: Snap?, along: Double, current: Int): Int {
        if (prev == null) return current
        progressAccum = (progressAccum + along - prev.distAlongM).coerceIn(-DIRECTION_LOCK_M, DIRECTION_LOCK_M)
        return when {
            progressAccum >= DIRECTION_LOCK_M / 2 -> 1
            progressAccum <= -DIRECTION_LOCK_M / 2 -> -1
            else -> current
        }
    }

    private companion object {
        const val M_PER_DEG = 111_320.0
        const val SEARCH_RADIUS_M = 1_000.0
        const val MIN_SPEED_FOR_HEADING = 3.0
        const val MAX_SPEED_MPS = 30.0
        const val HEADING_WEIGHT_M = 80.0
        const val CONTINUITY_WEIGHT = 0.5
        const val CONTINUITY_SLACK_M = 150.0
        const val CONTINUITY_WINDOW_MS = 30_000L
        const val DIRECTION_LOCK_M = 200.0
    }
}
