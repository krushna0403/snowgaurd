package com.snowguard.app.terrain

import java.util.Calendar
import java.util.TimeZone

enum class FreezeRisk { NONE, ELEVATED, HIGH }

/** FR-L1/L2: the first shaded, freeze-prone sample within the look-ahead window. */
data class LookAhead(
    val distanceM: Double,
    val risk: FreezeRisk,
    /** The hazard starts on (or just before) a sharp bend. */
    val blindCorner: Boolean,
    val windowM: Double,
)

data class TerrainState(
    val onRoute: Boolean,
    val snap: Snap?,
    val elevationM: Double?,
    val sun: SunPosition,
    val horizonDeg: Double?,
    val sunlit: Boolean?,
    val inTunnel: Boolean,
    val minutesInShade: Int?,
    val northFacing: Boolean,
    val threshold: FreezeThreshold?,
    val freezeRisk: FreezeRisk,
    val nearestLandmark: Landmark?,
    val landmarkDistanceM: Double?,
    val computeMicros: Long,
    val lookAhead: LookAhead? = null,
    val speedMps: Double? = null,
)

/**
 * Terrain layer (FR-T1..T8): sun position from GPS + time, shade from the
 * precomputed horizon at the snapped route sample, and freeze risk from the
 * altitude-month threshold, slope aspect and how long the road has been
 * out of the sun.
 */
class TerrainEngine(private val route: RouteData) {

    private val snapper = RouteSnapper(route)
    private val calendar = Calendar.getInstance(TimeZone.getTimeZone("Asia/Kolkata"))

    fun evaluate(fix: RouteFix): TerrainState {
        val t0 = System.nanoTime()
        val sun = SolarPosition.compute(fix.lat, fix.lon, fix.timeMillis)
        val snap = snapper.snap(fix)
        if (snap == null) {
            return TerrainState(
                onRoute = false, snap = null, elevationM = null, sun = sun, horizonDeg = null, sunlit = null,
                inTunnel = false, minutesInShade = null, northFacing = false, threshold = null,
                freezeRisk = FreezeRisk.NONE, nearestLandmark = null, landmarkDistanceM = null,
                computeMicros = (System.nanoTime() - t0) / 1000, speedMps = fix.speedMps,
            )
        }

        val i = snap.sample
        val tunnel = route.tunnel[i]
        val horizon = route.horizonAt(i, sun.azimuthDeg)
        val sunlit = !tunnel && sun.elevationDeg > horizon
        val minutesInShade = if (sunlit) 0 else minutesSinceSunlit(i, fix)
        val northFacing = route.slopeDeg[i] >= NORTH_FACING_MIN_SLOPE &&
            (route.aspectDeg[i] >= 315.0 || route.aspectDeg[i] <= 45.0)

        calendar.timeInMillis = fix.timeMillis
        val threshold = route.thresholdFor(calendar.get(Calendar.MONTH) + 1)
        val elevation = route.elevM[i]
        val freezeProne = threshold != null && elevation >= threshold.minElevationM
        val risk = when {
            !freezeProne || sunlit -> FreezeRisk.NONE
            sun.elevationDeg <= 0 || northFacing || minutesInShade >= LONG_SHADE_MIN -> FreezeRisk.HIGH
            else -> FreezeRisk.ELEVATED
        }

        val (landmark, landmarkDist) = nearestLandmark(snap.distAlongM)
        val ahead = if (snap.direction != 0) lookAhead(snap, sun, threshold, fix.speedMps) else null
        return TerrainState(
            onRoute = true, snap = snap, elevationM = elevation, sun = sun, horizonDeg = horizon,
            sunlit = sunlit, inTunnel = tunnel, minutesInShade = minutesInShade, northFacing = northFacing,
            threshold = threshold, freezeRisk = risk, nearestLandmark = landmark, landmarkDistanceM = landmarkDist,
            computeMicros = (System.nanoTime() - t0) / 1000, lookAhead = ahead, speedMps = fix.speedMps,
        )
    }

    /**
     * FR-L1/L2: walk the route in the travel direction for ~30 s of driving
     * (clamped to 200-500 m) and return the first sample that is shaded and
     * above the season's freeze threshold, flagging it if it sits on a sharp bend.
     */
    fun lookAhead(snap: Snap, sun: SunPosition, threshold: FreezeThreshold?, speedMps: Double?): LookAhead? {
        if (threshold == null || snap.direction == 0) return null
        val window = ((speedMps ?: DEFAULT_SPEED_MPS) * LOOKAHEAD_S).coerceIn(MIN_LOOKAHEAD_M, MAX_LOOKAHEAD_M)
        val dir = snap.direction
        var j = snap.sample + dir
        while (j in 0 until route.size) {
            val d = kotlin.math.abs(route.distM[j] - snap.distAlongM)
            if (d > window) break
            if (!route.tunnel[j] && route.elevM[j] >= threshold.minElevationM &&
                sun.elevationDeg <= route.horizonAt(j, sun.azimuthDeg)
            ) {
                val north = route.slopeDeg[j] >= NORTH_FACING_MIN_SLOPE &&
                    (route.aspectDeg[j] >= 315.0 || route.aspectDeg[j] <= 45.0)
                val corner = (-1..CORNER_SPAN_SAMPLES).any { k ->
                    val m = j + dir * k
                    m in 0 until route.size && kotlin.math.abs(route.curvatureDegPer100m[m]) >= BLIND_CORNER_DEG_PER_100M
                }
                val risk = if (sun.elevationDeg <= 0 || north) FreezeRisk.HIGH else FreezeRisk.ELEVATED
                return LookAhead(d, risk, corner, window)
            }
            j += dir
        }
        return null
    }

    /** FR-T6: minutes since the sun last reached this sample, looking back up to 12 h in 10-min steps. */
    private fun minutesSinceSunlit(i: Int, fix: RouteFix): Int {
        if (route.tunnel[i]) return MAX_LOOKBACK_MIN
        var minutes = SHADE_STEP_MIN
        while (minutes <= MAX_LOOKBACK_MIN) {
            val sun = SolarPosition.compute(fix.lat, fix.lon, fix.timeMillis - minutes * 60_000L)
            if (sun.elevationDeg > route.horizonAt(i, sun.azimuthDeg)) return minutes
            minutes += SHADE_STEP_MIN
        }
        return MAX_LOOKBACK_MIN
    }

    private fun nearestLandmark(along: Double): Pair<Landmark?, Double?> {
        var best: Landmark? = null
        var bestD = Double.MAX_VALUE
        for (m in route.landmarks) {
            val d = kotlin.math.abs(route.distM[m.sampleIndex] - along)
            if (d < bestD) {
                bestD = d
                best = m
            }
        }
        return best to best?.let { route.distM[it.sampleIndex] - along }
    }

    private companion object {
        const val NORTH_FACING_MIN_SLOPE = 10.0
        const val LONG_SHADE_MIN = 60
        const val SHADE_STEP_MIN = 10
        const val MAX_LOOKBACK_MIN = 720
        const val LOOKAHEAD_S = 30.0
        const val DEFAULT_SPEED_MPS = 10.0
        const val MIN_LOOKAHEAD_M = 200.0
        const val MAX_LOOKAHEAD_M = 500.0
        /** ~top 5 % of route samples by curvature: hairpins and tight bends. */
        const val BLIND_CORNER_DEG_PER_100M = 40.0
        const val CORNER_SPAN_SAMPLES = 2
    }
}
