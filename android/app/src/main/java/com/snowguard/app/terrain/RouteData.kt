package com.snowguard.app.terrain

/** FR-T3: a road segment is freeze-prone in these months at or above [minElevationM]. */
data class FreezeThreshold(val season: String, val months: Set<Int>, val minElevationM: Double)

data class Landmark(val name: String, val kind: String, val sampleIndex: Int, val elevationM: Double)

/**
 * The precomputed corridor, held entirely in memory (~6k samples). Arrays are
 * indexed by sample; [horizonDeg] is row-major `[sample * nAzimuth + k]` with
 * azimuth `k * 360 / nAzimuth` clockwise from north.
 */
class RouteData(
    val lat: DoubleArray,
    val lon: DoubleArray,
    val distM: DoubleArray,
    val elevM: DoubleArray,
    val headingDeg: DoubleArray,
    val curvatureDegPer100m: DoubleArray,
    val slopeDeg: DoubleArray,
    val aspectDeg: DoubleArray,
    val tunnel: BooleanArray,
    val horizonDeg: FloatArray,
    val nAzimuth: Int,
    val thresholds: List<FreezeThreshold>,
    val landmarks: List<Landmark>,
    val name: String,
) {
    val size get() = lat.size

    init {
        require(horizonDeg.size == size * nAzimuth) { "horizon array size mismatch" }
    }

    /** Horizon angle at [sample] towards [azimuthDeg], linearly interpolated between directions. */
    fun horizonAt(sample: Int, azimuthDeg: Double): Double {
        val step = 360.0 / nAzimuth
        val f = ((azimuthDeg % 360.0) + 360.0) % 360.0 / step
        val k0 = f.toInt() % nAzimuth
        val k1 = (k0 + 1) % nAzimuth
        val w = f - f.toInt()
        val base = sample * nAzimuth
        return horizonDeg[base + k0] * (1 - w) + horizonDeg[base + k1] * w
    }

    fun thresholdFor(month: Int): FreezeThreshold? = thresholds.firstOrNull { month in it.months }

    companion object {
        val DEFAULT_THRESHOLDS = listOf(
            FreezeThreshold("winter", setOf(11, 12, 1, 2, 3), 3000.0),
            FreezeThreshold("shoulder", setOf(4, 5, 10), 4000.0),
            FreezeThreshold("summer", setOf(6, 7, 8, 9), 4800.0),
        )
    }
}
