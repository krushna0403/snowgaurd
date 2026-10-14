package com.snowguard.app.vision

import kotlin.math.abs
import kotlin.math.sqrt

enum class CameraHealth { OK, DARK, GLARE, BLURRED }

data class FrameQuality(
    val meanLuma: Float,
    /** Luma standard deviation: low for fog, a frosted lens or a covered camera. */
    val contrast: Float,
    /** Fraction of near-white pixels: high for sun glare straight into the lens. */
    val saturated: Float,
    /** Mean absolute horizontal luma gradient: low when the image is blurred. */
    val sharpness: Float,
) {
    /**
     * Thresholds are deliberately conservative: a snow-covered road is bright
     * and smooth, so only clearly unusable frames are rejected.
     */
    val verdict: CameraHealth
        get() = when {
            meanLuma < DARK_MEAN -> CameraHealth.DARK
            saturated > GLARE_FRACTION -> CameraHealth.GLARE
            contrast < FLAT_STD && sharpness < FLAT_GRADIENT -> CameraHealth.BLURRED
            else -> CameraHealth.OK
        }

    companion object {
        const val DARK_MEAN = 25f
        const val GLARE_FRACTION = 0.45f
        const val FLAT_STD = 8f
        const val FLAT_GRADIENT = 1.5f
        const val SATURATED_LUMA = 250
    }
}

/** FR-V6: per-frame image statistics on the model's ROI crop. */
object ImageQuality {

    fun measure(pixels: IntArray, width: Int, step: Int = 2): FrameQuality {
        val height = pixels.size / width
        var n = 0
        var sum = 0.0
        var sumSq = 0.0
        var saturated = 0
        var grad = 0.0
        var gradN = 0
        var y = 0
        while (y < height) {
            var prev = -1
            var x = 0
            val row = y * width
            while (x < width) {
                val l = luma(pixels[row + x])
                sum += l
                sumSq += l.toDouble() * l
                if (l >= FrameQuality.SATURATED_LUMA) saturated++
                if (prev >= 0) {
                    grad += abs(l - prev)
                    gradN++
                }
                prev = l
                n++
                x += step
            }
            y += step
        }
        if (n == 0) return FrameQuality(0f, 0f, 0f, 0f)
        val mean = sum / n
        val variance = (sumSq / n - mean * mean).coerceAtLeast(0.0)
        return FrameQuality(
            meanLuma = mean.toFloat(),
            contrast = sqrt(variance).toFloat(),
            saturated = saturated.toFloat() / n,
            sharpness = if (gradN == 0) 0f else (grad / gradN).toFloat(),
        )
    }

    private fun luma(argb: Int): Int {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        return (299 * r + 587 * g + 114 * b) / 1000
    }
}

/**
 * Debounces per-frame verdicts: the camera is declared unreliable only after
 * [badFrames] consecutive bad frames and recovers after [goodFrames] good ones.
 */
class CameraHealthMonitor(private val badFrames: Int = 8, private val goodFrames: Int = 4) {
    var health = CameraHealth.OK
        private set
    private var badRun = 0
    private var goodRun = 0

    fun push(verdict: CameraHealth): CameraHealth {
        if (verdict == CameraHealth.OK) {
            badRun = 0
            goodRun++
            if (health != CameraHealth.OK && goodRun >= goodFrames) health = CameraHealth.OK
        } else {
            goodRun = 0
            badRun++
            if (badRun >= badFrames || health != CameraHealth.OK) health = verdict
        }
        return health
    }
}
