package com.snowguard.app.camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.snowguard.app.vision.CameraHealth
import com.snowguard.app.vision.CameraHealthMonitor
import com.snowguard.app.vision.FrameQuality
import com.snowguard.app.vision.ImageQuality
import com.snowguard.app.vision.Prediction
import com.snowguard.app.vision.RoadRoi
import com.snowguard.app.vision.SurfaceClassifier

data class FrameStats(
    val fps: Float,
    val inferenceFps: Float,
    val uprightWidth: Int,
    val uprightHeight: Int,
    val meanLuma: Float,
)

data class FrameResult(
    val stats: FrameStats,
    val prediction: Prediction?,
    /** True only on frames where a new inference ran (prediction is fresh). */
    val inferred: Boolean,
    val health: CameraHealth,
    val quality: FrameQuality?,
)

/**
 * Receives RGBA_8888 frames from CameraX on a background thread, crops the
 * road ROI, resizes it to the model input, checks image quality and classifies
 * it at most every [minInferenceIntervalMs] (10-15 fps is plenty; the activity
 * raises the interval when stopped or when the phone runs hot, FR-V7).
 */
class FrameAnalyzer(
    private val classifier: SurfaceClassifier?,
    private val onResult: (FrameResult) -> Unit,
) : ImageAnalysis.Analyzer {

    @Volatile
    var minInferenceIntervalMs: Long = DEFAULT_INTERVAL_MS

    private val fpsCounter = RateCounter()
    private val inferenceCounter = RateCounter()
    private var lastInferenceMs = 0L
    private var lastPrediction: Prediction? = null
    private var lastQuality: FrameQuality? = null
    private val healthMonitor = CameraHealthMonitor()

    private val modelInput: Bitmap? = classifier?.let {
        Bitmap.createBitmap(it.inputSize, it.inputSize, Bitmap.Config.ARGB_8888)
    }
    private val roiPixels = classifier?.let { IntArray(it.inputSize * it.inputSize) }
    private val canvas = modelInput?.let { Canvas(it) }
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()
    private val bounds = RectF()

    override fun analyze(image: ImageProxy) {
        try {
            val now = SystemClock.elapsedRealtime()
            fpsCounter.tick(now)
            val rotation = image.imageInfo.rotationDegrees
            val sideways = rotation % 180 != 0
            val uprightW = if (sideways) image.height else image.width
            val uprightH = if (sideways) image.width else image.height

            var inferred = false
            if (classifier != null && now - lastInferenceMs >= minInferenceIntervalMs) {
                lastInferenceMs = now
                inferenceCounter.tick(now)
                val frame = image.toBitmap()
                cropRoadRoi(frame, rotation)
                frame.recycle()
                val input = modelInput!!
                input.getPixels(roiPixels!!, 0, input.width, 0, 0, input.width, input.height)
                val quality = ImageQuality.measure(roiPixels, input.width)
                lastQuality = quality
                healthMonitor.push(quality.verdict)
                lastPrediction = classifier.classify(input)
                inferred = true
            }

            onResult(
                FrameResult(
                    FrameStats(fpsCounter.rate, inferenceCounter.rate, uprightW, uprightH, meanLuma(image)),
                    lastPrediction,
                    inferred,
                    healthMonitor.health,
                    lastQuality,
                )
            )
        } finally {
            image.close()
        }
    }

    /** Rotate upright, crop the road ROI and scale it into [modelInput] in one draw. */
    private fun cropRoadRoi(frame: Bitmap, rotationDegrees: Int) {
        val size = modelInput!!.width.toFloat()
        matrix.reset()
        matrix.postRotate(rotationDegrees.toFloat())
        bounds.set(0f, 0f, frame.width.toFloat(), frame.height.toFloat())
        matrix.mapRect(bounds)
        matrix.postTranslate(-bounds.left, -bounds.top)
        val roi = RoadRoi.compute(bounds.width(), bounds.height())
        matrix.postTranslate(-roi.left, -roi.top)
        matrix.postScale(size / roi.width, size / roi.height)
        canvas!!.drawBitmap(frame, matrix, paint)
    }

    /** Mean Rec.601 luma (0-255) over a sparse pixel grid. */
    private fun meanLuma(image: ImageProxy): Float {
        val plane = image.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        var sum = 0L
        var n = 0
        val step = 16
        var y = 0
        while (y < image.height) {
            var x = 0
            val row = y * rowStride
            while (x < image.width) {
                val i = row + x * pixelStride
                val r = buf.get(i).toInt() and 0xFF
                val g = buf.get(i + 1).toInt() and 0xFF
                val b = buf.get(i + 2).toInt() and 0xFF
                sum += (299 * r + 587 * g + 114 * b) / 1000
                n++
                x += step
            }
            y += step
        }
        return if (n == 0) 0f else sum.toFloat() / n
    }

    companion object {
        const val DEFAULT_INTERVAL_MS = 70L
    }

    private class RateCounter {
        private var windowStart = SystemClock.elapsedRealtime()
        private var count = 0
        var rate = 0f
            private set

        fun tick(now: Long) {
            count++
            val elapsed = now - windowStart
            if (elapsed >= 1000) {
                rate = count * 1000f / elapsed
                count = 0
                windowStart = now
            }
        }
    }
}
