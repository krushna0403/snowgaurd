package com.snowguard.app.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.roundToInt

data class Prediction(
    val surface: SurfaceClass,
    val confidence: Float,
    val probabilities: FloatArray,
    val inferenceMs: Float,
)

/**
 * Runs the MobileNetV2 road-surface model on a [inputSize] x [inputSize] RGB
 * bitmap. Handles both the INT8 model (uint8 in/out) and a float32 fallback.
 * Not thread-safe: call from a single analysis thread.
 */
class SurfaceClassifier private constructor(
    private val interpreter: Interpreter,
    val isQuantized: Boolean,
    private val classBias: FloatArray?,
    /** `int8`, `dynamic_int8` (INT8 weights, float activations) or `fp32`, from labels.json. */
    val quantization: String,
) : Closeable {

    val inputSize: Int
    private val input: ByteBuffer
    private val pixels: IntArray
    private val inputLut: FloatArray?
    private val outScale: Float
    private val outZero: Int
    private val outputBytes: ByteArray
    private val outputFloats: Array<FloatArray>

    init {
        val inTensor = interpreter.getInputTensor(0)
        inputSize = inTensor.shape()[1]
        input = ByteBuffer.allocateDirect(inTensor.numBytes()).order(ByteOrder.nativeOrder())
        pixels = IntArray(inputSize * inputSize)
        inputLut = if (isQuantized) {
            val q = inTensor.quantizationParams()
            FloatArray(256) { v -> (v / q.scale + q.zeroPoint).coerceIn(0f, 255f) }
        } else null

        val outTensor = interpreter.getOutputTensor(0)
        val numClasses = outTensor.shape()[1]
        require(numClasses == SurfaceClass.entries.size) {
            "Model has $numClasses outputs, app expects ${SurfaceClass.entries.size}"
        }
        outScale = outTensor.quantizationParams().scale
        outZero = outTensor.quantizationParams().zeroPoint
        outputBytes = ByteArray(numClasses)
        outputFloats = arrayOf(FloatArray(numClasses))
    }

    fun classify(rgb: Bitmap): Prediction {
        require(rgb.width == inputSize && rgb.height == inputSize)
        rgb.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        input.rewind()
        if (inputLut != null) {
            for (p in pixels) {
                input.put(inputLut[(p shr 16) and 0xFF].roundToInt().toByte())
                input.put(inputLut[(p shr 8) and 0xFF].roundToInt().toByte())
                input.put(inputLut[p and 0xFF].roundToInt().toByte())
            }
        } else {
            for (p in pixels) {
                input.putFloat(((p shr 16) and 0xFF).toFloat())
                input.putFloat(((p shr 8) and 0xFF).toFloat())
                input.putFloat((p and 0xFF).toFloat())
            }
        }
        input.rewind()

        val t0 = SystemClock.elapsedRealtimeNanos()
        val probs: FloatArray
        if (isQuantized) {
            val out = ByteBuffer.wrap(outputBytes)
            interpreter.run(input, out)
            probs = FloatArray(outputBytes.size) { i -> ((outputBytes[i].toInt() and 0xFF) - outZero) * outScale }
        } else {
            interpreter.run(input, outputFloats)
            probs = outputFloats[0].copyOf()
        }
        val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f

        classBias?.let { applyClassBias(probs, it) }
        var best = 0
        for (i in probs.indices) if (probs[i] > probs[best]) best = i
        return Prediction(SurfaceClass.fromIndex(best), probs[best], probs, ms)
    }

    override fun close() = interpreter.close()

    companion object {
        const val MODEL_ASSET = "snowguard_mnv2.tflite"
        const val LABELS_ASSET = "labels.json"
        private const val TAG = "SurfaceClassifier"
        private const val MIN_PROB = 1e-4f

        /**
         * Same as `apply_bias` in ml/snowguard_ml/export.py: shift log-probabilities
         * by the val-tuned per-class bias and renormalise, in place.
         */
        fun applyClassBias(probs: FloatArray, bias: FloatArray) {
            var sum = 0f
            for (i in probs.indices) {
                probs[i] = probs[i].coerceIn(MIN_PROB, 1f) * exp(bias[i])
                sum += probs[i]
            }
            for (i in probs.indices) probs[i] /= sum
        }

        private fun loadLabels(context: Context): JSONObject? = try {
            JSONObject(context.assets.open(LABELS_ASSET).bufferedReader().use { it.readText() })
        } catch (e: Exception) {
            Log.w(TAG, "Cannot read $LABELS_ASSET", e)
            null
        }

        private fun classBias(labels: JSONObject?, numClasses: Int): FloatArray? =
            labels?.optJSONArray("class_bias")
                ?.let { arr -> FloatArray(arr.length()) { arr.getDouble(it).toFloat() } }
                ?.takeIf { it.size == numClasses && it.any { b -> b != 0f } }

        /** Returns null if the model is missing or fails to load (app falls back to terrain-only). */
        fun create(
            context: Context,
            asset: String = MODEL_ASSET,
            quantization: String? = null,
            threads: Int = 4,
        ): SurfaceClassifier? = try {
            val buffer = context.assets.openFd(asset).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { ch ->
                    ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
            val interpreter = Interpreter(buffer, Interpreter.Options().setNumThreads(threads))
            val quantized = interpreter.getInputTensor(0).dataType() == DataType.UINT8
            val labels = loadLabels(context)
            SurfaceClassifier(
                interpreter, quantized, classBias(labels, SurfaceClass.entries.size),
                quantization
                    ?: labels?.optString("quantization")?.takeIf { it.isNotEmpty() }
                    ?: if (quantized) "int8" else "fp32",
            )
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed", e)
            null
        }
    }
}
