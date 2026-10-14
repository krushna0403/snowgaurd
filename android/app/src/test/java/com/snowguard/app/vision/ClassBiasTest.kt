package com.snowguard.app.vision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.exp

class ClassBiasTest {

    @Test
    fun zeroBiasKeepsDistribution() {
        val p = floatArrayOf(0.1f, 0.2f, 0.3f, 0.15f, 0.05f, 0.2f)
        val q = p.copyOf()
        SurfaceClassifier.applyClassBias(q, FloatArray(6))
        assertArrayEquals(p, q, 1e-6f)
    }

    @Test
    fun matchesPythonApplyBias() {
        // ml/snowguard_ml/export.py apply_bias: softmax(log(clip(p, 1e-4)) + b)
        val p = floatArrayOf(0.45f, 0.05f, 0f, 0.02f, 0.08f, 0.40f)
        val b = floatArrayOf(0f, -0.5f, 0.1f, 0f, 0.2f, 0.35f)
        val expected = FloatArray(6) { maxOf(p[it], 1e-4f) * exp(b[it]) }.let { e -> val s = e.sum(); FloatArray(6) { e[it] / s } }
        val q = p.copyOf()
        SurfaceClassifier.applyClassBias(q, b)
        assertArrayEquals(expected, q, 1e-6f)
        assertEquals(1f, q.sum(), 1e-5f)
    }

    @Test
    fun iceBiasCanFlipCloseCall() {
        val q = floatArrayOf(0.45f, 0.05f, 0.01f, 0.02f, 0.07f, 0.40f)
        SurfaceClassifier.applyClassBias(q, floatArrayOf(0f, 0f, 0f, 0f, 0f, 0.2f))
        assertEquals(SurfaceClass.ICE.ordinal, q.indices.maxBy { q[it] })
    }
}
