package com.snowguard.app.vision

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.random.Random

class ImageQualityTest {

    private val size = 224

    private fun gray(v: Int) = (0xFF shl 24) or (v shl 16) or (v shl 8) or v

    private fun image(fn: (x: Int, y: Int) -> Int) = IntArray(size * size) { fn(it % size, it / size).coerceIn(0, 255).let(::gray) }

    private fun verdict(px: IntArray) = ImageQuality.measure(px, size).verdict

    @Test
    fun texturedRoadIsOk() {
        val rnd = Random(1)
        assertEquals(CameraHealth.OK, verdict(image { _, _ -> 90 + rnd.nextInt(60) }))
    }

    @Test
    fun brightSnowWithTextureIsNotGlare() {
        val rnd = Random(2)
        assertEquals(CameraHealth.OK, verdict(image { _, _ -> 200 + rnd.nextInt(45) }))
    }

    @Test
    fun darkFrame() {
        val rnd = Random(3)
        assertEquals(CameraHealth.DARK, verdict(image { _, _ -> rnd.nextInt(30) }))
    }

    @Test
    fun sunGlare() {
        assertEquals(CameraHealth.GLARE, verdict(image { x, _ -> if (x < size * 0.6) 255 else 120 }))
    }

    @Test
    fun foggedLensIsFlatAndSmooth() {
        assertEquals(CameraHealth.BLURRED, verdict(image { x, _ -> 150 + x / 50 }))
    }

    @Test
    fun monitorDebouncesBadAndGoodRuns() {
        val m = CameraHealthMonitor(badFrames = 8, goodFrames = 4)
        repeat(7) { assertEquals(CameraHealth.OK, m.push(CameraHealth.GLARE)) }
        assertEquals(CameraHealth.OK, m.push(CameraHealth.OK)) // run broken
        repeat(7) { m.push(CameraHealth.DARK) }
        assertEquals(CameraHealth.DARK, m.push(CameraHealth.DARK))
        repeat(3) { assertEquals(CameraHealth.DARK, m.push(CameraHealth.OK)) }
        assertEquals(CameraHealth.OK, m.push(CameraHealth.OK))
    }
}
