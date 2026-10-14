package com.snowguard.app.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RoadRoiTest {

    private fun assertInside(roi: RoiRect, w: Float, h: Float) {
        assertTrue(roi.left >= 0f && roi.top >= 0f)
        assertTrue(roi.right <= w + 1e-3f && roi.bottom <= h + 1e-3f)
    }

    @Test
    fun portraitFrameKeepsRscdAspectAndSitsAboveBonnet() {
        val roi = RoadRoi.compute(480f, 640f)
        assertEquals(240f / 360f, roi.height / roi.width, 1e-4f)
        assertEquals(640f * 0.9f, roi.bottom, 1e-3f)
        assertEquals((480f - roi.width) / 2f, roi.left, 1e-3f)
        assertInside(roi, 480f, 640f)
    }

    @Test
    fun landscapeFrameFits() {
        val roi = RoadRoi.compute(640f, 480f)
        assertEquals(240f / 360f, roi.height / roi.width, 1e-4f)
        assertInside(roi, 640f, 480f)
    }

    @Test
    fun veryWideFrameShrinksWidthInsteadOfOverflowing() {
        val roi = RoadRoi.compute(2000f, 300f)
        assertEquals(240f / 360f, roi.height / roi.width, 1e-4f)
        assertInside(roi, 2000f, 300f)
    }
}
