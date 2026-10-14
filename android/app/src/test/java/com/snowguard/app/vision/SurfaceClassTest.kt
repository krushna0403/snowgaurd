package com.snowguard.app.vision

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

class SurfaceClassTest {

    @Test
    fun orderMatchesModelOutput() {
        val expected = listOf("DRY", "WET", "WATER", "FRESH_SNOW", "MELTED_SNOW", "ICE")
        assertEquals(expected, SurfaceClass.entries.map { it.name })
    }

    @Test
    fun orderMatchesBundledLabelsJson() {
        val json = File("src/main/assets/labels.json").readText()
        val classes = Regex("\"classes\"\\s*:\\s*\\[([^]]*)]").find(json)!!.groupValues[1]
            .split(",").map { it.trim().trim('"').uppercase() }
        assertEquals(classes, SurfaceClass.entries.map { it.name })
    }

    @Test
    fun hazardousClassesAreNotGreen() {
        SurfaceClass.entries.filter { it.raisesAlert }.forEach {
            assert(it.band != FrictionBand.GREEN) { "$it is an alert class but has a green band" }
        }
    }
}
