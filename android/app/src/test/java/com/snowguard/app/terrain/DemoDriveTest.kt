package com.snowguard.app.terrain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoDriveTest {

    private val stepM = 75.0
    private val n = 200
    private val nAz = 8

    private val route = RouteData(
        lat = DoubleArray(n) { 32.5 + it * stepM / 111_320.0 },
        lon = DoubleArray(n) { 77.0 },
        distM = DoubleArray(n) { it * stepM },
        elevM = DoubleArray(n) { 3500.0 },
        headingDeg = DoubleArray(n),
        curvatureDegPer100m = DoubleArray(n),
        slopeDeg = DoubleArray(n),
        aspectDeg = DoubleArray(n),
        tunnel = BooleanArray(n),
        horizonDeg = FloatArray(n * nAz),
        nAzimuth = nAz,
        thresholds = RouteData.DEFAULT_THRESHOLDS,
        landmarks = emptyList(),
        name = "straight",
    )

    @Test
    fun interpolatesBetweenSamples() {
        val drive = DemoDrive(route, startM = 1000.0, endM = 5000.0, speedMps = 10.0, timeScale = 1.0, simStartMillis = 0L)
        val fix = drive.fixAt(5.0)!!
        val expectedLat = 32.5 + 1050.0 / 111_320.0
        assertEquals(expectedLat, fix.lat, 1e-7)
        assertEquals(5_000L, fix.timeMillis)
        assertEquals(10.0, fix.speedMps!!, 1e-9)
    }

    @Test
    fun timeScaleAdvancesSimulatedClockAndDistance() {
        val drive = DemoDrive(route, startM = 0.0, endM = 10_000.0, speedMps = 10.0, timeScale = 4.0, simStartMillis = 1_000L)
        drive.next()
        drive.next()
        assertEquals(80.0, drive.distanceM, 1e-9)
        assertEquals(9_000L, drive.simTimeMillis)
    }

    @Test
    fun stopsAtEndOfStretch() {
        val drive = DemoDrive(route, startM = 0.0, endM = 300.0, speedMps = 100.0, timeScale = 1.0, simStartMillis = 0L)
        assertNotNull(drive.next())
        assertNotNull(drive.next())
        assertNotNull(drive.next())
        assertNotNull(drive.next())
        assertNull(drive.next())
        assertTrue(drive.finished)
    }

    @Test
    fun demoFixesSnapOntoTheRouteWithLockedDirection() {
        val engine = TerrainEngine(route)
        val drive = DemoDrive(route, startM = 1000.0, endM = 12_000.0, speedMps = 11.1, timeScale = 4.0,
            simStartMillis = DemoDrive.defaultStartMillis())
        var last: TerrainState? = null
        repeat(30) { last = engine.evaluate(drive.next()!!) }
        val s = last!!
        assertTrue(s.onRoute)
        assertEquals(1, s.snap!!.direction)
        assertEquals(1000.0 + 29 * 4 * 11.1, s.snap!!.distAlongM, stepM)
    }
}
