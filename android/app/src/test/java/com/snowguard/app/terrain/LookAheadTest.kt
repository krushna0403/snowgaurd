package com.snowguard.app.terrain

import com.snowguard.app.fusion.AlertGate
import com.snowguard.app.fusion.Assessment
import com.snowguard.app.fusion.FusionEngine
import com.snowguard.app.fusion.Hazard
import com.snowguard.app.fusion.Severity
import com.snowguard.app.vision.SurfaceClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LookAheadTest {

    private val nAz = 8
    private val step = 50.0
    private val winter = RouteData.DEFAULT_THRESHOLDS.first()
    private val noonSun = SunPosition(elevationDeg = 35.0, azimuthDeg = 180.0)

    /** 100 samples, 50 m apart; from [shadeFrom] on, the horizon is 45 deg (shaded at noon). */
    private fun route(shadeFrom: Int, elev: Double = 3500.0, cornerAt: Int? = null): RouteData {
        val n = 100
        return RouteData(
            lat = DoubleArray(n) { 32.5 + it * step / 111_320.0 },
            lon = DoubleArray(n) { 77.0 },
            distM = DoubleArray(n) { it * step },
            elevM = DoubleArray(n) { elev },
            headingDeg = DoubleArray(n),
            curvatureDegPer100m = DoubleArray(n) { if (it == cornerAt) 80.0 else 5.0 },
            slopeDeg = DoubleArray(n),
            aspectDeg = DoubleArray(n) { 180.0 },
            tunnel = BooleanArray(n),
            horizonDeg = FloatArray(n * nAz) { if (it / nAz >= shadeFrom) 45f else 5f },
            nAzimuth = nAz,
            thresholds = RouteData.DEFAULT_THRESHOLDS,
            landmarks = emptyList(),
            name = "lookahead",
        )
    }

    private fun snapAt(i: Int, direction: Int) = Snap(i, 0.0, i * step, direction)

    @Test
    fun findsShadedZoneWithinWindow() {
        val engine = TerrainEngine(route(shadeFrom = 60))
        val ahead = engine.lookAhead(snapAt(55, 1), noonSun, winter, speedMps = 12.0) // 360 m window
        assertNotNull(ahead)
        assertEquals(250.0, ahead!!.distanceM, 1e-6)
        assertEquals(FreezeRisk.ELEVATED, ahead.risk)
        assertFalse(ahead.blindCorner)
    }

    @Test
    fun windowScalesWithSpeedButStaysWithin200To500() {
        val engine = TerrainEngine(route(shadeFrom = 60))
        assertNull(engine.lookAhead(snapAt(55, 1), noonSun, winter, speedMps = 3.0)) // 200 m min < 250 m
        assertNull(engine.lookAhead(snapAt(45, 1), noonSun, winter, speedMps = 40.0)) // 750 m > 500 m cap
        assertNotNull(engine.lookAhead(snapAt(51, 1), noonSun, winter, speedMps = 40.0)) // 450 m
    }

    @Test
    fun respectsTravelDirection() {
        val engine = TerrainEngine(route(shadeFrom = 60))
        assertNull(engine.lookAhead(snapAt(55, -1), noonSun, winter, 12.0)) // driving away from the zone
        assertNull(engine.lookAhead(snapAt(55, 0), noonSun, winter, 12.0)) // direction unknown
    }

    @Test
    fun belowThresholdOrSunlitIsNotAHazard() {
        assertNull(TerrainEngine(route(60, elev = 2500.0)).lookAhead(snapAt(55, 1), noonSun, winter, 12.0))
        assertNull(TerrainEngine(route(shadeFrom = 100)).lookAhead(snapAt(55, 1), noonSun, winter, 12.0))
    }

    @Test
    fun flagsShadedBlindCorner() {
        val ahead = TerrainEngine(route(shadeFrom = 60, cornerAt = 61)).lookAhead(snapAt(55, 1), noonSun, winter, 12.0)
        assertTrue(ahead!!.blindCorner)
        val a = FusionEngine.withLookAhead(Assessment.CLEAR, ahead)
        assertEquals(Hazard.CORNER_AHEAD, a.hazard)
        assertEquals(250, a.distanceM)
    }

    @Test
    fun lookAheadOnlySpeaksWhenNothingIsHappeningNow() {
        val ahead = LookAhead(330.0, FreezeRisk.ELEVATED, blindCorner = false, windowM = 400.0)
        val clear = FusionEngine.withLookAhead(Assessment.CLEAR, ahead)
        assertEquals(Severity.ADVISORY, clear.severity)
        assertEquals(Hazard.FREEZE_AHEAD, clear.hazard)
        assertEquals(350, clear.distanceM) // rounded to 50 m for speech
        val warning = FusionEngine.assess(SurfaceClass.ICE, null)
        assertEquals(warning, FusionEngine.withLookAhead(warning, ahead))
    }

    @Test
    fun zoneAheadAndInZoneShareCooldown() {
        val gate = AlertGate()
        val ahead = FusionEngine.withLookAhead(Assessment.CLEAR, LookAhead(300.0, FreezeRisk.HIGH, false, 400.0))
        val inZone = FusionEngine.assess(null, FreezeRisk.HIGH)
        assertTrue(gate.shouldFire(ahead, 0))
        assertFalse(gate.shouldFire(inZone, 30_000)) // entering the zone 30 s later stays quiet
    }
}
