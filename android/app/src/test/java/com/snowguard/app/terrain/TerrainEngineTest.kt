package com.snowguard.app.terrain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TerrainEngineTest {

    private val nAz = 8
    private val stepM = 50.0
    private val mPerDeg = 111_320.0

    /** Straight north-going road at Keylong's latitude with a uniform horizon. */
    private fun straightRoute(n: Int, elev: Double, horizon: Float, slope: Double = 0.0, aspect: Double = 180.0) =
        RouteData(
            lat = DoubleArray(n) { 32.5 + it * stepM / mPerDeg },
            lon = DoubleArray(n) { 77.0 },
            distM = DoubleArray(n) { it * stepM },
            elevM = DoubleArray(n) { elev },
            headingDeg = DoubleArray(n) { 0.0 },
            curvatureDegPer100m = DoubleArray(n),
            slopeDeg = DoubleArray(n) { slope },
            aspectDeg = DoubleArray(n) { aspect },
            tunnel = BooleanArray(n),
            horizonDeg = FloatArray(n * nAz) { horizon },
            nAzimuth = nAz,
            thresholds = RouteData.DEFAULT_THRESHOLDS,
            landmarks = listOf(Landmark("Mid", "village", n / 2, elev)),
            name = "test",
        )

    // 2026-01-15 06:30 UTC (noon IST): sun ~35 deg; 2026-01-15 18:30 UTC: night.
    private val januaryNoon = 1768458600_000L
    private val januaryNight = 1768501800_000L
    private val julyNoon = 1784097000_000L

    private fun fixAt(route: RouteData, i: Int, t: Long) = RouteFix(route.lat[i], route.lon[i] + 0.0002, t)

    @Test
    fun lowHorizonIsSunlitAtNoon() {
        val r = straightRoute(100, 3500.0, horizon = 10f)
        val s = TerrainEngine(r).evaluate(fixAt(r, 40, januaryNoon))
        assertTrue(s.onRoute)
        assertEquals(true, s.sunlit)
        assertEquals(FreezeRisk.NONE, s.freezeRisk)
    }

    @Test
    fun highHorizonAboveWinterThresholdIsHighRisk() {
        val r = straightRoute(100, 3500.0, horizon = 45f)
        val s = TerrainEngine(r).evaluate(fixAt(r, 40, januaryNoon))
        assertEquals(false, s.sunlit)
        assertEquals("winter", s.threshold?.season)
        assertEquals(FreezeRisk.HIGH, s.freezeRisk) // shaded all day -> long shade
        assertTrue(s.minutesInShade!! >= 60)
    }

    @Test
    fun shadedButBelowThresholdHasNoRisk() {
        val r = straightRoute(100, 2500.0, horizon = 45f)
        val s = TerrainEngine(r).evaluate(fixAt(r, 40, januaryNoon))
        assertEquals(false, s.sunlit)
        assertEquals(FreezeRisk.NONE, s.freezeRisk)
    }

    @Test
    fun summerThresholdIsHigher() {
        val r = straightRoute(100, 4500.0, horizon = 89f)
        val s = TerrainEngine(r).evaluate(fixAt(r, 40, julyNoon))
        assertEquals("summer", s.threshold?.season)
        assertEquals(FreezeRisk.NONE, s.freezeRisk) // 4500 m < 4800 m summer threshold
    }

    @Test
    fun nightAboveThresholdIsHighRisk() {
        val r = straightRoute(100, 3200.0, horizon = 0f)
        val s = TerrainEngine(r).evaluate(fixAt(r, 40, januaryNight))
        assertEquals(false, s.sunlit)
        assertEquals(FreezeRisk.HIGH, s.freezeRisk)
    }

    @Test
    fun offCorridorIsIgnored() {
        val r = straightRoute(100, 3500.0, horizon = 45f)
        val s = TerrainEngine(r).evaluate(RouteFix(r.lat[40], r.lon[40] + 0.01, januaryNoon)) // ~940 m east
        assertTrue(!s.onRoute)
        assertNull(s.snap)
        assertEquals(FreezeRisk.NONE, s.freezeRisk)
    }

    @Test
    fun evaluationIsUnderOneMillisecondOnFullSizeRoute() {
        val r = straightRoute(6000, 3500.0, horizon = 20f)
        val engine = TerrainEngine(r)
        repeat(50) { engine.evaluate(fixAt(r, 3000 + it, januaryNoon + it * 1000L)) }
        var total = 0L
        repeat(200) { total += engine.evaluate(fixAt(r, 3050 + it, januaryNoon + (50 + it) * 1000L)).computeMicros }
        assertTrue("mean ${total / 200} us", total / 200 < 1000)
    }

    @Test
    fun snapperFollowsHairpinLegBeingDriven() {
        // Leg A goes north 1 km, a U-turn, leg B comes back south 40 m to the east.
        val legN = 21
        val lat = DoubleArray(legN * 2) { i -> 32.5 + (if (i < legN) i else 2 * legN - 1 - i) * stepM / mPerDeg }
        val lon = DoubleArray(legN * 2) { i -> 77.0 + if (i < legN) 0.0 else 40.0 / (mPerDeg * 0.8434) }
        val heading = DoubleArray(legN * 2) { i -> if (i < legN) 0.0 else 180.0 }
        val n = lat.size
        val r = RouteData(
            lat, lon, DoubleArray(n) { it * stepM }, DoubleArray(n) { 3500.0 }, heading, DoubleArray(n),
            DoubleArray(n), DoubleArray(n), BooleanArray(n), FloatArray(n * nAz), nAz,
            RouteData.DEFAULT_THRESHOLDS, emptyList(), "hairpin",
        )
        val snapper = RouteSnapper(r)
        var t = januaryNoon
        // Drive up leg A, round the U-turn and down leg B at 10 m/s. On leg B the GPS is
        // biased 22 m towards leg A (18 m away), so raw distance alone would pick the wrong leg.
        var snap: Snap? = null
        for (i in 0 until n - 2) {
            val onB = i > legN
            val bias = if (onB) -22.0 / (mPerDeg * 0.8434) else 0.0
            snap = snapper.snap(RouteFix(lat[i], lon[i] + bias, t, heading[i], 10.0))
            assertNotNull(snap)
            if (onB && i > legN + 3) assertTrue("fix $i snapped to leg A sample ${snap!!.sample}", snap.sample >= legN)
            t += 5_000
        }
        assertEquals(1, snap!!.direction)
    }
}
