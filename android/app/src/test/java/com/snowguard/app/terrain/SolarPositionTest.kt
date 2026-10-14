package com.snowguard.app.terrain

import org.junit.Assert.assertEquals
import org.junit.Test

/** Reference values from pvlib's NREL SPA (terrain/snowguard_terrain/validate_solar.py). */
class SolarPositionTest {

    private data class Fixture(val lat: Double, val lon: Double, val epochSec: Long, val elev: Double, val az: Double)

    private val fixtures = listOf(
        Fixture(32.2396, 77.1887, 1768451400L, 25.6548, 140.8241), // Manali 2026-01-15 04:30Z
        Fixture(32.2396, 77.1887, 1768458600L, 36.1669, 171.1653), // Manali 2026-01-15 06:30Z
        Fixture(32.2396, 77.1887, 1773990000L, 57.6382, 180.5871), // Manali 2026-03-20 07:00Z
        Fixture(32.2396, 77.1887, 1782007200L, 25.8691, 77.4660), // Manali 2026-06-21 02:00Z
        Fixture(32.2396, 77.1887, 1782023400L, 79.8407, 148.5790), // Manali 2026-06-21 06:30Z
        Fixture(32.2396, 77.1887, 1791454500L, 26.0275, 243.6126), // Manali 2026-10-08 10:15Z
        Fixture(32.2396, 77.1887, 1797829200L, 28.5439, 151.3929), // Manali 2026-12-21 05:00Z
        Fixture(32.5711, 77.0322, 1768451400L, 25.3114, 140.7912), // Keylong 2026-01-15 04:30Z
        Fixture(32.5711, 77.0322, 1782023400L, 79.4885, 148.8204), // Keylong 2026-06-21 06:30Z
        Fixture(32.91, 77.58, 1773990000L, 56.9606, 181.2940), // Sarchu 2026-03-20 07:00Z
        Fixture(32.91, 77.58, 1791454500L, 25.4280, 243.6047), // Sarchu 2026-10-08 10:15Z
        Fixture(33.508, 77.77, 1782007200L, 26.6025, 78.3419), // Tanglang La 2026-06-21 02:00Z
        Fixture(33.508, 77.77, 1797829200L, 27.6485, 152.2575), // Tanglang La 2026-12-21 05:00Z
        Fixture(34.1526, 77.5771, 1768458600L, 34.3211, 171.8050), // Leh 2026-01-15 06:30Z
        Fixture(34.1526, 77.5771, 1782023400L, 78.3065, 154.9646), // Leh 2026-06-21 06:30Z
    )

    @Test
    fun matchesNrelSpaWithinBudget() {
        for (f in fixtures) {
            val sun = SolarPosition.compute(f.lat, f.lon, f.epochSec * 1000)
            assertEquals("elevation $f", f.elev, sun.elevationDeg, 0.1)
            val dAz = ((sun.azimuthDeg - f.az + 540.0) % 360.0) - 180.0
            assertEquals("azimuth $f", 0.0, dAz, 0.1)
        }
    }

    @Test
    fun sunIsDownAtNight() {
        // 2026-01-15 18:30 UTC = midnight IST in Keylong.
        val sun = SolarPosition.compute(32.5711, 77.0322, 1768501800_000L)
        assert(sun.elevationDeg < -30) { "expected deep night, got ${sun.elevationDeg}" }
    }

    @Test
    fun computeIsFast() {
        val t0 = System.nanoTime()
        repeat(10_000) { SolarPosition.compute(32.5, 77.0, 1768451400_000L + it * 60_000L) }
        val perCallMicros = (System.nanoTime() - t0) / 10_000 / 1000.0
        assert(perCallMicros < 100) { "solar compute too slow: $perCallMicros us" }
    }
}
