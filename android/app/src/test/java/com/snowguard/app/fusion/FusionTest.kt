package com.snowguard.app.fusion

import com.snowguard.app.terrain.FreezeRisk
import com.snowguard.app.vision.SurfaceClass
import com.snowguard.app.vision.TemporalSmoother
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FusionTest {

    @Test
    fun tieredSeverityTable() {
        fun s(v: SurfaceClass?, t: FreezeRisk?) = FusionEngine.assess(v, t).severity
        assertEquals(Severity.NONE, s(SurfaceClass.DRY, FreezeRisk.NONE))
        assertEquals(Severity.NONE, s(null, null))
        assertEquals(Severity.ADVISORY, s(SurfaceClass.DRY, FreezeRisk.HIGH))
        assertEquals(Severity.ADVISORY, s(null, FreezeRisk.ELEVATED)) // camera lost: terrain-only
        assertEquals(Severity.WARNING, s(SurfaceClass.ICE, FreezeRisk.NONE))
        assertEquals(Severity.WARNING, s(SurfaceClass.FRESH_SNOW, null)) // GPS lost: vision-only
        assertEquals(Severity.CRITICAL, s(SurfaceClass.ICE, FreezeRisk.HIGH))
        assertEquals(Severity.NONE, s(SurfaceClass.WET, FreezeRisk.NONE))
    }

    @Test
    fun wetInFreezeZoneIsBlackIceCritical() {
        val a = FusionEngine.assess(SurfaceClass.WET, FreezeRisk.ELEVATED)
        assertEquals(Severity.CRITICAL, a.severity)
        assertEquals(Hazard.BLACK_ICE_RISK, a.hazard)
        assertTrue(a.hazard!!.isIce)
    }

    @Test
    fun orGateFlagsEverythingAtOneLevel() {
        val terrainOnly = FusionEngine.assess(SurfaceClass.DRY, FreezeRisk.HIGH, FusionMode.OR_GATE)
        val both = FusionEngine.assess(SurfaceClass.ICE, FreezeRisk.HIGH, FusionMode.OR_GATE)
        val wetShade = FusionEngine.assess(SurfaceClass.WET, FreezeRisk.HIGH, FusionMode.OR_GATE)
        assertEquals(Severity.WARNING, terrainOnly.severity)
        assertEquals(Severity.WARNING, both.severity)
        assertEquals(Hazard.FREEZE_ZONE, wetShade.hazard) // OR gate cannot see the black-ice signature
    }

    @Test
    fun gateSuppressesRepeatsButAllowsEscalation() {
        val gate = AlertGate()
        val warn = FusionEngine.assess(SurfaceClass.ICE, null)
        val crit = FusionEngine.assess(SurfaceClass.ICE, FreezeRisk.HIGH)
        assertTrue(gate.shouldFire(warn, 0))
        assertFalse(gate.shouldFire(warn, 10_000))
        assertTrue(gate.shouldFire(crit, 12_000)) // escalation
        assertFalse(gate.shouldFire(crit, 20_000))
        assertFalse(gate.shouldFire(warn, 40_000)) // de-escalation within critical cooldown
        assertTrue(gate.shouldFire(crit, 32_001))
        assertFalse(gate.shouldFire(Assessment.CLEAR, 100_000))
    }

    @Test
    fun gateTracksHazardsIndependently() {
        val gate = AlertGate()
        assertTrue(gate.shouldFire(FusionEngine.assess(SurfaceClass.ICE, null), 0))
        assertTrue(gate.shouldFire(FusionEngine.assess(SurfaceClass.WATER, null), 1_000))
        assertTrue(gate.shouldFire(FusionEngine.assess(null, FreezeRisk.HIGH), 2_000))
        assertFalse(gate.shouldFire(FusionEngine.assess(null, FreezeRisk.HIGH), 100_000))
        assertTrue(gate.shouldFire(FusionEngine.assess(null, FreezeRisk.HIGH), 182_000))
    }

    @Test
    fun smootherNeedsFourOfSixAndHoldsWithHysteresis() {
        val sm = TemporalSmoother()
        val ice = SurfaceClass.ICE
        val dry = SurfaceClass.DRY
        listOf(ice, dry, ice, dry, ice).forEach { assertNull(sm.push(it)) } // 3 of 5
        assertEquals(ice, sm.push(ice)) // 4 of 6
        assertEquals(ice, sm.push(dry)) // ice 3, dry 3: held
        assertEquals(ice, sm.push(dry)) // ice 3, dry 3: held
        assertEquals(dry, sm.push(dry)) // dry 4 > ice 2: dry takes over
    }

    @Test
    fun smootherIgnoresSingleFrameFlicker() {
        val sm = TemporalSmoother()
        repeat(6) { sm.push(SurfaceClass.DRY) }
        assertEquals(SurfaceClass.DRY, sm.push(SurfaceClass.ICE))
        assertEquals(SurfaceClass.DRY, sm.push(null))
        assertEquals(SurfaceClass.DRY, sm.push(SurfaceClass.DRY))
    }

    @Test
    fun smootherClearsWhenFramesBecomeUnreliable() {
        val sm = TemporalSmoother()
        repeat(6) { sm.push(SurfaceClass.ICE) }
        repeat(4) { sm.push(null) }
        assertEquals(SurfaceClass.ICE, sm.stable) // 2 of 6 left
        assertNull(sm.push(null))
    }
}
