package com.snowguard.app.fusion

import com.snowguard.app.terrain.FreezeRisk
import com.snowguard.app.terrain.LookAhead
import com.snowguard.app.vision.SurfaceClass

enum class Severity { NONE, ADVISORY, WARNING, CRITICAL }

enum class Hazard(val isIce: Boolean) {
    ICE(true),
    SNOW(false),
    SLUSH(false),
    WATER(false),
    /** Wet road inside a shaded freeze zone: the black-ice signature (FR-F2). */
    BLACK_ICE_RISK(true),
    /** Terrain-only: shaded and above the altitude-month threshold. */
    FREEZE_ZONE(false),
    /** Look-ahead (FR-L1): a shaded freeze-prone stretch within the next 200-500 m. */
    FREEZE_AHEAD(false),
    /** Look-ahead (FR-L2): that stretch starts on a sharp bend. */
    CORNER_AHEAD(false),
    ;

    /** Hazards sharing a key share one cooldown, so "zone ahead" is not followed by "in zone". */
    val cooldownKey: Hazard get() = if (this == FREEZE_AHEAD) FREEZE_ZONE else this

    companion object {
        fun fromSurface(surface: SurfaceClass): Hazard? = when (surface) {
            SurfaceClass.ICE -> ICE
            SurfaceClass.FRESH_SNOW -> SNOW
            SurfaceClass.MELTED_SNOW -> SLUSH
            SurfaceClass.WATER -> WATER
            SurfaceClass.DRY, SurfaceClass.WET -> null
        }
    }
}

/** Ablation switch: the BRD baseline OR gate (FR-F1) vs tiered fusion (FR-F2). */
enum class FusionMode { OR_GATE, TIERED }

data class Assessment(
    val severity: Severity,
    val hazard: Hazard?,
    val visionAlert: Boolean,
    val terrainAlert: Boolean,
    /** Distance to a look-ahead hazard, rounded for speech. */
    val distanceM: Int? = null,
) {
    companion object {
        val CLEAR = Assessment(Severity.NONE, null, visionAlert = false, terrainAlert = false)
    }
}

/**
 * Decision-level fusion of the smoothed vision class and the terrain freeze
 * risk. Either input may be null when that layer is unavailable (camera loss,
 * GPS loss, off-corridor), in which case the other layer decides alone.
 */
object FusionEngine {

    fun assess(vision: SurfaceClass?, terrain: FreezeRisk?, mode: FusionMode = FusionMode.TIERED): Assessment {
        val visionHazard = vision?.let { Hazard.fromSurface(it) }
        val visionAlert = visionHazard != null
        val terrainAlert = terrain != null && terrain != FreezeRisk.NONE

        return when (mode) {
            FusionMode.OR_GATE -> when {
                visionAlert -> Assessment(Severity.WARNING, visionHazard, true, terrainAlert)
                terrainAlert -> Assessment(Severity.WARNING, Hazard.FREEZE_ZONE, false, true)
                else -> Assessment.CLEAR
            }
            FusionMode.TIERED -> when {
                visionAlert && terrainAlert -> Assessment(Severity.CRITICAL, visionHazard, true, true)
                vision == SurfaceClass.WET && terrainAlert ->
                    Assessment(Severity.CRITICAL, Hazard.BLACK_ICE_RISK, false, true)
                visionAlert -> Assessment(Severity.WARNING, visionHazard, true, false)
                terrainAlert -> Assessment(Severity.ADVISORY, Hazard.FREEZE_ZONE, false, true)
                else -> Assessment.CLEAR
            }
        }
    }

    /** A look-ahead hazard only speaks when nothing is happening here and now. */
    fun withLookAhead(current: Assessment, ahead: LookAhead?): Assessment {
        if (current.severity != Severity.NONE || ahead == null || ahead.risk == FreezeRisk.NONE) return current
        val rounded = (((ahead.distanceM + 25) / 50).toInt() * 50).coerceAtLeast(50)
        return Assessment(
            Severity.ADVISORY,
            if (ahead.blindCorner) Hazard.CORNER_AHEAD else Hazard.FREEZE_AHEAD,
            visionAlert = false,
            terrainAlert = true,
            distanceM = rounded,
        )
    }
}

/**
 * FR-F6: suppresses repeats of the same hazard. A hazard re-fires only after
 * its severity's cooldown, unless it escalates to a higher severity.
 */
class AlertGate(
    private val cooldownMs: Map<Severity, Long> = DEFAULT_COOLDOWN_MS,
) {
    private data class Fired(val severity: Severity, val atMs: Long)

    private val last = HashMap<Hazard, Fired>()

    fun shouldFire(a: Assessment, nowMs: Long): Boolean {
        val hazard = a.hazard?.cooldownKey ?: return false
        if (a.severity == Severity.NONE) return false
        val prev = last[hazard]
        val fire = prev == null || a.severity > prev.severity ||
            nowMs - prev.atMs >= (cooldownMs[a.severity] ?: 0L)
        if (fire) last[hazard] = Fired(a.severity, nowMs)
        return fire
    }

    fun reset() = last.clear()

    companion object {
        val DEFAULT_COOLDOWN_MS = mapOf(
            Severity.ADVISORY to 180_000L,
            Severity.WARNING to 45_000L,
            Severity.CRITICAL to 20_000L,
        )
    }
}
