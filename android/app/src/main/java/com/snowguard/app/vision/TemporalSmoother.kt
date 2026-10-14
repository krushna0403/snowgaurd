package com.snowguard.app.vision

/**
 * FR-V5: K-of-N vote over recent frames with hysteresis. A class becomes the
 * stable surface once [raiseK] of the last [window] confident frames agree, and
 * stays until it drops below [clearK]; this stops single-frame flicker from
 * raising or dropping alerts. Frames that are low-confidence or come from an
 * unreliable camera are pushed as null and count for nothing.
 */
class TemporalSmoother(
    private val window: Int = 6,
    raiseK: Int = 4,
    private val clearK: Int = 2,
) {
    var raiseK: Int = raiseK
        set(value) {
            field = value.coerceIn(1, window)
        }

    private val history = arrayOfNulls<SurfaceClass>(window)
    private val counts = IntArray(SurfaceClass.entries.size)
    private var next = 0
    private var filled = 0

    var stable: SurfaceClass? = null
        private set

    fun push(surface: SurfaceClass?): SurfaceClass? {
        if (filled == window) history[next]?.let { counts[it.ordinal]-- } else filled++
        history[next] = surface
        surface?.let { counts[it.ordinal]++ }
        next = (next + 1) % window

        val current = stable
        if (current != null && counts[current.ordinal] >= clearK) {
            // Hold the current class unless another one has a clear majority.
            val challenger = leader()
            if (challenger != null && challenger != current && counts[challenger.ordinal] >= raiseK &&
                counts[challenger.ordinal] > counts[current.ordinal]
            ) stable = challenger
        } else {
            stable = leader()?.takeIf { counts[it.ordinal] >= raiseK }
        }
        return stable
    }

    fun agreement(surface: SurfaceClass) = counts[surface.ordinal]

    fun reset() {
        history.fill(null)
        counts.fill(0)
        next = 0
        filled = 0
        stable = null
    }

    private fun leader(): SurfaceClass? {
        var best = -1
        for (i in counts.indices) if (counts[i] > 0 && (best < 0 || counts[i] > counts[best])) best = i
        return if (best < 0) null else SurfaceClass.entries[best]
    }
}
