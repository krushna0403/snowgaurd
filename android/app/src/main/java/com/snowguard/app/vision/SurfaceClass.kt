package com.snowguard.app.vision

import androidx.annotation.ColorRes
import androidx.annotation.StringRes
import com.snowguard.app.R

enum class FrictionBand(@ColorRes val color: Int) {
    GREEN(R.color.band_green),
    AMBER(R.color.band_amber),
    RED(R.color.band_red),
    UNKNOWN(R.color.band_unknown),
}

/**
 * The six SnowGuard road-surface classes. Declaration order is the model's
 * output order and must match `ml/snowguard_ml/config.py` CLASSES.
 */
enum class SurfaceClass(
    @StringRes val label: Int,
    @StringRes val action: Int,
    val friction: Float,
    val band: FrictionBand,
    val raisesAlert: Boolean,
) {
    DRY(R.string.surface_dry, R.string.action_dry, 0.75f, FrictionBand.GREEN, false),
    WET(R.string.surface_wet, R.string.action_wet, 0.50f, FrictionBand.AMBER, false),
    WATER(R.string.surface_water, R.string.action_water, 0.40f, FrictionBand.AMBER, true),
    FRESH_SNOW(R.string.surface_fresh_snow, R.string.action_fresh_snow, 0.20f, FrictionBand.RED, true),
    MELTED_SNOW(R.string.surface_melted_snow, R.string.action_melted_snow, 0.30f, FrictionBand.AMBER, true),
    ICE(R.string.surface_ice, R.string.action_ice, 0.10f, FrictionBand.RED, true);

    companion object {
        fun fromIndex(i: Int): SurfaceClass = entries[i]
    }
}
