package com.snowguard.app

import android.content.Context
import com.snowguard.app.fusion.FusionMode

enum class Sensitivity(val raiseK: Int) { LOW(5), NORMAL(4), HIGH(3) }

/** On-device model choice; latency of the two differs a lot between phones. */
enum class ModelVariant(val asset: String, val quantization: String?) {
    /** INT8 weights, float activations; quantization label comes from labels.json. */
    INT8W("snowguard_mnv2.tflite", null),
    FP16("snowguard_mnv2_fp16.tflite", "fp16"),
}

/** FR-U5: persisted user settings (language, alert volume, sensitivity). */
class AppSettings(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("snowguard", Context.MODE_PRIVATE)

    var language: String
        get() = prefs.getString(KEY_LANGUAGE, "en") ?: "en"
        set(v) = prefs.edit().putString(KEY_LANGUAGE, v).apply()

    var volume: Float
        get() = prefs.getFloat(KEY_VOLUME, 1f)
        set(v) = prefs.edit().putFloat(KEY_VOLUME, v.coerceIn(0f, 1f)).apply()

    var sensitivity: Sensitivity
        get() = enumOr(prefs.getString(KEY_SENSITIVITY, null), Sensitivity.NORMAL)
        set(v) = prefs.edit().putString(KEY_SENSITIVITY, v.name).apply()

    var fusionMode: FusionMode
        get() = enumOr(prefs.getString(KEY_FUSION, null), FusionMode.TIERED)
        set(v) = prefs.edit().putString(KEY_FUSION, v.name).apply()

    var voiceEnabled: Boolean
        get() = prefs.getBoolean(KEY_VOICE, true)
        set(v) = prefs.edit().putBoolean(KEY_VOICE, v).apply()

    var vibrationEnabled: Boolean
        get() = prefs.getBoolean(KEY_VIBRATION, true)
        set(v) = prefs.edit().putBoolean(KEY_VIBRATION, v).apply()

    var modelVariant: ModelVariant
        get() = enumOr(prefs.getString(KEY_MODEL, null), ModelVariant.INT8W)
        set(v) = prefs.edit().putString(KEY_MODEL, v.name).apply()

    private inline fun <reified T : Enum<T>> enumOr(name: String?, default: T): T =
        name?.let { n -> enumValues<T>().firstOrNull { it.name == n } } ?: default

    private companion object {
        const val KEY_LANGUAGE = "language"
        const val KEY_VOLUME = "volume"
        const val KEY_SENSITIVITY = "sensitivity"
        const val KEY_FUSION = "fusion_mode"
        const val KEY_VOICE = "voice"
        const val KEY_VIBRATION = "vibration"
        const val KEY_MODEL = "model_variant"
    }
}
