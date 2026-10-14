package com.snowguard.app.alert

import android.content.Context
import android.content.res.Configuration
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.snowguard.app.AppSettings
import com.snowguard.app.R
import com.snowguard.app.fusion.Assessment
import com.snowguard.app.fusion.Hazard
import com.snowguard.app.fusion.Severity
import java.util.Locale
import kotlin.math.PI
import kotlin.math.min
import kotlin.math.sin

/**
 * Audio, voice and haptic alert output (FR-F3..F5). Tones are synthesised so
 * the pitch can follow the hazard (2 kHz ice, 1.5 kHz other) and the beep
 * pattern the severity. Voice uses the on-device TextToSpeech engine, so it
 * works offline once the English / Hindi voice data is installed.
 */
class AlertPlayer(context: Context, private val settings: AppSettings) {

    private val appContext = context.applicationContext
    private val thread = HandlerThread("snowguard-alerts").apply { start() }
    private val handler = Handler(thread.looper)
    private val audioManager = appContext.getSystemService(AudioManager::class.java)
    private val vibrator: Vibrator? = if (android.os.Build.VERSION.SDK_INT >= 31) {
        appContext.getSystemService(VibratorManager::class.java)?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        appContext.getSystemService(Vibrator::class.java)
    }
    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
        .build()
    private val focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
        .setAudioAttributes(attributes)
        .build()

    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var ttsLocale: Locale? = null

    /** Language actually used for voice; may fall back to English if Hindi voice data is missing. */
    var voiceLanguage: String = "en"
        private set

    init {
        tts = TextToSpeech(appContext) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                tts?.setAudioAttributes(attributes)
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) = Unit
                    override fun onDone(utteranceId: String?) = releaseFocus()
                    @Deprecated("Deprecated in API 21")
                    override fun onError(utteranceId: String?) = releaseFocus()
                })
                applyLanguage()
            } else {
                Log.w(TAG, "TextToSpeech unavailable; tones and vibration only")
            }
        }
    }

    fun applyLanguage() {
        val engine = tts ?: return
        if (!ttsReady) return
        val wanted = if (settings.language == "hi") Locale("hi", "IN") else Locale.ENGLISH
        val result = engine.setLanguage(wanted)
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            Log.w(TAG, "No offline voice for $wanted, falling back to English")
            engine.setLanguage(Locale.ENGLISH)
            ttsLocale = Locale.ENGLISH
        } else {
            ttsLocale = wanted
        }
        voiceLanguage = ttsLocale?.language ?: "en"
    }

    fun play(a: Assessment, spoken: String? = null) {
        val hazard = a.hazard ?: return
        val text = spoken ?: message(a)
        handler.post {
            requestFocus()
            val toneMs = playTone(if (hazard.isIce) ICE_HZ else OTHER_HZ, PATTERNS.getValue(a.severity))
            if (settings.vibrationEnabled) vibrate(a.severity)
            if (settings.voiceEnabled && ttsReady) {
                handler.postDelayed({ speak(text) }, toneMs + 150L)
            } else {
                handler.postDelayed({ releaseFocus() }, toneMs + 100L)
            }
        }
    }

    /** Localised voice prompt for [a] in the voice language actually available. */
    fun message(a: Assessment): String {
        val res = localizedContext().resources
        val distance = a.distanceM ?: 0
        val body = when (a.hazard) {
            Hazard.ICE -> res.getString(R.string.voice_ice)
            Hazard.SNOW -> res.getString(R.string.voice_snow)
            Hazard.SLUSH -> res.getString(R.string.voice_slush)
            Hazard.WATER -> res.getString(R.string.voice_water)
            Hazard.BLACK_ICE_RISK -> res.getString(R.string.voice_black_ice)
            Hazard.FREEZE_AHEAD -> res.getString(R.string.voice_freeze_ahead, distance)
            Hazard.CORNER_AHEAD -> res.getString(R.string.voice_corner_ahead, distance)
            Hazard.FREEZE_ZONE, null -> res.getString(R.string.voice_freeze_zone)
        }
        return if (a.severity == Severity.CRITICAL && a.hazard != Hazard.BLACK_ICE_RISK) {
            res.getString(R.string.voice_critical_prefix) + " " + body
        } else body
    }

    fun localizedContext(): Context {
        val lang = if (ttsReady) voiceLanguage else settings.language
        val config = Configuration(appContext.resources.configuration).apply { setLocale(Locale(lang)) }
        return appContext.createConfigurationContext(config)
    }

    fun shutdown() {
        handler.removeCallbacksAndMessages(null)
        tts?.shutdown()
        tts = null
        thread.quitSafely()
        releaseFocus()
    }

    private fun speak(text: String) {
        val params = Bundle().apply { putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, settings.volume) }
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, "alert-${System.nanoTime()}")
    }

    /** Plays [pattern] of (on, off) millisecond pairs at [hz]; returns its duration. */
    private fun playTone(hz: Int, pattern: List<Pair<Int, Int>>): Long {
        val totalMs = pattern.sumOf { it.first + it.second }
        val pcm = ShortArray(SAMPLE_RATE * totalMs / 1000)
        var offset = 0
        for ((on, off) in pattern) {
            val n = SAMPLE_RATE * on / 1000
            val fade = min(n / 2, SAMPLE_RATE * FADE_MS / 1000)
            for (i in 0 until n) {
                val env = when {
                    i < fade -> i.toFloat() / fade
                    i >= n - fade -> (n - i).toFloat() / fade
                    else -> 1f
                }
                pcm[offset + i] = (sin(2 * PI * hz * i / SAMPLE_RATE) * env * AMPLITUDE).toInt().toShort()
            }
            offset += n + SAMPLE_RATE * off / 1000
        }
        return try {
            val track = AudioTrack.Builder()
                .setAudioAttributes(attributes)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setTransferMode(AudioTrack.MODE_STATIC)
                .setBufferSizeInBytes(pcm.size * 2)
                .build()
            track.write(pcm, 0, pcm.size)
            track.setVolume(settings.volume)
            track.play()
            handler.postDelayed({ track.release() }, totalMs + 200L)
            totalMs.toLong()
        } catch (e: Exception) {
            Log.e(TAG, "Tone playback failed", e)
            0L
        }
    }

    private fun vibrate(severity: Severity) {
        val v = vibrator ?: return
        if (!v.hasVibrator()) return
        val timings = VIBRATION.getValue(severity)
        v.vibrate(VibrationEffect.createWaveform(timings, -1))
    }

    private fun requestFocus() {
        audioManager?.requestAudioFocus(focusRequest)
    }

    private fun releaseFocus() {
        audioManager?.abandonAudioFocusRequest(focusRequest)
    }

    companion object {
        private const val TAG = "AlertPlayer"
        const val ICE_HZ = 2000
        const val OTHER_HZ = 1500
        private const val SAMPLE_RATE = 22_050
        private const val FADE_MS = 8
        private const val AMPLITUDE = 0.8 * Short.MAX_VALUE

        /** (beep ms, gap ms): one long beep, two beeps, then two bursts of three fast beeps. */
        val PATTERNS = mapOf(
            Severity.NONE to emptyList(),
            Severity.ADVISORY to listOf(350 to 0),
            Severity.WARNING to listOf(180 to 120, 180 to 0),
            Severity.CRITICAL to listOf(110 to 70, 110 to 70, 110 to 250, 110 to 70, 110 to 70, 110 to 0),
        )

        private val VIBRATION = mapOf(
            Severity.NONE to longArrayOf(0),
            Severity.ADVISORY to longArrayOf(0, 250),
            Severity.WARNING to longArrayOf(0, 200, 120, 200),
            Severity.CRITICAL to longArrayOf(0, 400, 150, 400, 150, 400),
        )
    }
}
