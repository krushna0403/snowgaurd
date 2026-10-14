package com.snowguard.app.log

import android.content.Context
import android.util.Log
import com.snowguard.app.fusion.Assessment
import com.snowguard.app.fusion.FusionMode
import com.snowguard.app.terrain.RouteFix
import com.snowguard.app.terrain.TerrainState
import com.snowguard.app.vision.CameraHealth
import com.snowguard.app.vision.SurfaceClass
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * FR-D1/D2: on-device trip log of geotagged detections and alerts, one CSV per
 * trip in the app's files dir. No images are stored. The file is shared only
 * when the user explicitly exports it.
 */
class TripLogger(context: Context) {

    val dir: File = File(context.getExternalFilesDir(null) ?: context.filesDir, DIR_NAME).apply { mkdirs() }
    private var writer: BufferedWriter? = null
    private val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX", Locale.US)
    private val fileStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("Asia/Kolkata")
    }

    var file: File? = null
        private set

    fun start() {
        if (writer != null) return
        val f = File(dir, "trip_${fileStamp.format(Date())}.csv")
        try {
            writer = f.bufferedWriter().also {
                it.write(HEADER)
                it.newLine()
                it.flush()
            }
            file = f
        } catch (e: Exception) {
            Log.e(TAG, "Cannot open trip log", e)
        }
    }

    fun log(
        event: Event,
        fix: RouteFix?,
        terrain: TerrainState?,
        vision: SurfaceClass?,
        camera: CameraHealth,
        assessment: Assessment,
        mode: FusionMode,
        message: String? = null,
    ) {
        val w = writer ?: return
        val row = listOf(
            iso.format(Date()),
            event.name,
            fix?.lat?.let { "%.6f".format(Locale.US, it) },
            fix?.lon?.let { "%.6f".format(Locale.US, it) },
            fix?.speedMps?.let { "%.1f".format(Locale.US, it) },
            terrain?.snap?.distAlongM?.let { "%.2f".format(Locale.US, it / 1000) },
            terrain?.elevationM?.let { "%.0f".format(Locale.US, it) },
            terrain?.freezeRisk?.name,
            terrain?.sunlit?.toString(),
            terrain?.horizonDeg?.let { "%.1f".format(Locale.US, it) },
            terrain?.sun?.elevationDeg?.let { "%.1f".format(Locale.US, it) },
            terrain?.lookAhead?.distanceM?.let { "%.0f".format(Locale.US, it) },
            vision?.name,
            camera.name,
            assessment.severity.name,
            assessment.hazard?.name,
            mode.name,
            message,
        ).joinToString(",") { csv(it) }
        try {
            w.write(row)
            w.newLine()
            w.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Trip log write failed", e)
        }
    }

    fun close() {
        try {
            writer?.close()
        } catch (_: Exception) {
        }
        writer = null
    }

    fun latest(): File? = file?.takeIf { it.length() > HEADER.length + 1 }
        ?: dir.listFiles { f -> f.name.endsWith(".csv") && f.length() > HEADER.length + 1 }?.maxByOrNull { it.lastModified() }

    enum class Event { START, ALERT, VISION, TERRAIN, CAMERA, GPS, STOP }

    private fun csv(v: String?): String {
        if (v == null) return ""
        return if (v.any { it == ',' || it == '"' || it == '\n' }) "\"" + v.replace("\"", "\"\"") + "\"" else v
    }

    companion object {
        private const val TAG = "TripLogger"
        const val DIR_NAME = "trips"
        const val HEADER = "time,event,lat,lon,speed_mps,route_km,elev_m,terrain_risk,sunlit,horizon_deg," +
            "sun_elev_deg,lookahead_m,vision_class,camera,severity,hazard,fusion_mode,message"
    }
}
