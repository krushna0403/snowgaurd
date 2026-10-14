package com.snowguard.app.terrain

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Loads the bundled route database (built by `terrain/snowguard_terrain/build_db.py`).
 * SQLite cannot open files inside the APK, so the asset is copied to private
 * storage once per app version.
 */
object RouteRepository {
    const val ASSET = "route_manali_leh.db"
    private const val TAG = "RouteRepository"

    fun load(context: Context): RouteData? = try {
        val db = SQLiteDatabase.openDatabase(copyAsset(context).path, null, SQLiteDatabase.OPEN_READONLY)
        db.use { read(it) }
    } catch (e: Exception) {
        Log.e(TAG, "Route database load failed", e)
        null
    }

    private fun copyAsset(context: Context): File {
        val version = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        val target = File(context.noBackupFilesDir, "$version-$ASSET")
        if (!target.exists()) {
            context.noBackupFilesDir.listFiles { f -> f.name.endsWith(ASSET) }?.forEach { it.delete() }
            context.assets.open(ASSET).use { input -> target.outputStream().use { input.copyTo(it) } }
        }
        return target
    }

    private fun read(db: SQLiteDatabase): RouteData {
        val meta = HashMap<String, String>()
        db.rawQuery("SELECT key, value FROM meta", null).use { c ->
            while (c.moveToNext()) meta[c.getString(0)] = c.getString(1)
        }
        val nAz = meta.getValue("n_azimuth").toInt()
        val scale = meta.getValue("horizon_scale").toFloat()

        val n = db.rawQuery("SELECT COUNT(*) FROM samples", null).use { it.moveToFirst(); it.getInt(0) }
        val lat = DoubleArray(n); val lon = DoubleArray(n); val dist = DoubleArray(n); val elev = DoubleArray(n)
        val heading = DoubleArray(n); val curv = DoubleArray(n); val slope = DoubleArray(n); val aspect = DoubleArray(n)
        val tunnel = BooleanArray(n)
        val horizon = FloatArray(n * nAz)
        db.rawQuery(
            "SELECT lat, lon, dist_m, elev_m, heading_deg, curvature_deg_per_100m, slope_deg, aspect_deg, " +
                "tunnel, horizon FROM samples ORDER BY id", null
        ).use { c ->
            var i = 0
            while (c.moveToNext()) {
                lat[i] = c.getDouble(0); lon[i] = c.getDouble(1); dist[i] = c.getDouble(2); elev[i] = c.getDouble(3)
                heading[i] = c.getDouble(4); curv[i] = c.getDouble(5); slope[i] = c.getDouble(6); aspect[i] = c.getDouble(7)
                tunnel[i] = c.getInt(8) != 0
                val shorts = ByteBuffer.wrap(c.getBlob(9)).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                for (k in 0 until nAz) horizon[i * nAz + k] = shorts.get(k) / scale
                i++
            }
        }

        val landmarks = ArrayList<Landmark>()
        db.rawQuery("SELECT name, kind, sample_id, elev_m FROM landmarks ORDER BY sample_id", null).use { c ->
            while (c.moveToNext()) landmarks += Landmark(c.getString(0), c.getString(1), c.getInt(2), c.getDouble(3))
        }

        return RouteData(
            lat, lon, dist, elev, heading, curv, slope, aspect, tunnel, horizon, nAz,
            parseThresholds(meta["freeze_thresholds"]), landmarks,
            meta["route"]?.trim('"') ?: "route",
        )
    }

    private fun parseThresholds(json: String?): List<FreezeThreshold> {
        if (json == null) return RouteData.DEFAULT_THRESHOLDS
        val arr = JSONArray(json)
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            val months = o.getJSONArray("months")
            FreezeThreshold(
                o.getString("season"),
                (0 until months.length()).map { months.getInt(it) }.toSet(),
                o.getDouble("min_elev_m"),
            )
        }
    }
}
