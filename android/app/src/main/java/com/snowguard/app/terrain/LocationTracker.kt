package com.snowguard.app.terrain

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper

/**
 * Raw GNSS fixes from [LocationManager] (no Play Services, works offline and
 * with emulator GPX playback). Fills in bearing/speed from consecutive fixes
 * when the provider omits them, which mock and playback providers often do.
 */
class LocationTracker(
    context: Context,
    private val onFix: (RouteFix, Float) -> Unit,
) : LocationListener {

    private val manager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private var previous: Location? = null

    val isGpsEnabled get() = manager.isProviderEnabled(LocationManager.GPS_PROVIDER)

    @SuppressLint("MissingPermission")
    fun start() {
        manager.requestLocationUpdates(LocationManager.GPS_PROVIDER, INTERVAL_MS, 0f, this, Looper.getMainLooper())
        manager.getLastKnownLocation(LocationManager.GPS_PROVIDER)?.let { onLocationChanged(it) }
    }

    fun stop() {
        manager.removeUpdates(this)
        previous = null
    }

    override fun onLocationChanged(location: Location) {
        val prev = previous
        var bearing: Double? = if (location.hasBearing()) location.bearing.toDouble() else null
        var speed: Double? = if (location.hasSpeed()) location.speed.toDouble() else null
        if (prev != null && location.time > prev.time) {
            val moved = prev.distanceTo(location)
            if (moved >= MIN_MOVE_FOR_BEARING_M) {
                if (bearing == null) bearing = ((prev.bearingTo(location) + 360f) % 360f).toDouble()
                if (speed == null) speed = moved / ((location.time - prev.time) / 1000.0)
            }
        }
        if (prev == null || prev.distanceTo(location) >= MIN_MOVE_FOR_BEARING_M || location.time - prev.time > 10_000) {
            previous = location
        }
        onFix(RouteFix(location.latitude, location.longitude, location.time, bearing, speed), location.accuracy)
    }

    @Deprecated("Deprecated in API 29")
    override fun onStatusChanged(provider: String?, status: Int, extras: android.os.Bundle?) = Unit
    override fun onProviderEnabled(provider: String) = Unit
    override fun onProviderDisabled(provider: String) = Unit

    private companion object {
        const val INTERVAL_MS = 1_000L
        const val MIN_MOVE_FOR_BEARING_M = 5f
    }
}
