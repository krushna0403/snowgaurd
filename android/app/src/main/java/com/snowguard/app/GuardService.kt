package com.snowguard.app

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.snowguard.app.alert.AlertPlayer
import com.snowguard.app.fusion.AlertGate
import com.snowguard.app.fusion.Assessment
import com.snowguard.app.fusion.FusionEngine
import com.snowguard.app.fusion.Severity
import com.snowguard.app.log.TripLogger
import com.snowguard.app.terrain.DemoDrive
import com.snowguard.app.terrain.FreezeRisk
import com.snowguard.app.terrain.LocationTracker
import com.snowguard.app.terrain.RouteData
import com.snowguard.app.terrain.RouteFix
import com.snowguard.app.terrain.RouteRepository
import com.snowguard.app.terrain.TerrainEngine
import com.snowguard.app.terrain.TerrainState
import com.snowguard.app.vision.CameraHealth
import com.snowguard.app.vision.SurfaceClass
import com.snowguard.app.vision.TemporalSmoother
import java.io.File

enum class GpsStatus { DENIED, DISABLED, WAITING, FIX, OFF_ROUTE, LOST }
enum class RouteStatus { LOADING, READY, MISSING }

data class GuardState(
    val gps: GpsStatus,
    val accuracyM: Int?,
    val route: RouteStatus,
    val terrain: TerrainState?,
    val visionAvailable: Boolean,
    val vision: SurfaceClass?,
    val assessment: Assessment,
    val lastAlert: Assessment?,
    val lastAlertText: String?,
    val cameraHealth: CameraHealth,
    val speedMps: Double?,
    /** Simulated clock of a running demo drive, null when driving for real. */
    val demoTimeMillis: Long? = null,
)

/**
 * Owns everything that must keep running with the screen locked (FR-U3): GPS,
 * the terrain engine, fusion, cooldown and alert output. The activity binds to
 * it and feeds camera predictions in; when the camera stops (screen off, camera
 * failure) vision goes stale and fusion continues terrain-only, and when GPS is
 * lost it continues vision-only (NFR reliability).
 */
class GuardService : Service() {

    inner class LocalBinder : Binder() {
        val service: GuardService get() = this@GuardService
    }

    fun interface Listener {
        fun onGuardState(state: GuardState)
    }

    private val binder = LocalBinder()
    private val main = Handler(Looper.getMainLooper())
    private lateinit var settings: AppSettings
    private lateinit var player: AlertPlayer
    private val smoother = TemporalSmoother()
    private val gate = AlertGate()
    private var tracker: LocationTracker? = null
    private var terrain: TerrainEngine? = null
    private var route: RouteData? = null
    private var demo: DemoDrive? = null

    private var routeStatus = RouteStatus.LOADING
    private var gpsStatus = GpsStatus.DENIED
    private var accuracyM: Int? = null
    private var terrainState: TerrainState? = null
    private var lastFixAt = 0L
    private var lastVisionAt = 0L
    private var assessment = Assessment.CLEAR
    private var lastAlert: Assessment? = null
    private var lastAlertText: String? = null
    private var foreground = false
    private var notifiedSeverity: Severity? = null
    private val listeners = LinkedHashSet<Listener>()
    private var cameraHealth = CameraHealth.OK
    private var lastFix: RouteFix? = null
    private lateinit var logger: TripLogger
    private var loggedVision: SurfaceClass? = null
    private var loggedRisk: FreezeRisk? = null
    private var loggedGps: GpsStatus? = null

    val state: GuardState
        get() = GuardState(
            gps = gpsStatus,
            accuracyM = accuracyM,
            route = routeStatus,
            terrain = terrainState,
            visionAvailable = visionUsable(),
            vision = if (visionUsable()) smoother.stable else null,
            assessment = assessment,
            lastAlert = lastAlert,
            lastAlertText = lastAlertText,
            cameraHealth = cameraHealth,
            speedMps = lastFix?.speedMps,
            demoTimeMillis = demo?.let { lastFix?.timeMillis },
        )

    val voiceLanguage: String get() = player.voiceLanguage

    val demoActive: Boolean get() = demo != null

    fun latestTripLog(): File? = logger.latest()

    override fun onCreate() {
        super.onCreate()
        settings = AppSettings(this)
        player = AlertPlayer(this, settings)
        logger = TripLogger(this).apply { start() }
        log(TripLogger.Event.START)
        createChannel()
        Thread {
            val route = RouteRepository.load(this)
            main.post {
                if (route == null) {
                    routeStatus = RouteStatus.MISSING
                } else {
                    Log.i(TAG, "Route '${route.name}': ${route.size} samples")
                    this.route = route
                    terrain = TerrainEngine(route)
                    routeStatus = RouteStatus.READY
                }
                reevaluate()
            }
        }.start()
        main.postDelayed(watchdog, WATCHDOG_MS)
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopGps()
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            foreground = false
            stopSelf()
            return START_NOT_STICKY
        }
        if (!hasLocationPermission()) {
            gpsStatus = GpsStatus.DENIED
            stopSelf()
            return START_NOT_STICKY
        }
        if (!foreground) {
            ServiceCompat.startForeground(
                this, NOTIFICATION_ID, buildNotification(),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION,
            )
            foreground = true
        }
        startGps()
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        main.removeCallbacksAndMessages(null)
        stopGps()
        log(TripLogger.Event.STOP)
        logger.close()
        player.shutdown()
        listeners.clear()
        super.onDestroy()
    }

    fun addListener(l: Listener) {
        listeners += l
        l.onGuardState(state)
    }

    fun removeListener(l: Listener) {
        listeners -= l
    }

    /**
     * Called from the camera analysis thread once per inference with a confident
     * class, or null for a low-confidence frame; [health] is the debounced camera state.
     */
    fun onVisionFrame(surface: SurfaceClass?, health: CameraHealth) {
        main.post {
            lastVisionAt = SystemClock.elapsedRealtime()
            if (health != cameraHealth) {
                cameraHealth = health
                if (health != CameraHealth.OK) smoother.reset()
                log(TripLogger.Event.CAMERA)
            }
            smoother.raiseK = settings.sensitivity.raiseK
            smoother.push(surface.takeIf { health == CameraHealth.OK })
            reevaluate()
        }
    }

    fun onSettingsChanged() {
        player.applyLanguage()
        smoother.raiseK = settings.sensitivity.raiseK
        reevaluate()
    }

    /** Plays [a] regardless of cooldown (settings "test alert"). */
    fun testAlert(a: Assessment) {
        val text = player.message(a)
        player.play(a, text)
        lastAlert = a
        lastAlertText = text
        publish()
    }

    /**
     * Starts or stops a synthetic winter-morning drive over the demo stretch,
     * replacing real GPS while it runs. Returns false if the route is not loaded.
     */
    fun toggleDemo(): Boolean {
        if (demo != null) {
            stopDemo()
            return true
        }
        val r = route ?: return false
        stopGps()
        terrain = TerrainEngine(r)
        gate.reset()
        demo = DemoDrive(r)
        main.post(demoTick)
        return true
    }

    private fun stopDemo() {
        demo = null
        main.removeCallbacks(demoTick)
        route?.let { terrain = TerrainEngine(it) }
        terrainState = null
        lastFix = null
        lastFixAt = 0L
        gpsStatus = GpsStatus.WAITING
        gate.reset()
        if (hasLocationPermission()) startGps() else gpsStatus = GpsStatus.DENIED
        reevaluate()
    }

    private val demoTick = object : Runnable {
        override fun run() {
            val d = demo ?: return
            val fix = d.next(DEMO_TICK_MS / 1000.0)
            if (fix == null) {
                stopDemo()
                return
            }
            onFix(fix, DEMO_ACCURACY_M)
            main.postDelayed(this, DEMO_TICK_MS)
        }
    }

    private fun startGps() {
        if (demo != null) return
        if (tracker == null) tracker = LocationTracker(this, ::onFix)
        val t = tracker ?: return
        if (!t.isGpsEnabled) {
            gpsStatus = GpsStatus.DISABLED
        } else {
            if (gpsStatus == GpsStatus.DENIED || gpsStatus == GpsStatus.DISABLED) gpsStatus = GpsStatus.WAITING
            t.start()
        }
        publish()
    }

    private fun stopGps() {
        tracker?.stop()
    }

    private fun onFix(fix: RouteFix, accuracy: Float) {
        lastFixAt = SystemClock.elapsedRealtime()
        lastFix = fix
        accuracyM = accuracy.toInt()
        val engine = terrain
        if (engine == null) {
            gpsStatus = GpsStatus.FIX
        } else {
            val s = engine.evaluate(fix)
            terrainState = s
            gpsStatus = if (s.onRoute) GpsStatus.FIX else GpsStatus.OFF_ROUTE
        }
        reevaluate()
    }

    private val watchdog = object : Runnable {
        override fun run() {
            val now = SystemClock.elapsedRealtime()
            if (lastVisionAt != 0L && now - lastVisionAt > VISION_STALE_MS) {
                smoother.reset()
                lastVisionAt = 0L
            }
            if (lastFixAt != 0L && now - lastFixAt > GPS_STALE_MS && gpsStatus != GpsStatus.LOST) {
                gpsStatus = GpsStatus.LOST
                terrainState = null
            }
            reevaluate()
            main.postDelayed(this, WATCHDOG_MS)
        }
    }

    private fun visionFresh() = lastVisionAt != 0L && SystemClock.elapsedRealtime() - lastVisionAt <= VISION_STALE_MS

    private fun visionUsable() = visionFresh() && cameraHealth == CameraHealth.OK

    private fun reevaluate() {
        val vision = if (visionUsable()) smoother.stable else null
        val t = terrainState?.takeIf { it.onRoute && gpsStatus == GpsStatus.FIX }
        val a = FusionEngine.withLookAhead(FusionEngine.assess(vision, t?.freezeRisk, settings.fusionMode), t?.lookAhead)
        assessment = a
        if (gate.shouldFire(a, SystemClock.elapsedRealtime())) {
            val text = player.message(a)
            player.play(a, text)
            lastAlert = a
            lastAlertText = text
            Log.i(TAG, "ALERT ${a.severity} ${a.hazard}: $text")
            log(TripLogger.Event.ALERT, text)
        }
        if (vision != loggedVision) {
            loggedVision = vision
            log(TripLogger.Event.VISION)
        }
        if (t?.freezeRisk != loggedRisk) {
            loggedRisk = t?.freezeRisk
            log(TripLogger.Event.TERRAIN)
        }
        if (gpsStatus != loggedGps) {
            loggedGps = gpsStatus
            log(TripLogger.Event.GPS)
        }
        if (foreground && a.severity != notifiedSeverity) {
            notifiedSeverity = a.severity
            getSystemService(NotificationManager::class.java)?.notify(NOTIFICATION_ID, buildNotification())
        }
        publish()
    }

    private fun log(event: TripLogger.Event, message: String? = null) {
        if (!::logger.isInitialized) return
        logger.log(
            event, lastFix, terrainState, if (visionUsable()) smoother.stable else null,
            cameraHealth, assessment, settings.fusionMode, message,
        )
    }

    private fun publish() {
        if (listeners.isEmpty()) return
        val s = state
        listeners.forEach { it.onGuardState(s) }
    }

    private fun hasLocationPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED

    private fun createChannel() {
        val channel = NotificationChannel(CHANNEL_ID, getString(R.string.notif_channel), NotificationManager.IMPORTANCE_LOW)
        channel.setShowBadge(false)
        getSystemService(NotificationManager::class.java)?.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, GuardService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE,
        )
        val text = when (assessment.severity) {
            Severity.NONE -> getString(R.string.notif_clear)
            else -> getString(severityLabel(assessment.severity)) + " · " + getString(hazardLabel(assessment.hazard))
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(getString(R.string.notif_title))
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(open)
            .addAction(0, getString(R.string.notif_stop), stop)
            .setCategory(NotificationCompat.CATEGORY_NAVIGATION)
            .build()
    }

    companion object {
        private const val TAG = "GuardService"
        const val ACTION_STOP = "com.snowguard.app.STOP"
        private const val CHANNEL_ID = "guard"
        private const val NOTIFICATION_ID = 1
        private const val WATCHDOG_MS = 1_000L
        const val VISION_STALE_MS = 2_000L
        const val GPS_STALE_MS = 15_000L
        private const val DEMO_TICK_MS = 1_000L
        private const val DEMO_ACCURACY_M = 5f

        fun severityLabel(s: Severity) = when (s) {
            Severity.NONE -> R.string.severity_none
            Severity.ADVISORY -> R.string.severity_advisory
            Severity.WARNING -> R.string.severity_warning
            Severity.CRITICAL -> R.string.severity_critical
        }

        fun hazardLabel(h: com.snowguard.app.fusion.Hazard?) = when (h) {
            com.snowguard.app.fusion.Hazard.ICE -> R.string.hazard_ice
            com.snowguard.app.fusion.Hazard.SNOW -> R.string.hazard_snow
            com.snowguard.app.fusion.Hazard.SLUSH -> R.string.hazard_slush
            com.snowguard.app.fusion.Hazard.WATER -> R.string.hazard_water
            com.snowguard.app.fusion.Hazard.BLACK_ICE_RISK -> R.string.hazard_black_ice
            com.snowguard.app.fusion.Hazard.FREEZE_AHEAD -> R.string.hazard_freeze_ahead
            com.snowguard.app.fusion.Hazard.CORNER_AHEAD -> R.string.hazard_corner_ahead
            com.snowguard.app.fusion.Hazard.FREEZE_ZONE, null -> R.string.hazard_freeze_zone
        }
    }
}
