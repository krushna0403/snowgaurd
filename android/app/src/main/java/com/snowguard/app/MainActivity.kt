package com.snowguard.app

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.ColorRes
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.snowguard.app.camera.FrameAnalyzer
import com.snowguard.app.camera.FrameResult
import com.snowguard.app.databinding.ActivityMainBinding
import com.snowguard.app.databinding.DialogSettingsBinding
import com.snowguard.app.fusion.Assessment
import com.snowguard.app.fusion.FusionMode
import com.snowguard.app.fusion.Hazard
import com.snowguard.app.fusion.Severity
import com.snowguard.app.terrain.FreezeRisk
import com.snowguard.app.terrain.TerrainState
import com.snowguard.app.vision.CameraHealth
import com.snowguard.app.vision.FrictionBand
import com.snowguard.app.vision.Prediction
import com.snowguard.app.vision.SurfaceClassifier
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var settings: AppSettings
    private var classifier: SurfaceClassifier? = null
    private var shownSurface: Prediction? = null
    private var frameAnalyzer: FrameAnalyzer? = null
    private val demoClock = SimpleDateFormat("d MMM HH:mm", Locale.ENGLISH).apply {
        timeZone = TimeZone.getTimeZone("Asia/Kolkata")
    }

    @Volatile
    private var guard: GuardService? = null
    private val guardListener = GuardService.Listener { runOnUiThread { showGuardState(it) } }
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val service = (binder as GuardService.LocalBinder).service
            guard = service
            service.addListener(guardListener)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            guard = null
        }
    }

    private val permissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            if (result[Manifest.permission.CAMERA] == true) startCamera()
            else showCameraStatus(Status.ERROR, R.string.status_camera_denied)
            if (result[Manifest.permission.ACCESS_FINE_LOCATION] == true) startGuardService()
            else setStatus(binding.chipGps, Status.ERROR, getString(R.string.status_gps_denied))
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        applySystemBarInsets()
        settings = AppSettings(this)
        analysisExecutor = Executors.newSingleThreadExecutor()

        loadModel()
        setStatus(binding.chipGps, Status.PENDING, getString(R.string.status_gps_none))
        showSurfacePlaceholder(if (classifier == null) R.string.action_no_model else R.string.action_waiting)
        showTerrainMessage(R.string.terrain_waiting)
        binding.chipSettings.setOnClickListener { showSettings() }

        val wanted = buildList {
            add(Manifest.permission.CAMERA)
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            add(Manifest.permission.ACCESS_COARSE_LOCATION)
            if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (isGranted(Manifest.permission.CAMERA)) startCamera()
        if (isGranted(Manifest.permission.ACCESS_FINE_LOCATION)) startGuardService()
        val missing = wanted.filterNot(::isGranted)
        if (missing.isNotEmpty()) permissions.launch(missing.toTypedArray())
    }

    override fun onStart() {
        super.onStart()
        bindService(Intent(this, GuardService::class.java), connection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        guard?.removeListener(guardListener)
        guard = null
        unbindService(connection)
        super.onStop()
    }

    override fun onDestroy() {
        super.onDestroy()
        val model = classifier
        analysisExecutor.execute { model?.close() }
        analysisExecutor.shutdown()
    }

    private fun loadModel() {
        val variant = settings.modelVariant
        val model = SurfaceClassifier.create(this, variant.asset, variant.quantization)
            ?: SurfaceClassifier.create(this).takeIf { variant != ModelVariant.INT8W }
        classifier = model
        if (model != null) {
            setStatus(binding.chipModel, Status.OK, getString(
                when (model.quantization) {
                    "int8" -> R.string.status_model_int8
                    "dynamic_int8" -> R.string.status_model_int8w
                    "fp16" -> R.string.status_model_fp16
                    else -> R.string.status_model_fp32
                }
            ))
        } else {
            setStatus(binding.chipModel, Status.ERROR, getString(R.string.status_model_none))
        }
    }

    /** Swaps the classifier and rebinds the camera so the new analyzer uses it. */
    private fun switchModel() {
        val old = classifier
        loadModel()
        shownSurface = null
        if (isGranted(Manifest.permission.CAMERA)) startCamera()
        analysisExecutor.execute { old?.close() }
    }

    private fun startGuardService() {
        ContextCompat.startForegroundService(this, Intent(this, GuardService::class.java))
    }

    /** Android 15+ forces edge-to-edge; keep the overlays clear of the status and gesture bars. */
    private fun applySystemBarInsets() {
        val top = binding.statusBar
        val bottom = binding.surfacePanel
        val topPad = top.paddingTop
        val bottomPad = bottom.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            top.updatePadding(left = bars.left + topPad, top = bars.top + topPad, right = bars.right + topPad)
            bottom.updatePadding(bottom = bars.bottom + bottomPad)
            insets
        }
    }

    private fun isGranted(permission: String) =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        showCameraStatus(Status.PENDING, R.string.status_camera_starting)
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                // Preview and analysis share 4:3 so the on-screen ROI box matches what the model sees.
                val aspect = AspectRatioStrategy(AspectRatio.RATIO_4_3, AspectRatioStrategy.FALLBACK_RULE_AUTO)
                val preview = Preview.Builder()
                    .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(aspect).build())
                    .build()
                    .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }
                // Inference downsamples to 224x224, so a small analysis stream saves battery.
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(aspect)
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    android.util.Size(640, 480),
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                )
                            )
                            .build()
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also {
                        val analyzer = FrameAnalyzer(classifier, ::onFrameResult)
                        frameAnalyzer = analyzer
                        it.setAnalyzer(analysisExecutor, analyzer)
                    }

                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                showCameraStatus(Status.OK, R.string.status_camera_ok)
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                showCameraStatus(Status.ERROR, R.string.status_camera_error)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Runs on the analysis thread. */
    private fun onFrameResult(result: FrameResult) {
        val p = result.prediction
        if (result.inferred && p != null) {
            guard?.onVisionFrame(p.surface.takeIf { p.confidence >= MIN_CONFIDENCE }, result.health)
        }
        runOnUiThread {
            if (result.health != CameraHealth.OK) {
                showCameraUnreliable(result.health)
                shownSurface = null
                return@runOnUiThread
            }
            val s = result.stats
            binding.roiOverlay.setFrameSize(s.uprightWidth, s.uprightHeight)
            binding.debugText.text = getString(
                R.string.debug_frame,
                s.fps,
                s.inferenceFps,
                p?.inferenceMs ?: 0f,
                s.uprightWidth,
                s.uprightHeight,
                s.meanLuma,
            )
            if (p != null && p !== shownSurface) {
                shownSurface = p
                showPrediction(p)
            }
        }
    }

    private fun showPrediction(p: Prediction) {
        val surface = p.surface
        val confident = p.confidence >= MIN_CONFIDENCE
        binding.surfaceLabel.text = getString(surface.label)
        binding.surfaceLabel.alpha = if (confident) 1f else 0.5f
        binding.confidenceText.text = getString(R.string.confidence_value, (p.confidence * 100).roundToInt())
        binding.frictionBand.setBackgroundColor(
            ContextCompat.getColor(this, if (confident) surface.band.color else FrictionBand.UNKNOWN.color)
        )
        binding.actionText.text = getString(if (confident) surface.action else R.string.action_low_confidence)
    }

    private fun cameraIssueText(h: CameraHealth) = getString(
        when (h) {
            CameraHealth.DARK -> R.string.camera_dark
            CameraHealth.GLARE -> R.string.camera_glare
            else -> R.string.camera_blurred
        }
    )

    private fun showCameraUnreliable(h: CameraHealth) {
        binding.surfaceLabel.text = getString(R.string.surface_placeholder)
        binding.confidenceText.text = getString(R.string.confidence_placeholder)
        binding.frictionBand.setBackgroundColor(ContextCompat.getColor(this, FrictionBand.UNKNOWN.color))
        binding.actionText.text = cameraIssueText(h)
    }

    private fun showSurfacePlaceholder(action: Int) {
        binding.surfaceLabel.text = getString(R.string.surface_placeholder)
        binding.confidenceText.text = getString(R.string.confidence_placeholder)
        binding.frictionBand.setBackgroundColor(ContextCompat.getColor(this, FrictionBand.UNKNOWN.color))
        binding.actionText.text = getString(action)
    }

    // ----------------------------------------------------------------------------- guard state

    private fun showGuardState(s: GuardState) {
        showGpsChip(s)
        showAlertBanner(s.assessment)
        adaptFrameRate(s.speedMps)
        val terrainUsable = s.gps == GpsStatus.FIX && s.terrain?.onRoute == true
        binding.modeText.visibility = View.VISIBLE
        binding.modeText.text = when {
            s.cameraHealth != CameraHealth.OK ->
                getString(R.string.mode_camera_unreliable, cameraIssueText(s.cameraHealth))
            !s.visionAvailable && terrainUsable -> getString(R.string.mode_terrain_only)
            s.visionAvailable && !terrainUsable && s.route == RouteStatus.READY -> getString(R.string.mode_vision_only)
            else -> {
                binding.modeText.visibility = View.GONE
                ""
            }
        }
        when {
            s.route == RouteStatus.MISSING -> showTerrainMessage(R.string.terrain_no_route)
            s.gps == GpsStatus.OFF_ROUTE -> showTerrainMessage(R.string.terrain_off_route)
            s.terrain != null && s.gps == GpsStatus.FIX -> showTerrain(s.terrain)
            else -> showTerrainMessage(R.string.terrain_waiting)
        }
    }

    private fun showGpsChip(s: GuardState) {
        val chip = binding.chipGps
        when {
            s.demoTimeMillis != null -> setStatus(chip, Status.PENDING, getString(R.string.status_gps_demo, demoClock.format(s.demoTimeMillis)))
            s.route == RouteStatus.MISSING -> setStatus(chip, Status.ERROR, getString(R.string.status_no_route))
            s.gps == GpsStatus.DENIED -> setStatus(chip, Status.ERROR, getString(R.string.status_gps_denied))
            s.gps == GpsStatus.DISABLED -> setStatus(chip, Status.ERROR, getString(R.string.status_gps_none))
            s.gps == GpsStatus.LOST -> setStatus(chip, Status.ERROR, getString(R.string.status_gps_lost))
            s.gps == GpsStatus.WAITING -> setStatus(chip, Status.PENDING, getString(R.string.status_gps_waiting))
            s.gps == GpsStatus.OFF_ROUTE -> setStatus(chip, Status.PENDING, getString(R.string.status_gps_off_route))
            else -> setStatus(chip, Status.OK, getString(R.string.status_gps_fix, s.accuracyM ?: 0))
        }
    }

    private fun showAlertBanner(a: Assessment) {
        val banner = binding.alertBanner
        if (a.severity == Severity.NONE) {
            banner.visibility = View.GONE
            return
        }
        banner.visibility = View.VISIBLE
        val severity = getString(GuardService.severityLabel(a.severity))
        val hazard = getString(GuardService.hazardLabel(a.hazard))
        banner.text = a.distanceM?.let { getString(R.string.alert_banner_distance, severity, hazard, it) }
            ?: getString(R.string.alert_banner, severity, hazard)
        val color = when (a.severity) {
            Severity.CRITICAL -> R.color.band_red
            Severity.WARNING -> R.color.band_amber
            else -> R.color.severity_advisory
        }
        banner.background.mutate().setTint(ContextCompat.getColor(this, color))
        banner.setTextColor(ContextCompat.getColor(this, if (a.severity == Severity.WARNING) R.color.black else R.color.white))
    }

    private fun showTerrain(s: TerrainState) {
        val snap = s.snap ?: return showTerrainMessage(R.string.terrain_off_route)
        val (riskText, riskColor) = when (s.freezeRisk) {
            FreezeRisk.NONE -> R.string.terrain_risk_none to R.color.band_green
            FreezeRisk.ELEVATED -> R.string.terrain_risk_elevated to R.color.band_amber
            FreezeRisk.HIGH -> R.string.terrain_risk_high to R.color.band_red
        }
        binding.terrainRisk.text = getString(riskText)
        binding.terrainRisk.setTextColor(
            ContextCompat.getColor(this, if (s.freezeRisk == FreezeRisk.NONE) R.color.white else riskColor)
        )
        binding.terrainPanel.backgroundTintList =
            ContextCompat.getColorStateList(this, if (s.freezeRisk == FreezeRisk.HIGH) R.color.risk_high_bg else R.color.scrim)

        val light = when {
            s.inTunnel -> getString(R.string.terrain_tunnel)
            s.sun.elevationDeg <= 0 -> getString(R.string.terrain_night)
            s.sunlit == true -> getString(R.string.terrain_sunlit)
            else -> getString(R.string.terrain_shaded, formatShade(s.minutesInShade ?: 0))
        }
        val parts = mutableListOf(light)
        if (s.northFacing) parts += getString(R.string.terrain_north)
        s.threshold?.let { parts += getString(R.string.terrain_threshold, it.season, it.minElevationM.roundToInt()) }
        binding.terrainDetail.text = parts.joinToString("  ·  ")

        val ahead = s.lookAhead
        binding.lookAheadText.visibility = if (ahead != null && s.freezeRisk == FreezeRisk.NONE) View.VISIBLE else View.GONE
        if (ahead != null) {
            binding.lookAheadText.text = getString(
                if (ahead.blindCorner) R.string.lookahead_corner else R.string.lookahead_zone,
                ahead.distanceM.roundToInt(),
            )
        }

        val landmark = s.nearestLandmark?.let { m ->
            val d = s.landmarkDistanceM ?: 0.0
            when {
                abs(d) < 500 -> getString(R.string.terrain_landmark_here, m.name)
                (d > 0) == (snap.direction >= 0) -> getString(R.string.terrain_landmark_ahead, m.name, abs(d) / 1000)
                else -> getString(R.string.terrain_landmark_behind, m.name, abs(d) / 1000)
            }
        }
        binding.terrainDebug.text = listOfNotNull(
            getString(R.string.terrain_km, snap.distAlongM / 1000),
            getString(R.string.terrain_elev, (s.elevationM ?: 0.0).roundToInt()),
            getString(R.string.terrain_sun, s.sun.elevationDeg, s.sun.azimuthDeg, s.horizonDeg ?: 0.0),
            landmark,
        ).joinToString("  ")
    }

    private fun formatShade(minutes: Int) = when {
        minutes >= 720 -> "12 h+"
        minutes >= 60 -> "${minutes / 60} h ${minutes % 60} min"
        else -> "$minutes min"
    }

    private fun showTerrainMessage(msg: Int) {
        binding.terrainRisk.text = getString(msg)
        binding.terrainRisk.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
        binding.terrainPanel.backgroundTintList = ContextCompat.getColorStateList(this, R.color.scrim)
        binding.terrainDetail.text = ""
        binding.terrainDebug.text = ""
        binding.lookAheadText.visibility = View.GONE
    }

    /** FR-V7: fewer inferences when parked or when the phone is getting hot. */
    private fun adaptFrameRate(speedMps: Double?) {
        val analyzer = frameAnalyzer ?: return
        val thermal = getSystemService(PowerManager::class.java)?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
        analyzer.minInferenceIntervalMs = when {
            thermal >= PowerManager.THERMAL_STATUS_SEVERE -> 400L
            thermal >= PowerManager.THERMAL_STATUS_MODERATE -> 150L
            speedMps != null && speedMps < STOPPED_SPEED_MPS -> 250L
            else -> FrameAnalyzer.DEFAULT_INTERVAL_MS
        }
    }

    private fun shareTripLog() {
        val file = guard?.latestTripLog()
        if (file == null) {
            Toast.makeText(this, R.string.share_log_none, Toast.LENGTH_SHORT).show()
            return
        }
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        val send = Intent(Intent.ACTION_SEND)
            .setType("text/csv")
            .putExtra(Intent.EXTRA_STREAM, uri)
            .putExtra(Intent.EXTRA_SUBJECT, file.name)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, getString(R.string.share_log_title)))
    }

    // ----------------------------------------------------------------------------- settings

    private fun showSettings() {
        val d = DialogSettingsBinding.inflate(layoutInflater)
        d.language.check(if (settings.language == "hi") R.id.langHi else R.id.langEn)
        d.volume.progress = (settings.volume * 10).roundToInt()
        d.sensitivity.check(
            when (settings.sensitivity) {
                Sensitivity.LOW -> R.id.sensLow
                Sensitivity.NORMAL -> R.id.sensNormal
                Sensitivity.HIGH -> R.id.sensHigh
            }
        )
        d.fusion.check(if (settings.fusionMode == FusionMode.OR_GATE) R.id.fusionOr else R.id.fusionTiered)
        d.voice.isChecked = settings.voiceEnabled
        d.vibration.isChecked = settings.vibrationEnabled
        d.model.check(if (settings.modelVariant == ModelVariant.FP16) R.id.modelFp16 else R.id.modelInt8w)

        fun save() {
            val variant = if (d.model.checkedRadioButtonId == R.id.modelFp16) ModelVariant.FP16 else ModelVariant.INT8W
            if (variant != settings.modelVariant) {
                settings.modelVariant = variant
                switchModel()
            }
            settings.language = if (d.language.checkedRadioButtonId == R.id.langHi) "hi" else "en"
            settings.volume = d.volume.progress / 10f
            settings.sensitivity = when (d.sensitivity.checkedRadioButtonId) {
                R.id.sensLow -> Sensitivity.LOW
                R.id.sensHigh -> Sensitivity.HIGH
                else -> Sensitivity.NORMAL
            }
            settings.fusionMode = if (d.fusion.checkedRadioButtonId == R.id.fusionOr) FusionMode.OR_GATE else FusionMode.TIERED
            settings.voiceEnabled = d.voice.isChecked
            settings.vibrationEnabled = d.vibration.isChecked
            guard?.onSettingsChanged()
        }

        d.shareLog.setOnClickListener { shareTripLog() }
        d.demo.setText(if (guard?.demoActive == true) R.string.settings_demo_stop else R.string.settings_demo_start)
        d.testAlert.setOnClickListener {
            save()
            d.testAlert.postDelayed({
                guard?.testAlert(Assessment(Severity.CRITICAL, Hazard.ICE, visionAlert = true, terrainAlert = true))
                if (settings.language == "hi" && guard?.voiceLanguage == "en") {
                    Toast.makeText(this, R.string.settings_hindi_missing, Toast.LENGTH_LONG).show()
                }
            }, 300)
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.settings)
            .setView(d.root)
            .setPositiveButton(R.string.settings_done) { _, _ -> save() }
            .show()
        d.demo.setOnClickListener {
            save()
            if (guard?.toggleDemo() != true) Toast.makeText(this, R.string.demo_route_loading, Toast.LENGTH_SHORT).show()
            dialog.dismiss()
        }
    }

    private fun showCameraStatus(status: Status, msg: Int) = setStatus(binding.chipCamera, status, getString(msg))

    private fun setStatus(chip: TextView, status: Status, text: String) {
        chip.text = text
        chip.background.mutate().setTint(ContextCompat.getColor(this, status.color))
    }

    private enum class Status(@ColorRes val color: Int) {
        OK(R.color.band_green),
        PENDING(R.color.status_pending),
        ERROR(R.color.band_red),
    }

    private companion object {
        const val TAG = "SnowGuard"
        const val MIN_CONFIDENCE = 0.5f
        const val STOPPED_SPEED_MPS = 1.0
    }
}
