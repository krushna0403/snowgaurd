package com.snowguard.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.snowguard.app.R
import com.snowguard.app.vision.RoadRoi
import kotlin.math.max

/**
 * Outlines the road ROI on top of a PreviewView in FILL_CENTER mode, so the
 * user can aim the mount. Preview and analysis both use 4:3, so the analysis
 * frame maps onto the view with the same centre-crop scaling.
 */
class RoiOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = resources.displayMetrics.density * 2f
        pathEffect = DashPathEffect(floatArrayOf(24f, 16f), 0f)
        color = ContextCompat.getColor(context, R.color.roi_outline)
    }

    private var frameW = 0
    private var frameH = 0

    fun setFrameSize(uprightWidth: Int, uprightHeight: Int) {
        if (uprightWidth != frameW || uprightHeight != frameH) {
            frameW = uprightWidth
            frameH = uprightHeight
            invalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        if (frameW == 0 || frameH == 0) return
        val scale = max(width.toFloat() / frameW, height.toFloat() / frameH)
        val dx = (width - frameW * scale) / 2f
        val dy = (height - frameH * scale) / 2f
        val roi = RoadRoi.compute(frameW.toFloat(), frameH.toFloat())
        canvas.drawRect(
            dx + roi.left * scale,
            dy + roi.top * scale,
            dx + roi.right * scale,
            dy + roi.bottom * scale,
            paint,
        )
    }
}
