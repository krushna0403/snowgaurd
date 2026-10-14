package com.snowguard.app.vision

data class RoiRect(val left: Float, val top: Float, val width: Float, val height: Float) {
    val right get() = left + width
    val bottom get() = top + height
}

/**
 * Road region of interest (FR-V4) in an upright frame: a centred patch in the
 * lower part of the image, above the bonnet/handlebar band, with the same 2:3
 * (h:w) aspect as RSCD's road patches so the resize to 224x224 distorts live
 * frames the same way it distorted the training data.
 */
object RoadRoi {
    const val WIDTH_FRACTION = 0.8f
    const val BOTTOM_MARGIN_FRACTION = 0.1f
    private const val ASPECT_H_OVER_W = 240f / 360f

    fun compute(frameWidth: Float, frameHeight: Float): RoiRect {
        var w = frameWidth * WIDTH_FRACTION
        var h = w * ASPECT_H_OVER_W
        val maxH = frameHeight * (1f - BOTTOM_MARGIN_FRACTION)
        if (h > maxH) {
            h = maxH
            w = h / ASPECT_H_OVER_W
        }
        val left = (frameWidth - w) / 2f
        val top = frameHeight * (1f - BOTTOM_MARGIN_FRACTION) - h
        return RoiRect(left, top, w, h)
    }
}
