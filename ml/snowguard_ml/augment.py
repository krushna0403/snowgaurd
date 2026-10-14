"""Training-time augmentations targeting the RSCD (China) -> Himalaya domain shift.

All ops take and return a float32 HxWx3 image in [0, 255]. They run inside
tf.data only; the exported model contains no augmentation.
"""

from __future__ import annotations

import math

import tensorflow as tf


def _maybe(p: float, fn, img):
    return tf.cond(tf.random.uniform([]) < p, lambda: fn(img), lambda: img)


def color_jitter(img):
    img = tf.image.random_brightness(img, max_delta=40.0)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_hue(img, 0.03)
    return img


def low_light(img):
    """Dusk / tunnel / overcast: darken and add sensor noise."""
    img = img * tf.random.uniform([], 0.35, 0.8)
    return img + tf.random.normal(tf.shape(img), stddev=tf.random.uniform([], 2.0, 8.0))


def gaussian_blur(img, ksize: int = 7):
    """Motion / focus / vibration blur from a handlebar or dash mount."""
    sigma = tf.random.uniform([], 0.6, 2.2)
    ax = tf.range(-(ksize // 2), ksize // 2 + 1, dtype=tf.float32)
    k1 = tf.exp(-(ax**2) / (2.0 * sigma**2))
    k1 = k1 / tf.reduce_sum(k1)
    kernel = tf.tile((k1[:, None] * k1[None, :])[:, :, None, None], [1, 1, 3, 1])
    out = tf.nn.depthwise_conv2d(img[None], kernel, [1, 1, 1, 1], padding="SAME")
    return out[0]


def snow_glare(img):
    """Low winter sun reflecting off snow: a soft bright blob plus a global veil."""
    shape = tf.shape(img)
    h, w = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
    cy = tf.random.uniform([], 0.0, 0.6) * h
    cx = tf.random.uniform([], 0.0, 1.0) * w
    radius = tf.random.uniform([], 0.15, 0.5) * w
    yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
    mask = tf.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * radius**2))[..., None]
    img = img + mask * tf.random.uniform([], 0.4, 0.9) * (255.0 - img)
    veil = tf.random.uniform([], 0.0, 0.2)
    return img * (1.0 - veil) + 255.0 * veil


def tilt(img, max_deg: float = 10.0):
    """Mount roll: rotate about the centre, reflecting edges to avoid black corners."""
    theta = tf.random.uniform([], -max_deg, max_deg) * math.pi / 180.0
    shape = tf.shape(img)
    h, w = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
    c, s = tf.cos(theta), tf.sin(theta)
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    # Output->input projective transform [a0 a1 a2 b0 b1 b2 c0 c1].
    transform = tf.stack([c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0.0, 0.0])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None],
        transforms=transform[None],
        output_shape=shape[:2],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    return out[0]


def augment(img, out_size: int):
    img = tf.image.random_crop(img, [out_size, out_size, 3])
    img = tf.image.random_flip_left_right(img)
    img = _maybe(0.8, color_jitter, img)
    img = _maybe(0.3, tilt, img)
    img = _maybe(0.25, gaussian_blur, img)
    img = _maybe(0.2, snow_glare, img)
    img = _maybe(0.15, low_light, img)
    return tf.clip_by_value(img, 0.0, 255.0)
