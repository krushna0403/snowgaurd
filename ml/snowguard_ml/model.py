"""MobileNetV2 transfer-learning model.

The model takes raw RGB pixels in [0, 255] and normalises internally to the
[-1, 1] range MobileNetV2's ImageNet weights were trained with, so the Android
side only has to resize the frame. Backbone layers are flattened into the
outer model so Grad-CAM can reach ``out_relu`` directly.
"""

from __future__ import annotations

import keras
from keras import layers

from .config import IMG_SIZE, NUM_CLASSES

GRADCAM_LAYER = "out_relu"


def build_model(
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
    alpha: float = 1.0,
) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    backbone = keras.applications.MobileNetV2(
        input_tensor=x, include_top=False, weights=weights, alpha=alpha
    )
    x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    # float32 head keeps softmax numerically stable under mixed precision.
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probs")(x)
    model = keras.Model(inputs, outputs, name="snowguard_mobilenetv2")
    return model, backbone


def set_backbone_trainable(backbone: keras.Model, n_trainable: int) -> None:
    """Freeze the backbone except its last ``n_trainable`` layers.

    BatchNorm layers stay frozen throughout so their ImageNet statistics are
    not destroyed by small fine-tuning batches (frozen BN runs in inference mode).
    """
    total = len(backbone.layers)
    for i, layer in enumerate(backbone.layers):
        trainable = n_trainable > 0 and i >= total - n_trainable
        if isinstance(layer, layers.BatchNormalization):
            trainable = False
        layer.trainable = trainable
