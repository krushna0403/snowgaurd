"""Single source of truth for the SnowGuard label space.

RSCD annotates every image with friction x material x unevenness (27 classes,
e.g. ``dry_asphalt_slight``, ``water_mud``, ``fresh_snow``). SnowGuard only
cares about the friction state, so material and unevenness are dropped:

    RSCD friction token   -> SnowGuard class
    dry                   -> dry
    wet                   -> wet
    water                 -> water
    fresh_snow            -> fresh_snow
    melted_snow           -> melted_snow
    ice                   -> ice

The class order below is baked into the exported model and must match
``SurfaceClass`` in ``android/app/src/main/java/com/snowguard/app/vision/SurfaceClass.kt``.
"""

from __future__ import annotations

CLASSES: tuple[str, ...] = ("dry", "wet", "water", "fresh_snow", "melted_snow", "ice")
NUM_CLASSES = len(CLASSES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

IMG_SIZE = 224

# Approximate tyre-road friction coefficients used for the dashboard band.
FRICTION_COEFF = {
    "dry": 0.75,
    "wet": 0.50,
    "water": 0.40,
    "melted_snow": 0.30,
    "fresh_snow": 0.20,
    "ice": 0.10,
}


def friction_band(name: str) -> str:
    mu = FRICTION_COEFF[name]
    if mu >= 0.6:
        return "green"
    if mu >= 0.3:
        return "amber"
    return "red"


# Classes that raise a vision-layer alert on their own (FR-F2 "Warning").
# "wet" is not an alert by itself; it only escalates inside a shaded freeze
# zone (black-ice signature), which is decided by the fusion layer.
ALERT_CLASSES: tuple[str, ...] = ("water", "fresh_snow", "melted_snow", "ice")

# BRD section 4.2 targets.
TARGET_ACCURACY = 0.85
TARGET_RECALL = {"ice": 0.85, "fresh_snow": 0.85}
