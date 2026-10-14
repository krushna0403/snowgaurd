"""Grad-CAM heatmaps to check the model looks at the road surface, not sky or bonnet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import CLASSES, IMG_SIZE
from .model import GRADCAM_LAYER


def gradcam(model, images: np.ndarray, class_idx: np.ndarray | None = None,
            layer_name: str = GRADCAM_LAYER) -> tuple[np.ndarray, np.ndarray]:
    """Return (heatmaps in [0,1] at input resolution, probabilities) for a batch."""
    import keras
    import tensorflow as tf

    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(images, tf.float32)
    with tf.GradientTape() as tape:
        conv, probs = grad_model(x, training=False)
        idx = tf.argmax(probs, axis=1) if class_idx is None else tf.constant(class_idx)
        score = tf.gather(probs, idx, axis=1, batch_dims=1)
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    cam = tf.image.resize(cam[..., None], images.shape[1:3])[..., 0]
    return cam.numpy(), probs.numpy()


def save_gradcam_grid(model, paths: list[str], labels: np.ndarray, out_path: str | Path,
                      per_class: int = 3, seed: int = 0) -> None:
    """For each class, show correct and incorrect predictions with their heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .data import load_image

    rng = np.random.default_rng(seed)
    rows = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(labels == c)
        if idx.size:
            rows.append(rng.choice(idx, size=min(per_class * 4, idx.size), replace=False))
    if not rows:
        return
    chosen = np.concatenate(rows)
    images = np.stack([load_image(paths[i], IMG_SIZE).numpy() for i in chosen])
    cams, probs = gradcam(model, images)
    preds = probs.argmax(1)

    picks: list[int] = []
    for c in range(len(CLASSES)):
        mine = [k for k, i in enumerate(chosen) if labels[i] == c]
        right = [k for k in mine if preds[k] == c][:per_class]
        wrong = [k for k in mine if preds[k] != c][:per_class]
        picks.extend(right + wrong)

    cols = 4
    nrows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(nrows, cols * 2, figsize=(cols * 4, nrows * 2.1))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for n, k in enumerate(picks):
        r, col = divmod(n, cols)
        img = images[k].astype(np.uint8)
        true, pred = CLASSES[labels[chosen[k]]], CLASSES[preds[k]]
        ok = "OK" if true == pred else "X"
        axes[r, col * 2].imshow(img)
        axes[r, col * 2].set_title(f"{ok} {true}", fontsize=7)
        axes[r, col * 2 + 1].imshow(img)
        axes[r, col * 2 + 1].imshow(cams[k], cmap="jet", alpha=0.45)
        axes[r, col * 2 + 1].set_title(f"pred {pred} {probs[k].max():.2f}", fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import keras

    from .data import read_manifest, split_arrays

    ap = argparse.ArgumentParser(description="Grad-CAM grid for a SnowGuard model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--per-class", type=int, default=3)
    args = ap.parse_args(argv)

    model = keras.models.load_model(args.model)
    paths, y = split_arrays(read_manifest(args.manifest), args.split)
    save_gradcam_grid(model, paths, y, args.out, args.per_class)


if __name__ == "__main__":
    main()
