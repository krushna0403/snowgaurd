"""Train the SnowGuard MobileNetV2 classifier.

Stage 1 trains only the new head on a frozen ImageNet backbone; stage 2
fine-tunes the top of the backbone at a low learning rate. Both stages back up
state every epoch so a dropped Colab/Kaggle session can resume by re-running
the same command with the same ``--output-dir``.

Example (Kaggle, dataset attached):
    snowguard-train --data-root "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/train" \
        --output-dir /kaggle/working/run1 --max-per-class 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import CLASSES, FRICTION_COEFF, IMG_SIZE, NUM_CLASSES, friction_band


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="RSCD train directory (or any tree with RSCD-style names)")
    ap.add_argument("--manifest", help="reuse an existing split manifest instead of re-indexing")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--finetune-layers", type=int, default=60, help="backbone layers unfrozen in stage 2")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=3e-5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-per-class", type=int, default=8000, help="train cap per class (0 = all)")
    ap.add_argument("--max-eval-per-class", type=int, default=2000, help="val/test cap per class (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--session-gap", type=float, default=300.0, help="seconds between frames that start a new drive")
    ap.add_argument("--ice-weight", type=float, default=1.5, help="extra class-weight multiplier for ice")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (offline smoke tests)")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--gradcam-per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if not args.data_root and not args.manifest:
        ap.error("one of --data-root or --manifest is required")
    return args


def build_splits(args, out: Path):
    from .data import cap_per_class, index_dataset, read_manifest, split_by_session, summarise, write_manifest

    manifest = out / "manifest.csv"
    if args.manifest:
        samples = read_manifest(args.manifest)
        write_manifest(samples, manifest)
    elif manifest.exists():
        print(f"[data] resuming with existing {manifest}")
        samples = read_manifest(manifest)
    else:
        samples = index_dataset(args.data_root)
        split_by_session(samples, args.val_frac, args.test_frac, args.session_gap, args.seed)
        capped = []
        for split, cap in (("train", args.max_per_class),
                           ("val", args.max_eval_per_class),
                           ("test", args.max_eval_per_class)):
            capped += cap_per_class([s for s in samples if s.split == split], cap, args.seed)
        samples = capped
        write_manifest(samples, manifest)
    print(summarise(samples))
    return samples


def best_val_accuracy(history_csv: Path) -> float | None:
    import csv

    if not history_csv.exists():
        return None
    with history_csv.open(newline="") as f:
        values = [float(r["val_accuracy"]) for r in csv.DictReader(f) if r.get("val_accuracy")]
    return max(values) if values else None


def class_weights(y: np.ndarray, ice_boost: float) -> dict[int, float]:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)), 0.0)
    w[CLASSES.index("ice")] *= ice_boost
    return {i: float(v) for i, v in enumerate(w)}


def callbacks(stage_dir: Path, ckpt: Path, patience: int, best_so_far: float | None = None):
    import keras

    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.BackupAndRestore(str(stage_dir / "backup")),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy", mode="max", save_best_only=True,
                                        initial_value_threshold=best_so_far),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        keras.callbacks.CSVLogger(str(stage_dir / "history.csv"), append=True),
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import keras
    import tensorflow as tf

    from .data import make_dataset, split_arrays
    from .evaluate import evaluate_predictions, predict_keras
    from .gradcam import save_gradcam_grid
    from .model import build_model, set_backbone_trainable

    keras.utils.set_random_seed(args.seed)
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_splits(args, out)

    tr_paths, tr_y = split_arrays(samples, "train")
    va_paths, va_y = split_arrays(samples, "val")
    te_paths, te_y = split_arrays(samples, "test")
    train_ds = make_dataset(tr_paths, tr_y, NUM_CLASSES, args.batch_size, True, args.img_size, args.seed)
    val_ds = make_dataset(va_paths, va_y, NUM_CLASSES, args.batch_size, False, args.img_size)
    cw = class_weights(tr_y, args.ice_weight)
    print("[train] class weights:", {CLASSES[k]: round(v, 3) for k, v in cw.items()})

    weights = None if args.weights.lower() == "none" else args.weights
    model, backbone = build_model(NUM_CLASSES, args.img_size, args.dropout, weights)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    best = out / "best.keras"

    stage1_weights = out / "stage1" / "final.weights.h5"
    if stage1_weights.exists():
        print(f"[train] stage 1 already complete, loading {stage1_weights}")
        model.load_weights(stage1_weights)
    else:
        print(f"[train] stage 1: head only, {args.head_epochs} epochs")
        set_backbone_trainable(backbone, 0)
        model.compile(keras.optimizers.Adam(args.lr_head), loss, metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, class_weight=cw,
                  callbacks=callbacks(out / "stage1", best, patience=3))
        model.save_weights(stage1_weights)

    if args.finetune_epochs > 0:
        print(f"[train] stage 2: fine-tune top {args.finetune_layers} backbone layers, "
              f"{args.finetune_epochs} epochs")
        set_backbone_trainable(backbone, args.finetune_layers)
        model.compile(keras.optimizers.Adam(args.lr_finetune), loss, metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, class_weight=cw,
                  callbacks=callbacks(out / "stage2", best, patience=4,
                                      best_so_far=best_val_accuracy(out / "stage1" / "history.csv")))

    model = keras.models.load_model(best)
    final = out / "snowguard_mobilenetv2.keras"
    model.save(final)
    labels = {
        "classes": list(CLASSES),
        "friction": {c: FRICTION_COEFF[c] for c in CLASSES},
        "band": {c: friction_band(c) for c in CLASSES},
        "input": {"size": args.img_size, "range": [0, 255], "layout": "NHWC RGB"},
    }
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    reports = {}
    for split, paths, y in (("val", va_paths, va_y), ("test", te_paths, te_y)):
        if len(paths):
            reports[split] = evaluate_predictions(y, predict_keras(model, paths, args.batch_size, args.img_size),
                                                  out, f"keras_{split}")
    if len(te_paths):
        save_gradcam_grid(model, te_paths, te_y, out / "gradcam_test.png", args.gradcam_per_class, args.seed)

    (out / "summary.json").write_text(json.dumps({
        "args": vars(args),
        "tensorflow": tf.__version__,
        "model": str(final),
        "val_accuracy": reports.get("val", {}).get("accuracy"),
        "test_accuracy": reports.get("test", {}).get("accuracy"),
        "test_targets": reports.get("test", {}).get("targets"),
    }, indent=2))
    print(f"[train] done -> {final}")


if __name__ == "__main__":
    main()
