"""Per-class metrics, confusion matrix and BRD target checks.

``evaluate_predictions`` works on plain probability arrays so the same report
can be produced for the float Keras model and (Phase 2) the INT8 TFLite model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import CLASSES, IMG_SIZE, NUM_CLASSES, TARGET_ACCURACY, TARGET_RECALL


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, out_dir: str | Path, tag: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(NUM_CLASSES))
    y_pred = probs.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        CLASSES[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    targets = {"accuracy": {"value": acc, "target": TARGET_ACCURACY, "met": acc >= TARGET_ACCURACY}}
    for name, target in TARGET_RECALL.items():
        value = per_class[name]["recall"]
        targets[f"{name}_recall"] = {"value": value, "target": target, "met": value >= target}

    report = {
        "tag": tag,
        "n": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "targets": targets,
    }
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(report, indent=2))

    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, CLASSES, rotation=35, ha="right")
    ax.set_yticks(labels, CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"SnowGuard {tag}: acc={acc:.3f}")
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{tag}.png", dpi=150)
    plt.close(fig)

    print(format_report(report))
    return report


def format_report(report: dict) -> str:
    lines = [f"== {report['tag']} (n={report['n']}) accuracy={report['accuracy']:.4f} "
             f"macro-F1={report['macro_f1']:.4f}",
             f"{'class':>12} {'prec':>6} {'recall':>6} {'f1':>6} {'n':>7}"]
    for name, m in report["per_class"].items():
        lines.append(f"{name:>12} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:7d}")
    for name, t in report["targets"].items():
        lines.append(f"  target {name}: {t['value']:.3f} >= {t['target']:.2f} -> {'PASS' if t['met'] else 'FAIL'}")
    return "\n".join(lines)


def predict_keras(model, paths: list[str], batch_size: int = 64, img_size: int = IMG_SIZE) -> np.ndarray:
    from .data import make_dataset

    ds = make_dataset(paths, np.zeros(len(paths), np.int32), NUM_CLASSES, batch_size,
                      training=False, img_size=img_size)
    return model.predict(ds.map(lambda x, _: x), verbose=1)


def main(argv: list[str] | None = None) -> None:
    import keras

    from .data import read_manifest, split_arrays

    ap = argparse.ArgumentParser(description="Evaluate a SnowGuard Keras model on a manifest split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    model = keras.models.load_model(args.model)
    paths, y = split_arrays(read_manifest(args.manifest), args.split)
    probs = predict_keras(model, paths, args.batch_size)
    evaluate_predictions(y, probs, args.out, f"keras_{args.split}")


if __name__ == "__main__":
    main()
