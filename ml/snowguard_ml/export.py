"""Export the trained Keras model to TFLite (float32 + full-integer INT8) and verify it.

* Full-integer INT8 uses post-training quantisation calibrated on a
  class-balanced sample of the *train* split (never test), with uint8 input and
  output. Dynamic-range (INT8 weights) and float16 variants are built too.
* Every variant is scored on the held-out test split; the app gets the first of
  VARIANTS whose accuracy drop vs float32 is within the BRD's 2 points.
* A per-class log-prior bias is tuned on the INT8 *validation* outputs so ice
  and fresh-snow recall clear their targets with margin, then checked once on
  test. The app adds the same bias before argmax (labels.json ``class_bias``).
* Desktop latency is indicative only; the on-device number comes from the app.

Example:
    snowguard-export --model runs/phase1/snowguard_mobilenetv2.keras \
        --manifest runs/phase1/manifest.csv --out runs/phase1/tflite \
        --android-assets ../android/app/src/main/assets
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .config import ALERT_CLASSES, CLASSES, FRICTION_COEFF, IMG_SIZE, NUM_CLASSES, TARGET_RECALL, friction_band

MODEL_NAME = "snowguard_mnv2.tflite"
MAX_ACCURACY_DROP = 0.02
# Preference order for the app model: smallest/fastest variant within MAX_ACCURACY_DROP of float32.
VARIANTS = ("int8", "dynamic_int8", "fp16", "fp32")
BIAS_RECALL_MARGIN = 0.02
BIAS_GRID = np.round(np.arange(-1.0, 1.501, 0.05), 2)


def _log_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, 1e-4, 1.0))


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = _log_probs(probs) + bias
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def tune_bias(y: np.ndarray, probs: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Coordinate search over per-class log-prior offsets (dry fixed at 0).

    Maximises accuracy subject to each TARGET_RECALL class reaching its target
    plus BIAS_RECALL_MARGIN; shortfalls are penalised heavily.
    """
    logp = _log_probs(probs)
    floors = {CLASSES.index(c): t + BIAS_RECALL_MARGIN for c, t in TARGET_RECALL.items()}
    masks = {k: y == k for k in floors}

    def score(b: np.ndarray) -> float:
        pred = np.argmax(logp + b, axis=1)
        acc = float(np.mean(pred == y))
        short = sum(max(0.0, t - float(np.mean(pred[masks[k]] == k))) for k, t in floors.items() if masks[k].any())
        return acc - 10.0 * short

    bias = np.zeros(NUM_CLASSES)
    best = score(bias)
    for _ in range(rounds):
        improved = False
        for k in range(1, NUM_CLASSES):
            for v in BIAS_GRID:
                trial = bias.copy()
                trial[k] = v
                s = score(trial)
                if s > best + 1e-9:
                    best, bias, improved = s, trial, True
        if not improved:
            break
    return bias


def hazard_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    """Alert-level view: was a hazardous surface flagged as *some* hazard, and ice not called dry/wet?"""
    alert = np.array([c in ALERT_CLASSES for c in CLASSES])
    pred = np.argmax(probs, axis=1)
    true_hazard, pred_hazard = alert[y], alert[pred]
    ice = y == CLASSES.index("ice")
    return {
        "hazard_recall": float(np.mean(pred_hazard[true_hazard])) if true_hazard.any() else None,
        "hazard_precision": float(np.mean(true_hazard[pred_hazard])) if pred_hazard.any() else None,
        "ice_missed_as_safe": float(np.mean(~pred_hazard[ice])) if ice.any() else None,
    }


def remap_paths(paths: list[str], data_root: str | None) -> list[str]:
    """Point manifest paths (absolute on the training machine) at a local RSCD ``train`` dir."""
    if not data_root:
        return paths
    root = Path(data_root)
    return [str(root / p.replace("\\", "/").rsplit("/train/", 1)[-1]) for p in paths]


def float32_clone(path: str | Path):
    """Reload weights into a fresh float32 graph (drops any mixed-precision policy)."""
    import keras

    from .model import build_model

    keras.mixed_precision.set_global_policy("float32")
    trained = keras.models.load_model(path, compile=False)
    size = trained.input_shape[1]
    model, _ = build_model(NUM_CLASSES, size, weights=None)
    model.set_weights(trained.get_weights())
    return model, size


def representative_images(paths: list[str], labels: np.ndarray, n: int, size: int, seed: int):
    from .data import load_image

    rng = np.random.default_rng(seed)
    per_class = max(1, n // NUM_CLASSES)
    chosen = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == c)
        if idx.size:
            chosen.extend(rng.choice(idx, size=min(per_class, idx.size), replace=False))
    rng.shuffle(chosen)

    def gen():
        for i in chosen:
            yield [load_image(paths[i], size).numpy()[None].astype(np.float32)]

    return gen, len(chosen)


def convert(model, saved_dir: Path, rep_gen=None, dynamic: bool = False, fp16: bool = False) -> bytes:
    """float32 by default; full-integer INT8 with ``rep_gen``; INT8 weights only with ``dynamic``;
    float16 weights with ``fp16``."""
    import tensorflow as tf

    if not saved_dir.exists():
        model.export(str(saved_dir), format="tf_saved_model", verbose=False)
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    if fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif dynamic:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif rep_gen is not None:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
    return conv.convert()


class TFLiteModel:
    def __init__(self, model_bytes: bytes, threads: int = 4):
        import tensorflow as tf

        self.interp = tf.lite.Interpreter(model_content=model_bytes, num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def _quant(self, x: np.ndarray) -> np.ndarray:
        if self.inp["dtype"] == np.float32:
            return x.astype(np.float32)
        scale, zero = self.inp["quantization"]
        q = np.round(x / scale + zero) if scale else x
        return np.clip(q, 0, 255).astype(self.inp["dtype"])

    def predict_one(self, img: np.ndarray) -> np.ndarray:
        self.interp.set_tensor(self.inp["index"], self._quant(img[None]))
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])[0]
        if self.out["dtype"] != np.float32:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y

    def predict(self, paths: list[str], size: int) -> np.ndarray:
        from .data import load_image

        return np.stack([self.predict_one(load_image(p, size).numpy()) for p in paths])

    def latency_ms(self, size: int, runs: int = 50) -> float:
        x = np.random.default_rng(0).uniform(0, 255, (size, size, 3)).astype(np.float32)
        for _ in range(5):
            self.predict_one(x)
        t = time.perf_counter()
        for _ in range(runs):
            self.predict_one(x)
        return (time.perf_counter() - t) * 1000 / runs


def main(argv: list[str] | None = None) -> None:
    from .data import read_manifest, split_arrays
    from .evaluate import evaluate_predictions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="trained .keras model")
    ap.add_argument("--manifest", required=True, help="manifest.csv from training (for calibration/test splits)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-images", type=int, default=600)
    ap.add_argument("--max-test", type=int, default=0, help="cap test images scored (0 = whole split)")
    ap.add_argument("--max-val", type=int, default=0, help="cap val images used for bias tuning (0 = all)")
    ap.add_argument("--data-root", help="RSCD train/ dir, if the manifest paths are from another machine")
    ap.add_argument("--no-bias", action="store_true", help="skip class-bias tuning")
    ap.add_argument("--android-assets", help="copy the chosen app model and labels.json here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = read_manifest(args.manifest)
    rng = np.random.default_rng(args.seed)

    def split(name: str, cap: int):
        paths, y = split_arrays(samples, name)
        if cap and len(paths) > cap:
            keep = rng.choice(len(paths), cap, replace=False)
            paths, y = [paths[i] for i in keep], y[keep]
        return remap_paths(paths, args.data_root), y

    tr_paths, tr_y = split("train", 0)
    va_paths, va_y = split("val", args.max_val)
    te_paths, te_y = split("test", args.max_test)

    model, size = float32_clone(args.model)
    saved = out / "saved_model"
    if saved.exists():
        shutil.rmtree(saved)

    blobs = {}
    print("[export] float32 TFLite")
    blobs["fp32"] = convert(model, saved)
    print("[export] dynamic-range TFLite (INT8 weights, float activations)")
    blobs["dynamic_int8"] = convert(model, saved, dynamic=True)
    print("[export] float16 TFLite")
    blobs["fp16"] = convert(model, saved, fp16=True)
    gen, n_calib = representative_images(tr_paths, tr_y, args.calib_images, size, args.seed)
    print(f"[export] full-integer INT8 TFLite, calibrating on {n_calib} train images")
    blobs["int8"] = convert(model, saved, gen)
    for name, blob in blobs.items():
        (out / f"snowguard_mnv2_{name}.tflite").write_bytes(blob)

    def summarise(rep: dict, probs: np.ndarray, y: np.ndarray) -> dict:
        per = rep.get("per_class", {})
        return {
            "accuracy": rep.get("accuracy"),
            "ice_recall": per.get("ice", {}).get("recall"),
            "fresh_snow_recall": per.get("fresh_snow", {}).get("recall"),
            **hazard_metrics(y, probs),
        }

    results: dict = {}
    models, test_probs = {}, {}
    for name in ("fp32", "fp16", "dynamic_int8", "int8"):
        tag = f"tflite_{name}"
        m = models[name] = TFLiteModel(blobs[name])
        print(f"[export] scoring {tag} on {len(te_paths)} test images")
        probs = test_probs[name] = m.predict(te_paths, size) if len(te_paths) else None
        rep = evaluate_predictions(te_y, probs, out, tag) if probs is not None else {}
        results[tag] = {
            "size_mb": round(len(blobs[name]) / 2**20, 2),
            "desktop_latency_ms": round(m.latency_ms(size), 2),
            **(summarise(rep, probs, te_y) if probs is not None else {}),
        }

    base = results["tflite_fp32"].get("accuracy")
    for name in ("fp16", "dynamic_int8", "int8"):
        acc = results[f"tflite_{name}"].get("accuracy")
        drop = None if base is None or acc is None else base - acc
        results[f"tflite_{name}"]["accuracy_drop_vs_fp32"] = drop
        results[f"tflite_{name}"]["drop_within_target"] = None if drop is None else drop <= MAX_ACCURACY_DROP
    chosen = next(
        (v for v in VARIANTS if v == "fp32" or results[f"tflite_{v}"].get("drop_within_target") in (True, None)),
        "fp32",
    )
    results["app_model"] = chosen
    print(f"[export] app model: {chosen}")
    shutil.copy(out / f"snowguard_mnv2_{chosen}.tflite", out / MODEL_NAME)

    bias = np.zeros(NUM_CLASSES)
    if not args.no_bias and len(va_paths):
        print(f"[export] tuning class bias on {len(va_paths)} val images ({chosen} outputs)")
        va_probs = models[chosen].predict(va_paths, size)
        bias = tune_bias(va_y, va_probs)
        val_before = evaluate_predictions(va_y, va_probs, out, "app_val")
        val_after = evaluate_predictions(va_y, apply_bias(va_probs, bias), out, "app_val_biased")
        results["class_bias"] = {c: round(float(b), 3) for c, b in zip(CLASSES, bias)}
        results["app_val"] = summarise(val_before, va_probs, va_y)
        results["app_val_biased"] = summarise(val_after, apply_bias(va_probs, bias), va_y)
        if test_probs[chosen] is not None:
            biased = apply_bias(test_probs[chosen], bias)
            rep = evaluate_predictions(te_y, biased, out, "app_test_biased")
            results["app_test_biased"] = {**summarise(rep, biased, te_y), "targets": rep.get("targets")}

    (out / "export_report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    full_int8 = chosen == "int8"
    labels = {
        "model": MODEL_NAME,
        "quantization": chosen,
        "classes": list(CLASSES),
        "class_bias": [round(float(b), 3) for b in bias],
        "friction": {c: FRICTION_COEFF[c] for c in CLASSES},
        "band": {c: friction_band(c) for c in CLASSES},
        "input": {"size": size, "dtype": "uint8" if full_int8 else "float32", "range": [0, 255], "layout": "NHWC RGB"},
    }
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    if args.android_assets:
        assets = Path(args.android_assets)
        assets.mkdir(parents=True, exist_ok=True)
        for stale in assets.glob("snowguard_mnv2*.tflite"):
            stale.unlink()
        shutil.copy(out / MODEL_NAME, assets / MODEL_NAME)
        shutil.copy(out / "labels.json", assets / "labels.json")
        print(f"[export] copied model + labels to {assets}")


if __name__ == "__main__":
    main()
