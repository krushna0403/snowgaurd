"""Fusion ablation (BRD Objective 4): vision-only vs terrain-only vs OR gate vs tiered.

Simulated drives over the real Manali-Leh route database:

* Terrain layer: real horizon angles, NOAA sun position along a 40 km/h drive
  and the altitude-month freeze thresholds, exactly as the app computes them.
* Ground truth: road-surface segments (150-600 m) drawn from condition-dependent
  priors (shaded freeze zone / sunlit high road / low road). Black ice is a
  separate truth class that *looks* wet to the camera.
* Vision layer: per-frame predictions sampled from the trained model's real
  test confusion matrix at 14 fps, with frame-to-frame correlated errors, then
  the app's 4-of-6 hysteresis smoother.
* Fusion: the app's OR gate and tiered rules plus its cooldowns.

The surface priors are scenario assumptions, not measurements; they are
written into the report so the result can be judged with them in view.

Example:
    snowguard-ablation --db out/route_manali_leh.db \
        --confusion ../ml/runs/phase2_export/metrics_tflite_fp32.json --out out/ablation
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .solar import solar_position

CLASSES = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]
TRUTH = CLASSES + ["black_ice"]
APPEARS_AS = np.array([0, 1, 2, 3, 4, 5, 1])  # black ice is optically wet road
VISION_HAZARD = np.array([False, False, True, True, True, True])
TRUE_HAZARD = np.array([False, False, True, True, True, True, True])
WET, BLACK_ICE = 1, 6

# Scenario priors over TRUTH, per road condition.
PRIORS = {
    "freeze_zone": [0.25, 0.10, 0.00, 0.15, 0.10, 0.20, 0.20],
    # Some refrozen melt-water on sunlit road, which terrain cannot predict.
    "sunlit_high": [0.53, 0.15, 0.05, 0.10, 0.12, 0.03, 0.02],
    "low_road": [0.75, 0.15, 0.07, 0.00, 0.03, 0.00, 0.00],
}
SEGMENT_M = (150, 600)

SPEED_KMH = 40.0
FPS = 14.0
FRAME_CORRELATION = 0.7
SMOOTH_N, SMOOTH_K, SMOOTH_CLEAR = 6, 4, 2

NONE, ADVISORY, WARNING, CRITICAL = 0, 1, 2, 3
COOLDOWN_S = {ADVISORY: 180.0, WARNING: 45.0, CRITICAL: 20.0}
SYSTEMS = ("vision_only", "terrain_only", "or_gate", "tiered")

IST = timezone(timedelta(hours=5, minutes=30))
SCENARIOS = [
    (f"{m} {h:02d}:00 IST", datetime(2026, mo, 15, h, 0, tzinfo=IST))
    for m, mo in (("15 Jan", 1), ("15 Apr", 4), ("15 Jul", 7))
    for h in (6, 9, 13)
]


def load_route(db: Path) -> dict:
    con = sqlite3.connect(db)
    meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
    rows = con.execute("SELECT lat, lon, dist_m, elev_m, tunnel, horizon FROM samples ORDER BY id").fetchall()
    n_az, scale = int(meta["n_azimuth"]), float(meta["horizon_scale"])
    return {
        "lat": np.array([r[0] for r in rows]),
        "lon": np.array([r[1] for r in rows]),
        "dist": np.array([r[2] for r in rows]),
        "elev": np.array([r[3] for r in rows]),
        "tunnel": np.array([bool(r[4]) for r in rows]),
        "horizon": np.stack([np.frombuffer(r[5], dtype="<i2") for r in rows]).astype(np.float64) / scale,
        "thresholds": meta["freeze_thresholds"],
        "n_az": n_az,
    }


def confusion_probs(path: Path) -> np.ndarray:
    cm = np.array(json.loads(path.read_text())["confusion_matrix"], dtype=np.float64)
    cm += 0.5  # keep every transition possible
    return cm / cm.sum(axis=1, keepdims=True)


def terrain_layer(r: dict, start: datetime) -> dict:
    t = start.timestamp() + r["dist"] / (SPEED_KMH / 3.6)
    el, az = solar_position(r["lat"], r["lon"], t)
    step = 360.0 / r["n_az"]
    f = (az % 360.0) / step
    k0 = f.astype(int) % r["n_az"]
    k1 = (k0 + 1) % r["n_az"]
    w = f - np.floor(f)
    idx = np.arange(len(el))
    hz = r["horizon"][idx, k0] * (1 - w) + r["horizon"][idx, k1] * w
    shaded = (el <= hz) & ~r["tunnel"]
    month = datetime.fromtimestamp(t[0], IST).month
    thr = next(x["min_elev_m"] for x in r["thresholds"] if month in x["months"])
    above = r["elev"] >= thr
    return {"time": t, "flag": shaded & above, "above": above, "sunlit": ~shaded & (el > 0), "night": el <= 0}


def ground_truth(rng, terrain: dict, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(terrain["flag"])
    truth = np.zeros(n, dtype=int)
    seg = np.zeros(n, dtype=int)
    i, s = 0, 0
    while i < n:
        length = max(1, int(rng.uniform(*SEGMENT_M) / spacing))
        cond = "freeze_zone" if terrain["flag"][i] else "sunlit_high" if terrain["above"][i] else "low_road"
        truth[i:i + length] = rng.choice(len(TRUTH), p=PRIORS[cond])
        seg[i:i + length] = s
        i += length
        s += 1
    return truth, seg


def vision_layer(rng, truth: np.ndarray, probs: np.ndarray, frames_per_sample: int,
                 correlation: float = FRAME_CORRELATION) -> np.ndarray:
    """Per-sample smoothed vision class (mode of the stable class over the sample's frames, -1 = none)."""
    cum = probs.cumsum(axis=1)
    app = APPEARS_AS[truth]
    base = (rng.random(len(app))[:, None] > cum[app]).sum(axis=1)
    app_f = np.repeat(app, frames_per_sample)
    indep = (rng.random(len(app_f))[:, None] > cum[app_f]).sum(axis=1)
    frames = np.where(rng.random(len(app_f)) < correlation, np.repeat(base, frames_per_sample), indep)
    frames = np.minimum(frames, len(CLASSES) - 1)

    stable_f = np.empty(len(frames), dtype=int)
    counts = [0] * len(CLASSES)
    hist = [-1] * SMOOTH_N
    pos, stable = 0, -1
    for j, c in enumerate(frames.tolist()):
        old = hist[pos]
        if old >= 0:
            counts[old] -= 1
        hist[pos] = c
        counts[c] += 1
        pos = (pos + 1) % SMOOTH_N
        lead = max(range(len(CLASSES)), key=counts.__getitem__)
        if stable >= 0 and counts[stable] >= SMOOTH_CLEAR:
            if lead != stable and counts[lead] >= SMOOTH_K and counts[lead] > counts[stable]:
                stable = lead
        else:
            stable = lead if counts[lead] >= SMOOTH_K else -1
        stable_f[j] = stable

    per = stable_f.reshape(-1, frames_per_sample) + 1  # shift so -1 -> 0 for bincount
    out = np.array([np.bincount(row, minlength=len(CLASSES) + 1).argmax() for row in per]) - 1
    return out


FREEZE_ZONE_KEY, BLACK_ICE_KEY = 10, 11


def severities(vision: np.ndarray, flag: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(severity, hazard key) per sample for each system, mirroring FusionEngine.assess."""
    v_alert = (vision >= 0) & VISION_HAZARD[np.clip(vision, 0, None)]
    v_wet = vision == WET
    zone = np.full(len(flag), FREEZE_ZONE_KEY)

    tiered = np.full(len(flag), NONE)
    t_key = np.full(len(flag), -1)
    tiered[flag], t_key[flag] = ADVISORY, FREEZE_ZONE_KEY
    tiered[v_alert], t_key[v_alert] = WARNING, vision[v_alert]
    black = v_wet & flag
    tiered[black], t_key[black] = CRITICAL, BLACK_ICE_KEY
    both = v_alert & flag
    tiered[both], t_key[both] = CRITICAL, vision[both]
    return {
        "vision_only": (np.where(v_alert, WARNING, NONE), np.where(v_alert, vision, -1)),
        "terrain_only": (np.where(flag, WARNING, NONE), np.where(flag, zone, -1)),
        "or_gate": (np.where(v_alert | flag, WARNING, NONE), np.where(v_alert, vision, np.where(flag, zone, -1))),
        "tiered": (tiered, t_key),
    }


def fired_alerts(sev: np.ndarray, key: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Severity of each alert that would actually sound, mirroring AlertGate (per-hazard cooldown)."""
    fired = np.full(len(sev), NONE)
    last: dict[int, tuple[int, float]] = {}
    for i, (s, k) in enumerate(zip(sev.tolist(), key.tolist())):
        if s == NONE:
            continue
        prev = last.get(k)
        if prev is None or s > prev[0] or time_s[i] - prev[1] >= COOLDOWN_S[s]:
            fired[i] = s
            last[k] = (s, time_s[i])
    return fired


def evaluate(sev: np.ndarray, fired: np.ndarray, truth: np.ndarray, seg: np.ndarray,
             sunlit: np.ndarray, spacing: float) -> dict:
    hazard = TRUE_HAZARD[truth]
    km = len(truth) * spacing / 1000
    out = {}
    # Hazard segments: detected if the system is alerting anywhere on the segment.
    seg_ids = np.unique(seg[hazard])
    detected, urgent, critical, black = [], [], [], []
    for s in seg_ids:
        m = seg == s
        detected.append((sev[m] > NONE).any())
        urgent.append((sev[m] >= WARNING).any())
        critical.append((sev[m] == CRITICAL).any())
        if truth[m][0] == BLACK_ICE:
            black.append(((sev[m] > NONE).any(), (sev[m] >= WARNING).any()))
    out["hazard_segments"] = len(seg_ids)
    out["detection_rate"] = float(np.mean(detected)) if detected else None
    out["urgent_detection_rate"] = float(np.mean(urgent)) if urgent else None
    out["black_ice_detection_rate"] = float(np.mean([b[0] for b in black])) if black else None
    out["black_ice_urgent_rate"] = float(np.mean([b[1] for b in black])) if black else None
    # False alerts: sounded alerts on non-hazard road, per 10 km of such road.
    safe = ~hazard
    safe_km = safe.sum() * spacing / 1000
    out["false_alerts_per_10km"] = float((fired[safe] > NONE).sum() / safe_km * 10) if safe_km else None
    out["urgent_false_alerts_per_10km"] = float((fired[safe] >= WARNING).sum() / safe_km * 10) if safe_km else None
    dry_sun = (truth == 0) & sunlit
    dry_km = dry_sun.sum() * spacing / 1000
    out["false_alerts_per_10km_sunlit_dry"] = float((fired[dry_sun] > NONE).sum() / dry_km * 10) if dry_km else None
    out["alerts_per_100km"] = float((fired > NONE).sum() / km * 100)
    out["urgent_alerts_per_100km"] = float((fired >= WARNING).sum() / km * 100)
    urgent_fired = fired >= WARNING
    out["urgent_alert_precision"] = float(hazard[urgent_fired].mean()) if urgent_fired.any() else None
    crit = fired == CRITICAL
    out["critical_alerts_per_100km"] = float(crit.sum() / km * 100)
    out["critical_precision"] = float(hazard[crit].mean()) if crit.any() else None
    return out


def mean_dicts(ds: list[dict]) -> dict:
    keys = ds[0].keys()
    res = {}
    for k in keys:
        vals = [d[k] for d in ds if d[k] is not None]
        res[k] = round(float(np.mean(vals)), 4) if vals else None
    return res


def plot(path: Path, summary: dict, correlation: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("detection_rate", "Hazard segments detected", 1.0),
        ("black_ice_detection_rate", "Black-ice segments detected", 1.0),
        ("false_alerts_per_10km_sunlit_dry", "False alerts / 10 km (sunlit dry)", None),
        ("urgent_false_alerts_per_10km", "Warning+ false alerts / 10 km safe road", None),
        ("urgent_alert_precision", "Precision of Warning+ alerts", 1.0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
    colors = ["#4FC3F7", "#8D6E63", "#FFA000", "#2E7D32"]
    for ax, (key, title, ymax) in zip(axes, panels):
        vals = [summary[s][key] or 0 for s in SYSTEMS]
        ax.bar(range(len(SYSTEMS)), vals, color=colors)
        ax.set_xticks(range(len(SYSTEMS)), [s.replace("_", "\n") for s in SYSTEMS], fontsize=8)
        ax.set_title(title, fontsize=10)
        if ymax:
            ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"SnowGuard fusion ablation (simulated drives, Manali-Leh, frame correlation {correlation})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="out/route_manali_leh.db")
    ap.add_argument("--confusion", required=True, help="metrics_*.json with a test confusion_matrix")
    ap.add_argument("--out", default="out/ablation")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--correlation", type=float, default=FRAME_CORRELATION,
                    help="share of frames repeating the per-75 m prediction (0 = independent frames)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = load_route(Path(args.db))
    probs = confusion_probs(Path(args.confusion))
    spacing = float(np.median(np.diff(r["dist"])))
    frames_per_sample = max(SMOOTH_N, round(spacing / (SPEED_KMH / 3.6) * FPS))

    per_system: dict[str, list[dict]] = {s: [] for s in SYSTEMS}
    per_scenario = {}
    for label, start in SCENARIOS:
        terrain = terrain_layer(r, start)
        scen = {s: [] for s in SYSTEMS}
        for seed in range(args.seeds):
            rng = np.random.default_rng([zlib.crc32(label.encode()), seed])
            truth, seg = ground_truth(rng, terrain, spacing)
            vision = vision_layer(rng, truth, probs, frames_per_sample, args.correlation)
            for name, (sev, key) in severities(vision, terrain["flag"]).items():
                res = evaluate(sev, fired_alerts(sev, key, terrain["time"]), truth, seg, terrain["sunlit"], spacing)
                scen[name].append(res)
                per_system[name].append(res)
        per_scenario[label] = {s: mean_dicts(v) for s, v in scen.items()}
        print(f"[ablation] {label}: freeze-zone {terrain['flag'].mean() * 100:.0f}% of route | "
              + " | ".join(f"{s} det {per_scenario[label][s]['detection_rate']:.2f}" for s in SYSTEMS))

    summary = {s: mean_dicts(v) for s, v in per_system.items()}
    report = {
        "assumptions": {
            "speed_kmh": SPEED_KMH, "fps": FPS, "frame_correlation": args.correlation,
            "smoother": f"{SMOOTH_K} of {SMOOTH_N}, clear below {SMOOTH_CLEAR}",
            "cooldown_s": {k: v for k, v in zip(("advisory", "warning", "critical"), COOLDOWN_S.values())},
            "segment_m": SEGMENT_M, "surface_priors": {k: dict(zip(TRUTH, v)) for k, v in PRIORS.items()},
            "confusion_source": str(args.confusion), "seeds": args.seeds,
            "scenarios": [label for label, _ in SCENARIOS],
        },
        "summary": summary,
        "per_scenario": per_scenario,
    }
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2))
    plot(out / "ablation.png", summary, args.correlation)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
