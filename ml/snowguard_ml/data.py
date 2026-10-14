"""RSCD indexing, 27->6 label mapping, leakage-free splitting and tf.data pipelines.

RSCD file names start with a capture timestamp, e.g.
``train/dry_asphalt_slight/202201270008469-dry-asphalt-slight.jpg``. Frames
captured within ``session_gap_s`` of each other belong to the same drive
("session"); splits are made over whole sessions so near-duplicate consecutive
frames never straddle train and test. RSCD's own ``vali_20k``/``test_50k`` are
randomly sampled from the same drives, so they are *not* used for the headline
numbers.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import CLASS_TO_INDEX, CLASSES, IMG_SIZE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TS_RE = re.compile(r"^(\d{14})")


@dataclass
class Sample:
    path: str
    label: int
    timestamp: float | None
    session: int = -1
    split: str = ""


def rscd_label(path: Path) -> str | None:
    """Map an RSCD path to a SnowGuard class using the folder name, then the file name."""
    for text in (path.parent.name, path.stem):
        tokens = set(re.split(r"[^a-z]+", text.lower()))
        if "fresh" in tokens and "snow" in tokens:
            return "fresh_snow"
        if "melted" in tokens and "snow" in tokens:
            return "melted_snow"
        for name in ("ice", "water", "wet", "dry"):
            if name in tokens:
                return name
    return None


def rscd_timestamp(path: Path) -> float | None:
    m = _TS_RE.match(path.stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


def index_dataset(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    skipped = 0
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        name = rscd_label(p)
        if name is None:
            skipped += 1
            continue
        samples.append(Sample(str(p), CLASS_TO_INDEX[name], rscd_timestamp(p)))
    if not samples:
        raise FileNotFoundError(f"No labelled images found under {root}")
    if skipped:
        print(f"[index] skipped {skipped} images with no recognisable label")
    return samples


def assign_sessions(samples: list[Sample], gap_s: float) -> int:
    """Group samples into drives by timestamp gaps. Returns the number of sessions."""
    timed = sorted((s for s in samples if s.timestamp is not None), key=lambda s: s.timestamp)
    session, prev = -1, None
    for s in timed:
        if prev is None or s.timestamp - prev > gap_s:
            session += 1
        s.session = session
        prev = s.timestamp

    untimed = [s for s in samples if s.timestamp is None]
    if untimed:
        print(f"[split] {len(untimed)} images have no timestamp; each becomes its own session "
              "(no leakage guarantee for these)")
        for s in untimed:
            session += 1
            s.session = session
    return session + 1


def _group_split(idx: np.ndarray, y: np.ndarray, groups: np.ndarray, frac: float, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1.0 / frac))
    n_groups = len(np.unique(groups[idx]))
    if n_groups < n_splits:
        raise ValueError(
            f"Only {n_groups} sessions available but {n_splits} folds needed; "
            "lower --session-gap or use a larger subset"
        )
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep, held = next(sgkf.split(idx, y[idx], groups[idx]))
    return idx[keep], idx[held]


def split_by_session(
    samples: list[Sample],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    session_gap_s: float = 300.0,
    seed: int = 42,
) -> None:
    """Assign ``sample.split`` in place to train / val / test with no shared sessions."""
    n_sessions = assign_sessions(samples, session_gap_s)
    print(f"[split] {len(samples)} images in {n_sessions} sessions (gap {session_gap_s:.0f}s)")

    y = np.array([s.label for s in samples])
    groups = np.array([s.session for s in samples])
    all_idx = np.arange(len(samples))

    trainval, test = _group_split(all_idx, y, groups, test_frac, seed)
    train, val = _group_split(trainval, y, groups, val_frac / (1.0 - test_frac), seed + 1)

    for name, ids in (("train", train), ("val", val), ("test", test)):
        for i in ids:
            samples[i].split = name

    leaked = (set(groups[train]) & set(groups[test])) | (set(groups[train]) & set(groups[val]))
    assert not leaked, f"session leakage across splits: {sorted(leaked)[:5]}"


def cap_per_class(samples: list[Sample], cap: int | None, seed: int) -> list[Sample]:
    """Class-balanced subset: keep at most ``cap`` samples per class (sampled by session)."""
    if not cap:
        return samples
    rng = random.Random(seed)
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)
    out: list[Sample] = []
    for items in by_class.values():
        if len(items) > cap:
            # Spread the budget over as many sessions as possible.
            by_session: dict[int, list[Sample]] = defaultdict(list)
            for s in items:
                by_session[s.session].append(s)
            pools = list(by_session.values())
            for pool in pools:
                rng.shuffle(pool)
            rng.shuffle(pools)
            picked: list[Sample] = []
            while len(picked) < cap:
                for pool in pools:
                    if pool and len(picked) < cap:
                        picked.append(pool.pop())
            items = picked
        out.extend(items)
    return out


def summarise(samples: list[Sample]) -> str:
    rows = []
    for split in ("train", "val", "test"):
        part = [s for s in samples if s.split == split]
        counts = Counter(s.label for s in part)
        sessions = len({s.session for s in part})
        per_class = "  ".join(f"{CLASSES[c]}={counts.get(c, 0)}" for c in range(len(CLASSES)))
        rows.append(f"{split:>5}: n={len(part):>7} sessions={sessions:>5} | {per_class}")
    return "\n".join(rows)


def write_manifest(samples: list[Sample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class", "session", "split"])
        for s in samples:
            w.writerow([s.path, s.label, CLASSES[s.label], s.session, s.split])


def read_manifest(path: str | Path) -> list[Sample]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [
            Sample(r["path"], int(r["label"]), None, int(r["session"]), r["split"])
            for r in csv.DictReader(f)
        ]


def split_arrays(samples: list[Sample], split: str) -> tuple[list[str], np.ndarray]:
    part = [s for s in samples if s.split == split]
    return [s.path for s in part], np.array([s.label for s in part], dtype=np.int32)


# --------------------------------------------------------------------------- tf.data

def load_image(path, size: int = IMG_SIZE):
    """Decode to float32 RGB in [0, 255]; the model does its own normalisation."""
    import tensorflow as tf

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(tf.cast(img, tf.float32), (size, size), antialias=True)


def make_dataset(
    paths: list[str],
    labels: np.ndarray,
    num_classes: int,
    batch_size: int,
    training: bool,
    img_size: int = IMG_SIZE,
    seed: int = 42,
):
    import tensorflow as tf

    from .augment import augment

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 50_000), seed=seed, reshuffle_each_iteration=True)

    # Train decodes slightly larger so augment() can random-crop back to img_size.
    load_size = int(img_size * 1.14) if training else img_size

    def _load(p, y):
        return load_image(p, load_size), tf.one_hot(y, num_classes)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(x, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
