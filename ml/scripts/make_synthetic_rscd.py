"""Generate a tiny fake RSCD-style tree for offline smoke tests of the pipeline.

Layout mirrors the real dataset:
    <out>/train/dry_asphalt_slight/202201270008469-dry-asphalt-slight.jpg
Each "drive" is a burst of frames one second apart; drives are hours apart,
so the session splitter has real groups to work with.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

RSCD_FOLDERS = {
    "dry_asphalt_smooth": (90, 90, 95),
    "wet_concrete_slight": (60, 65, 75),
    "water_asphalt_smooth": (40, 60, 90),
    "fresh_snow": (235, 235, 240),
    "melted_snow": (150, 140, 130),
    "ice": (170, 200, 220),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--drives", type=int, default=12)
    ap.add_argument("--frames-per-drive", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(seed=args.seed)
    t0 = datetime(2022, 1, 1, 8, 0, 0)
    n = 0
    for d in range(args.drives):
        start = t0 + timedelta(hours=3 * d)
        for k, (folder, rgb) in enumerate(RSCD_FOLDERS.items()):
            for f in range(args.frames_per_drive):
                ts = start + timedelta(seconds=k * args.frames_per_drive + f)
                img = np.clip(np.array(rgb) + rng.normal(0, 25, (240, 360, 3)), 0, 255).astype(np.uint8)
                name = f"{ts:%Y%m%d%H%M%S}{f % 10}-{folder.replace('_', '-')}.jpg"
                path = Path(args.out) / "train" / folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(path, quality=85)
                n += 1
    print(f"wrote {n} images to {args.out}")


if __name__ == "__main__":
    main()
