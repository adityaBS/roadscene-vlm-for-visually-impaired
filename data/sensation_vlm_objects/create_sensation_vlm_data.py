#!/usr/bin/env python3
"""
Creates SENSATION VLM training, Testing and validation data.

Builds:
- SFT JSONL:  {image, prompt, response, sidewalk_pos, road_pos, road_vs_sidewalk}
- DPO JSONL:  {image, prompt, chosen, rejected, sidewalk_pos, road_pos, road_vs_sidewalk}
- merged CSV for debugging

Mandatory in every response:
Sidewalk: <left|center|right|unknown>.
Road: <left|center|right|front|unknown>.
RoadVsSidewalk: <left_of|right_of|crossing_front|unknown>.

Usage:
python create_sensation_vlm_data.py \
  --captions_csv complete_captions_sensation.csv \
  --annotations_csv annotations_with_obj_pos.csv \
  --out_dir bvpi_out
"""


import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Adjust these if your annotations CSV uses different names
NEAR_COLS = [
    "m_obstacle",
    "m_person",
    "m_bicycle",
    "m_car",
    "m_traffic_sign",
    "traffic_light",
]

FAR_COLS = [
    "m_obstacle_outside",
    "m_person_outside",
    "m_bicycle_outside",
    "m_car_outside",
    "m_traffic_sign_outside",
    "traffic_light_outside",
]

# Map column -> (singular, plural)
OBJ_LABELS = {
    "m_obstacle": ("obstacle", "obstacles"),
    "m_person": ("person", "people"),
    "m_bicycle": ("bicycle", "bicycles"),
    "m_car": ("car", "cars"),
    "m_traffic_sign": ("traffic sign", "traffic signs"),
    "traffic_light": ("traffic light", "traffic lights"),

    "m_obstacle_outside": ("obstacle", "obstacles"),
    "m_person_outside": ("person", "people"),
    "m_bicycle_outside": ("bicycle", "bicycles"),
    "m_car_outside": ("car", "cars"),
    "m_traffic_sign_outside": ("traffic sign", "traffic signs"),
    "traffic_light_outside": ("traffic light", "traffic lights"),
}

POS_ORDER = {"Left": 0, "Center": 1, "Right": 2}

DEFAULT_PROMPT = (
    "Describe the scene for a blind pedestrian. "
    "You MUST always output: Sidewalk position and Road position (relative to the image). "
    "Then output RoadVsSidewalk (is the road left_of/right_of the sidewalk, or crossing_front if the road is in front). "
    "After that, always mention CLOSE objects first with left/center/right. "
    "If no close objects are visible, explicitly say so. "
    "Only mention objects you can see."
)


def _article_for(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def norm_pos(v) -> str:
    """Normalize sidewalk_pos / road_pos values to a closed set."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "not present"}:
        return "unknown"
    if s == "centre":
        s = "center"
    # allowed values: left|center|right|front|unknown
    if s not in {"left", "center", "right", "front", "unknown"}:
        # Keep unknown if unexpected
        return "unknown"
    return s


def road_vs_sidewalk(sidewalk_pos: str, road_pos: str) -> str:
    """Infer a coarse relation: left_of, right_of, crossing_front, unknown."""
    sw = norm_pos(sidewalk_pos)
    rd = norm_pos(road_pos)

    if rd == "front":
        return "crossing_front"

    lr = {"left", "center", "right"}
    if sw in lr and rd in lr:
        if sw == rd:
            return "unknown"
        # interpret: where is road relative to sidewalk?
        # e.g., sidewalk=left, road=right -> road is right_of sidewalk
        if sw == "left" and rd in {"center", "right"}:
            return "right_of"
        if sw == "right" and rd in {"center", "left"}:
            return "left_of"
        if sw == "center" and rd == "left":
            return "left_of"
        if sw == "center" and rd == "right":
            return "right_of"

    return "unknown"


def _parse_positions(val) -> list[str]:
    """
    Parse position fields like:
      Left
      Center-Left-Right
      Left,Right
      Center Left Right
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "not present":
        return []
    parts = [p for p in re.split(r"[-,/ ]+", s) if p]
    tokens = [p.capitalize() for p in parts if p.capitalize() in POS_ORDER]
    return tokens


def _pos_phrase(tokens: list[str]) -> tuple[int, str]:
    count = len(tokens)
    uniq = sorted(set(tokens), key=lambda x: POS_ORDER[x])
    if len(uniq) == 1:
        where = f"on the {uniq[0].lower()}"
    elif len(uniq) == 2:
        where = f"on the {uniq[0].lower()} and {uniq[1].lower()}"
    else:
        where = "across the left, center, and right"
    return count, where


def _item_phrase(col: str, val, near: bool) -> str | None:
    tokens = _parse_positions(val)
    if not tokens:
        return None

    count, where = _pos_phrase(tokens)
    sing, plur = OBJ_LABELS.get(col, (col, col + "s"))

    if count == 1:
        quant = _article_for(sing)
        name = sing
    else:
        quant = str(count)
        name = plur

    dist = "close" if near else "farther away"
    return f"{quant} {name} {dist} {where}"


def build_priority_caption(row: pd.Series) -> str:
    # Mandatory positions
    sidewalk = norm_pos(row.get("sidewalk_pos"))
    road = norm_pos(row.get("road_pos"))
    rel = road_vs_sidewalk(sidewalk, road)

    header = (
        f"Sidewalk: {sidewalk}. "
        f"Road: {road}. "
        f"RoadVsSidewalk: {rel}."
    )

    # Nearby / Far objects
    near_items = []
    for c in NEAR_COLS:
        phr = _item_phrase(c, row.get(c), near=True)
        if phr:
            near_items.append(phr)

    far_items = []
    for c in FAR_COLS:
        phr = _item_phrase(c, row.get(c), near=False)
        if phr:
            far_items.append(phr)

    if near_items:
        near_part = "Nearby: " + "; ".join(near_items) + "."
    else:
        near_part = "Nearby: no important close objects visible."

    parts = [header, near_part]

    if far_items:
        parts.append("Far: " + "; ".join(far_items) + ".")

    orig = str(row.get("target", "")).strip()
    if orig:
        parts.append(f"Context: {orig}")

    return " ".join(parts).strip()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_csv", required=True, help="CSV with image_path,target,split")
    ap.add_argument("--annotations_csv", required=True, help="CSV with image + m_* columns + sidewalk_pos,road_pos")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt for SFT/DPO")
    ap.add_argument("--require_all_splits", action="store_true",
                    help="If set, error if any split is missing (train/val/test).")
    args = ap.parse_args()

    captions = pd.read_csv(args.captions_csv)
    ann = pd.read_csv(args.annotations_csv)

    # Check expected columns
    for col in ["image_path", "target", "split"]:
        if col not in captions.columns:
            raise ValueError(f"captions_csv missing column: {col}")
    if "image" not in ann.columns:
        raise ValueError("annotations_csv missing column: image")

    # Add basename keys for merge
    captions["image_base"] = captions["image_path"].astype(str).apply(os.path.basename)
    ann["image_base"] = ann["image"].astype(str).apply(os.path.basename)

    ann = ann.drop_duplicates(subset=["image_base"], keep="first")

    merged = captions.merge(
        ann.drop(columns=["image"]),
        on="image_base",
        how="left"
    )

    # Build priority targets
    merged["priority_caption"] = merged.apply(build_priority_caption, axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_dir / "merged_captions_with_positions.csv", index=False)

    # Split checks
    splits_found = set(merged["split"].astype(str).unique())
    needed = {"train", "val", "test"}
    if args.require_all_splits and not needed.issubset(splits_found):
        raise ValueError(f"Missing splits. Found: {sorted(splits_found)} needed: {sorted(needed)}")

    # Write SFT JSONL per split
    for split in ["train", "val", "test"]:
        df = merged[merged["split"] == split]
        if df.empty:
            continue

        rows = []
        for _, r in df.iterrows():
            sidewalk = norm_pos(r.get("sidewalk_pos"))
            road = norm_pos(r.get("road_pos"))
            rel = road_vs_sidewalk(sidewalk, road)
            rows.append(
                {
                    "image": r["image_path"],  # keep exactly as in captions CSV
                    "prompt": args.prompt,
                    "response": r["priority_caption"],
                    "sidewalk_pos": sidewalk,
                    "road_pos": road,
                    "road_vs_sidewalk": rel,
                }
            )
        write_jsonl(out_dir / f"bvpi_sft_{split}.jsonl", rows)

    # Write DPO pairs (train/val)
    for split in ["train", "val"]:
        df = merged[merged["split"] == split]
        if df.empty:
            continue

        rows = []
        for _, r in df.iterrows():
            rejected = str(r.get("target", "")).strip()
            chosen = str(r.get("priority_caption", "")).strip()
            if not rejected:
                continue
            if chosen == rejected:
                continue

            sidewalk = norm_pos(r.get("sidewalk_pos"))
            road = norm_pos(r.get("road_pos"))
            rel = road_vs_sidewalk(sidewalk, road)

            rows.append(
                {
                    "image": r["image_path"],
                    "prompt": args.prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "sidewalk_pos": sidewalk,
                    "road_pos": road,
                    "road_vs_sidewalk": rel,
                }
            )
        write_jsonl(out_dir / f"bvpi_dpo_{split}.jsonl", rows)

    print("Done. Wrote outputs to:", out_dir.resolve())
    print("Files:")
    for name in [
        "bvpi_sft_train.jsonl",
        "bvpi_sft_val.jsonl",
        "bvpi_sft_test.jsonl",
        "bvpi_dpo_train.jsonl",
        "bvpi_dpo_val.jsonl",
        "merged_captions_with_positions.csv",
    ]:
        p = out_dir / name
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()
