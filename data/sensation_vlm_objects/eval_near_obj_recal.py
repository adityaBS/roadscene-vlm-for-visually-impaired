#!/usr/bin/env python3
"""
Check if export was successfully for objects localization.

Evaluates:
1) Mandatory field presence in model outputs:
   - Sidewalk:
   - Road:
   - RoadVsSidewalk:

2) Field accuracy if you provide ground-truth via annotations CSV:
   - sidewalk_pos accuracy
   - road_pos accuracy
   - road_vs_sidewalk accuracy

3) Near-object mention recall/precision (keyword-based proxy):
   - person, car, bicycle, obstacle, traffic sign, traffic light
   (Ground truth is "m_*" columns in the annotations CSV.)

Input predictions JSONL must contain per line:
- {"image": "...", "text": "..."}  OR {"image": "...", "response": "..."}

Usage:
python near_object_and_position_eval.py \
  --pred_jsonl preds.jsonl \
  --captions_csv complete_captions_sensation.csv \
  --annotations_csv annotations_with_obj_pos.csv
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd


NEAR_GT_COLS = {
    "m_person": ["person", "pedestrian", "man", "woman", "people"],
    "m_car": ["car", "vehicle", "auto"],
    "m_bicycle": ["bicycle", "bike", "cyclist"],
    "m_obstacle": ["obstacle", "barrier", "cone", "bollard", "block"],
    "m_traffic_sign": ["traffic sign", "sign"],
    "traffic_light": ["traffic light", "trafficlight", "signal"],
}

POS_SET = {"left", "center", "right", "front", "unknown"}
REL_SET = {"left_of", "right_of", "crossing_front", "unknown"}


def norm_pos(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "not present"}:
        return "unknown"
    if s == "centre":
        s = "center"
    if s not in POS_SET:
        return "unknown"
    return s


def road_vs_sidewalk(sidewalk_pos: str, road_pos: str) -> str:
    sw = norm_pos(sidewalk_pos)
    rd = norm_pos(road_pos)
    if rd == "front":
        return "crossing_front"
    lr = {"left", "center", "right"}
    if sw in lr and rd in lr:
        if sw == rd:
            return "unknown"
        if sw == "left" and rd in {"center", "right"}:
            return "right_of"
        if sw == "right" and rd in {"center", "left"}:
            return "left_of"
        if sw == "center" and rd == "left":
            return "left_of"
        if sw == "center" and rd == "right":
            return "right_of"
    return "unknown"


def is_present(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    s = str(v).strip().lower()
    return s not in {"", "nan", "none", "not present"}


FIELD_RE = {
    "sidewalk_pos": re.compile(r"\bsidewalk\s*:\s*(left|center|right|front|unknown)\b", re.IGNORECASE),
    "road_pos": re.compile(r"\broad\s*:\s*(left|center|right|front|unknown)\b", re.IGNORECASE),
    "road_vs_sidewalk": re.compile(r"\broadvssidewalk\s*:\s*(left_of|right_of|crossing_front|unknown)\b", re.IGNORECASE),
}


def parse_fields(text: str) -> dict:
    t = (text or "").strip()
    out = {}
    for k, rx in FIELD_RE.items():
        m = rx.search(t)
        out[k] = m.group(1).lower() if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help='JSONL with {"image":..., "text":...} or {"image":..., "response":...}')
    ap.add_argument("--captions_csv", required=True, help="CSV with image_path,target,split")
    ap.add_argument("--annotations_csv", required=True, help="CSV with image + m_* + sidewalk_pos,road_pos")
    ap.add_argument("--split", default="test", help="Evaluate only one split (default: test)")
    args = ap.parse_args()

    captions = pd.read_csv(args.captions_csv)
    ann = pd.read_csv(args.annotations_csv)

    captions["image_base"] = captions["image_path"].astype(str).apply(os.path.basename)
    ann["image_base"] = ann["image"].astype(str).apply(os.path.basename)
    ann = ann.drop_duplicates(subset=["image_base"], keep="first")

    # Read predictions
    preds = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            img = os.path.basename(r.get("image", ""))
            text = r.get("text") or r.get("response") or ""
            preds.append((img, text))
    pred_df = pd.DataFrame(preds, columns=["image_base", "pred_text"])

    # Filter to chosen split (based on captions CSV)
    split_df = captions[captions["split"] == args.split][["image_base"]].drop_duplicates()
    df = split_df.merge(pred_df, on="image_base", how="left").merge(ann, on="image_base", how="left")

    # Mandatory field presence + accuracy
    presence = {k: 0 for k in FIELD_RE.keys()}
    total = len(df)

    acc_counts = {"sidewalk_pos": 0, "road_pos": 0, "road_vs_sidewalk": 0}
    acc_denoms = {"sidewalk_pos": 0, "road_pos": 0, "road_vs_sidewalk": 0}

    # Near-object mention stats
    obj_rows = []
    for obj_col, kws in NEAR_GT_COLS.items():
        gt_present = df[obj_col].apply(is_present) if obj_col in df.columns else pd.Series([False] * total)
        pred_mention = df["pred_text"].fillna("").str.lower().apply(lambda t: any(k in t for k in kws))

        tp = int((gt_present & pred_mention).sum())
        fn = int((gt_present & (~pred_mention)).sum())
        fp = int(((~gt_present) & pred_mention).sum())

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        obj_rows.append((obj_col, int(gt_present.sum()), tp, fn, fp, recall, precision))

    # Field parsing per row
    for _, r in df.iterrows():
        text = r.get("pred_text") or ""
        fields = parse_fields(text)

        for k in presence.keys():
            if fields[k] is not None:
                presence[k] += 1

        # Accuracy against GT
        gt_sw = norm_pos(r.get("sidewalk_pos"))
        gt_rd = norm_pos(r.get("road_pos"))
        gt_rel = road_vs_sidewalk(gt_sw, gt_rd)

        pred_sw = fields["sidewalk_pos"]
        pred_rd = fields["road_pos"]
        pred_rel = fields["road_vs_sidewalk"]

        # Always count denom (we want these always)
        acc_denoms["sidewalk_pos"] += 1
        acc_denoms["road_pos"] += 1
        acc_denoms["road_vs_sidewalk"] += 1

        if pred_sw == gt_sw:
            acc_counts["sidewalk_pos"] += 1
        if pred_rd == gt_rd:
            acc_counts["road_pos"] += 1
        if pred_rel == gt_rel:
            acc_counts["road_vs_sidewalk"] += 1

    print(f"Split: {args.split} | N={total}")

    print("\nMandatory field presence rate:")
    for k, v in presence.items():
        print(f" - {k}: {v}/{total} = {v/total:.3f}")

    print("\nField accuracy (exact match to GT, using annotations CSV):")
    for k in acc_counts.keys():
        denom = acc_denoms[k]
        val = acc_counts[k]
        print(f" - {k}: {val}/{denom} = {val/denom:.3f}")

    print("\nNear-object mention (keyword proxy):")
    out = pd.DataFrame(
        obj_rows,
        columns=["object_col", "gt_present", "tp", "fn", "fp", "recall", "precision"]
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
