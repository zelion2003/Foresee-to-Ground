#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
from collections import Counter

import cv2


def build_path_candidates(root: str, vid_name: str):
    
    cands = [os.path.join(root, vid_name)]
    if "videos_1FPS_plain" in root:
        cands.append(os.path.join(root.replace("videos_1FPS_plain", "videos_1FPS"), vid_name))
        cands.append(os.path.join(root.replace("videos_1FPS_plain", "videos_1FPS_number_red_40_br"), vid_name))
    elif "videos_1FPS" in root:
        cands.append(os.path.join(root.replace("videos_1FPS", "videos_1FPS_plain"), vid_name))
        cands.append(os.path.join(root.replace("videos_1FPS", "videos_1FPS_number_red_40_br"), vid_name))
    return cands


def main():
    ap = argparse.ArgumentParser("Validate the public SFT dataset against the available video roots.")
    ap.add_argument("--data_path", required=True, help="Path to the training JSON file.")
    ap.add_argument("--anet_root", required=True, help="ActivityNet video root.")
    ap.add_argument("--didemo_root", required=True, help="DiDeMo video root.")
    ap.add_argument("--internvid_root", required=True, help="InternVid video root.")
    ap.add_argument("--default_root", default="", help="Fallback video root for unknown sources.")
    ap.add_argument("--min_frames", type=int, default=4, help="Minimum valid frame count.")
    ap.add_argument("--max_print", type=int, default=20, help="Maximum number of missing or short examples to print.")
    args = ap.parse_args()

    roots = {
        "anet": args.anet_root,
        "didemo": args.didemo_root,
        "internvid": args.internvid_root,
    }
    total = Counter()
    kept = Counter()
    missing_list = []
    short_list = []

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        src = item.get("source", "")
        vid = item.get("video", "")
        total[src] += 1

        root = roots.get(src, args.default_root)
        if not os.path.splitext(vid)[1]:
            vid = vid + ".mp4"

        full_path = None
        for p in build_path_candidates(root, vid):
            if p and os.path.exists(p):
                full_path = p
                break
        if full_path is None:
            if len(missing_list) < args.max_print:
                missing_list.append((src, vid))
            continue

        cap = cv2.VideoCapture(full_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if frames < args.min_frames:
            if len(short_list) < args.max_print:
                short_list.append((src, vid, frames))
            continue

        kept[src] += 1

    print("===== Dataset Check Summary =====")
    for k in sorted(total.keys()):
        print(f"{k:8s} total={total[k]:7d} kept={kept[k]:7d} missing/short={total[k]-kept[k]:7d}")
    print(f"overall: total={sum(total.values())} kept={sum(kept.values())} missing_or_short={sum(total.values())-sum(kept.values())}")

    if missing_list:
        print(f"\nMissing examples (up to {args.max_print}):")
        for src, vid in missing_list:
            print(f"  {src:8s} {vid}")
    if short_list:
        print(f"\nShort examples (up to {args.max_print}, min_frames={args.min_frames}):")
        for src, vid, fr in short_list:
            print(f"  {src:8s} {vid} frames={fr}")


if __name__ == "__main__":
    main()
