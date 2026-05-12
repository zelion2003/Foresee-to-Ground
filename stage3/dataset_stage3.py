#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import re
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
SPAN_PATTERN = re.compile(
    r"from\s+([0-9]*\.?[0-9]+)\s*seconds?\s*to\s*([0-9]*\.?[0-9]+)\s*seconds?",
    re.IGNORECASE,
)


def extract_spans_from_answer(answer: str) -> List[Tuple[float, float]]:
    spans = []
    for m in SPAN_PATTERN.finditer(answer):
        s = float(m.group(1))
        e = float(m.group(2))
        if e > s:
            spans.append((s, e))
    return spans


class Stage3VTGDataset(Dataset):
    

    def __init__(
        self,
        json_path: str,
        video_roots: Dict[str, str],
        min_frames: int = 2,
        duration_eps: float = 1e-6,
    ):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples: List[Dict] = []
        miss_video = 0
        miss_span = 0
        short_video = 0

        for item in data:
            source = item.get("source", "")
            video = item.get("video", "")
            duration = float(item.get("duration", 0.0) or 0.0)
            question = item.get("question", "")
            answer = item.get("answer", "")

            if not source or not video or duration <= duration_eps:
                miss_span += 1
                continue
            root = video_roots.get(source)
            if root is None:
                miss_video += 1
                continue

            video_name = video if video.endswith(".mp4") else f"{video}.mp4"
            video_path = os.path.join(root, video_name)
            if not os.path.isfile(video_path):
                miss_video += 1
                continue
            if min_frames is not None and min_frames > 0:
                try:
                    import cv2

                    cap = cv2.VideoCapture(video_path)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                    cap.release()
                except Exception:
                    total = 0
                if total < min_frames:
                    short_video += 1
                    continue
            spans_raw = item.get("spans", [])
            spans = []
            if spans_raw:
                for s in spans_raw:
                    try:
                        st = float(s.get("start", 0.0))
                        ed = float(s.get("end", 0.0))
                    except Exception:
                        continue
                    if ed > st:
                        spans.append((st, ed))
            if not spans:
                spans = extract_spans_from_answer(answer)
            if not spans:
                miss_span += 1
                continue
            st, ed = spans[0]
            st_n = max(0.0, min(1.0, st / duration))
            ed_n = max(0.0, min(1.0, ed / duration))
            if ed_n <= st_n:
                miss_span += 1
                continue

            self.samples.append(
                {
                    "video_path": video_path,
                    "duration": duration,
                    "question": question,
                    "answer": answer,
                    "gt_span": torch.tensor([st_n, ed_n], dtype=torch.float32),
                    "source": source,
                    "video": video,
                    "id": item.get("id", ""),
                }
            )

        if not self.samples:
            raise ValueError("Stage3VTGDataset is empty after filtering. Check the annotation file and video roots.")

        print(
            f"[Stage3Dataset] kept={len(self.samples)}, "
            f"miss_video={miss_video}, miss_span={miss_span}, short_video={short_video}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
