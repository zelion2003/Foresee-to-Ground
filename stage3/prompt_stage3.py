#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple

import torch


def render_prompt_b(s_topk: torch.Tensor, e_topk: torch.Tensor, k: int = 8, m: int = 2) -> List[str]:
    
    B, K = s_topk.shape
    lines_batch: List[str] = []
    for b in range(B):
        lines = [
            f"Here are {K} candidate event spans extracted from the video.\n"
            f"Each candidate provides (1) its time range and (2) {m} visual span tokens.\n"
            "You MUST cite exactly one span id token at the end of your answer.\n"
        ]
        for i in range(K):
            pads = " ".join(["<span_pad>"] * m)
            lines.append(
                f"Candidate {i + 1}: from {s_topk[b,i].item():.1f} seconds to {e_topk[b,i].item():.1f} seconds  "
                f"<Span_{i}> <|vision_start|>{pads}<|vision_end|>"
            )
        lines_batch.append("\n".join(lines))
    return lines_batch


def find_subseq(haystack: List[int], needle: List[int]) -> int | None:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def find_insert_pos(input_ids: List[int], tokenizer) -> int:
    
    q_ids = tokenizer.encode("Question:", add_special_tokens=False)
    pos = find_subseq(input_ids, q_ids)
    if pos is not None:
        return pos
    ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    ve_pos = max(i for i, t in enumerate(input_ids) if t == ve_id)
    return ve_pos + 1


def mask_labels_before_assistant(labels: torch.Tensor, tokenizer) -> torch.Tensor:
    
    B, L = labels.shape
    prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    ids0 = labels[0].tolist()
    pos = find_subseq(ids0, prefix)
    if pos is None:
        return labels
    start = pos + len(prefix)
    labels[:, :start] = -100
    return labels
