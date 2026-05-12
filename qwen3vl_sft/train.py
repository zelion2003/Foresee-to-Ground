#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
try:
    from torch.utils import _pytree as _torch_pytree  # type: ignore
    if not hasattr(torch.utils, "register_pytree_node"):
        def register_pytree_node(node_type, flatten_fn, unflatten_fn, serialized_type_name=None, *args, **kwargs):
            return _torch_pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)
        torch.utils.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
        _torch_pytree.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
except Exception:
    pass

if not hasattr(torch, "compiler"):
    class _DummyCompiler:
        @staticmethod
        def is_compiling():
            return False
    torch.compiler = _DummyCompiler()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
# ---------------------------------------------------

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
import swanlab
import os
try:
    from torch.utils import _pytree as _torch_pytree  # type: ignore
    if not hasattr(torch.utils, "register_pytree_node"):
        def register_pytree_node(node_type, flatten_fn, unflatten_fn, serialized_type_name=None, *args, **kwargs):
            return _torch_pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)
        torch.utils.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
        _torch_pytree.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
except Exception:
    pass

if not hasattr(torch, "compiler"):
    class _DummyCompiler:
        @staticmethod
        def is_compiling():
            return False
    torch.compiler = _DummyCompiler()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
# ---------------------------------------------------


def strip_image_token(text: str) -> str:
    
    return text.replace("<image>", "").strip()


def build_messages(video_path: str, question: str, answer: str):
    
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 1},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


class SingleTurnVTGDataset(Dataset):
    

    def __init__(self, json_path: str, roots: Dict[str, str], default_root: str):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.roots = roots
        self.default_root = default_root
        self.samples = []
        missing = 0
        too_short = 0

        for item in raw:
            src = item.get("source", "")
            vid_name = item.get("video", "")
            root = self.roots.get(src, self.default_root)
            if not root:
                missing += 1
                continue
            if not os.path.splitext(vid_name)[1]:
                vid_name = vid_name + ".mp4"
            candidates = [os.path.join(root, vid_name)]
            if "videos_1FPS_plain" in root:
                candidates.append(os.path.join(root.replace("videos_1FPS_plain", "videos_1FPS"), vid_name))
                candidates.append(os.path.join(root.replace("videos_1FPS_plain", "videos_1FPS_number_red_40_br"), vid_name))
            elif "videos_1FPS" in root:
                candidates.append(os.path.join(root.replace("videos_1FPS", "videos_1FPS_plain"), vid_name))

            full_path = None
            for p in candidates:
                if os.path.exists(p):
                    full_path = p
                    break
            if full_path is None:
                missing += 1
                continue
            try:
                import cv2

                cap = cv2.VideoCapture(full_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                cap.release()
            except Exception:
                total = 0
            if total < 4:
                too_short += 1
                continue
            item["_video_path"] = full_path
            self.samples.append(item)

        if missing:
            print(f"[WARN] filtered out {missing} samples with missing video/root")
        if too_short:
            print(f"[WARN] filtered out {too_short} samples with <4 frames")

    def __len__(self):
        return len(self.samples)

    def resolve_video_path(self, source: str, video_name: str) -> str:
        root = self.roots.get(source, self.default_root)
        if not os.path.splitext(video_name)[1]:
            video_name = video_name + ".mp4"
        return os.path.join(root, video_name)

    def __getitem__(self, idx):
        item = self.samples[idx]
        source = item.get("source", "")
        video_path = item.get("_video_path")
        if not video_path:
            video_name = item.get("video", "")
            video_path = self.resolve_video_path(source, video_name)
        question = strip_image_token(item.get("question", ""))
        answer = item.get("answer", "")

        messages_full = build_messages(video_path, question, answer)
        messages_prefix = messages_full[:1]
        return {
            "id": item.get("id"),
            "video": video_path,
            "messages_full": messages_full,
            "messages_prefix": messages_prefix,
        }


@dataclass
class DataCollator:
    processor: AutoProcessor
    num_frames: int
    fps: float
    do_sample_frames: bool
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_full = [
            self.processor.apply_chat_template(f["messages_full"], tokenize=False, add_generation_prompt=False)
            for f in features
        ]
        texts_prefix = [
            self.processor.apply_chat_template(f["messages_prefix"], tokenize=False, add_generation_prompt=True)
            for f in features
        ]
        videos = [f["video"] for f in features]
        proc_kwargs = dict(
            text=texts_full,
            videos=videos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            do_sample_frames=self.do_sample_frames,
        )
        if self.num_frames is not None:
            proc_kwargs["num_frames"] = self.num_frames
        else:
            proc_kwargs["fps"] = self.fps

        batch = self.processor(**proc_kwargs)
        labels = batch["input_ids"].clone()
        for i, prefix in enumerate(texts_prefix):
            prefix_ids = self.processor.tokenizer(
                prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            prefix_len = min(len(prefix_ids), labels.shape[1])
            labels[i, :prefix_len] = -100
        batch["labels"] = labels
        return batch


def freeze_non_lora_params(model: torch.nn.Module):
    
    for name, param in model.named_parameters():
        is_lora = ("lora_" in name) or ("lora_A" in name) or ("lora_B" in name)
        if is_lora:
            param.requires_grad = True
        else:
            param.requires_grad = False


class SwanLabCallback(TrainerCallback):
    

    def __init__(self, project: str, config: Dict[str, Any], run_name: str = None):
        self.enabled = int(os.environ.get("LOCAL_RANK", "0")) == 0
        if self.enabled:
            swanlab.init(project=project, run_name=run_name or "qwen3vl_sft", config=config)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.enabled and logs:
            to_log = dict(logs)
            to_log["step"] = state.global_step
            if state.max_steps is not None and state.max_steps > 0:
                to_log["max_steps"] = state.max_steps
                to_log["progress_pct"] = state.global_step / state.max_steps * 100.0
            if state.epoch is not None:
                to_log["epoch"] = float(state.epoch)
            swanlab.log(to_log, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.enabled:
            swanlab.finish()


def main():
    ap = argparse.ArgumentParser("Qwen3-VL LoRA SFT (multi-dataset, 1FPS, single-turn VTG)")
    ap.add_argument("--data_path", required=True, help="Path to the single-turn VTG training JSON file.")
    ap.add_argument("--anet_root", required=True, help="ActivityNet 1 FPS video root.")
    ap.add_argument("--didemo_root", required=True, help="DiDeMo 1 FPS video root.")
    ap.add_argument("--internvid_root", required=True, help="InternVid 1 FPS video root.")
    ap.add_argument("--default_root", default="", help="Fallback video root for unknown sources.")
    ap.add_argument("--output_dir", required=True, help="Directory used to store checkpoints and trainer outputs.")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--num_frames", type=int, default=None, help="Optional fixed number of sampled frames.")
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling FPS when num_frames is not set.")
    ap.add_argument("--max_frames", type=int, default=48, help="Maximum number of video frames processed by the video processor.")
    ap.add_argument("--no_sample_frames", action="store_true", help="Disable frame sampling and use the processor defaults instead.")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=5000)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--deepspeed", default=None, help="Path to a DeepSpeed configuration file, for example scripts/zero2.json.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    roots = {
        "anet": args.anet_root,
        "didemo": args.didemo_root,
        "internvid": args.internvid_root,
    }
    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = args.fps
        processor.video_processor.max_frames = args.max_frames
        processor.video_processor.do_sample_frames = not args.no_sample_frames

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    freeze_non_lora_params(model)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        print("[WARN] model has no `enable_input_require_grads` method, please check transformers version.")
    if hasattr(model, "config"):
        model.config.use_cache = False

    # ==== DEBUG: print trainable params ====
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[DEBUG] #trainable tensors: {len(trainable)}")
    total_trainable = sum(p.numel() for _, p in trainable)
    print(f"[DEBUG] total trainable params: {total_trainable}")
    print("[DEBUG] first 20 trainable tensors:")
    for n, p in trainable[:20]:
        print("  ", n, p.shape, p.numel())
    # ==== END DEBUG ====

    ds = SingleTurnVTGDataset(args.data_path, roots=roots, default_root=args.default_root or args.anet_root)
    collator = DataCollator(
        processor=processor,
        num_frames=args.num_frames,
        fps=args.fps,
        do_sample_frames=not args.no_sample_frames,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
    )
    sw_config = vars(args).copy()
    sw_callback = SwanLabCallback(
        project="qwen3-vl-8B_VTG-SFT",
        config=sw_config,
        run_name="178k-data",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=processor.tokenizer,
        data_collator=collator,
        callbacks=[sw_callback],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
