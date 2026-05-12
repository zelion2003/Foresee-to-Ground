#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
from pathlib import Path
from glob import glob

import torch
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

import torchvision


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser("Filter videos by decodability and fps/duration")
    ap.add_argument("--video_dir", default=str(REPO_ROOT / "data" / "stage1" / "clips"), help="Directory containing Stage-1 clips.")
    ap.add_argument("--ok_dir", default=str(REPO_ROOT / "data" / "stage1" / "clips_ok"), help="Directory used to store accepted clips.")
    ap.add_argument("--fail_dir", default=str(REPO_ROOT / "data" / "stage1" / "clips_fail"), help="Directory used to store rejected clips.")
    ap.add_argument(
        "--exts",
        default=".mp4,.avi,.mov",
        help="Comma-separated extension list, including the leading dot.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ok_dir, exist_ok=True)
    os.makedirs(args.fail_dir, exist_ok=True)
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(args.video_dir, f"*{ext}")))
    paths = sorted(paths)

    if not paths:
        print(f"No videos found in {args.video_dir} with exts {exts}")
        return

    ok_list, fail_list = [], []

    for p in paths:
        try:
            reader = torchvision.io.VideoReader(p, "video")
            meta = reader.get_metadata().get("video", {})
            fps = meta.get("fps", [None])
            fps = fps[0] if isinstance(fps, (list, tuple)) else fps
            duration = meta.get("duration", [None])
            duration = duration[0] if isinstance(duration, (list, tuple)) else duration
            first_frame = next(reader, None)
            if first_frame is None:
                raise RuntimeError("no frames decoded")

            is_ok = (fps == 1.0) and (duration == 48)
            if is_ok:
                ok_list.append((p, fps, duration))
                target = os.path.join(args.ok_dir, os.path.basename(p))
                print(f"[OK]   {p} fps={fps} duration={duration} -> {target}")
                with open(p, "rb") as fsrc, open(target, "wb") as fdst:
                    fdst.write(fsrc.read())
            else:
                fail_list.append((p, f"fps={fps}, duration={duration}"))
                target = os.path.join(args.fail_dir, os.path.basename(p))
                print(f"[FAIL] {p} fps={fps} duration={duration} -> {target}")
                with open(p, "rb") as fsrc, open(target, "wb") as fdst:
                    fdst.write(fsrc.read())
        except Exception as e:
            fail_list.append((p, str(e)))
            target = os.path.join(args.fail_dir, os.path.basename(p))
            print(f"[FAIL] {p} error={e} -> {target}")
            try:
                with open(p, "rb") as fsrc, open(target, "wb") as fdst:
                    fdst.write(fsrc.read())
            except Exception as e2:
                print(f"[WARN] copy failed for {p}: {e2}")

    print("\n=== Summary ===")
    print(f"Total: {len(paths)}, OK: {len(ok_list)}, FAIL: {len(fail_list)}")


if __name__ == "__main__":
    main()
