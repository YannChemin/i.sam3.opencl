#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Compare i.sam3.opencl text-prompt debug dumps with the PyTorch sam3 code.

Run i.sam3.opencl with prompt= (one crop, minsize=0) and the environment
variable I_SAM3_OPENCL_DUMP set to a directory; it then writes the 8-bit
crop, the normalized encoder input, the 72 x 72 detector features, the
first prompt's text features and the final mask list (masks.csv). This
script runs Sam3Processor.set_text_prompt on the same pixels and reports:

1. the maximum difference of the encoder input,
2. the relative errors of the detector and text features,
3. how the detections match (mask box IoU pairing), with their scores.

Requires torch, torchvision and sam3 (not needed by the module itself).

Usage:
    reference_compare_text.py DUMP_DIR CHECKPOINT PROMPT [PROMPT ...]
        [--threshold 0.5]
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor  # ty: ignore[unresolved-import]
from sam3.model_builder import build_sam3_image_model  # ty: ignore[unresolved-import]
from torchvision.transforms import v2


def cpu_only():
    """Map sam3's hard-coded CUDA tensors to the CPU when there is no GPU.

    Several sam3 constructors create buffers with device="cuda" whatever
    the model device (PositionEmbeddingSine, TransformerDecoder, ...).
    """
    if torch.cuda.is_available():
        return

    def on_cpu(factory):
        def wrapped(*args, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return wrapped

    for name in (
        "arange",
        "empty",
        "full",
        "linspace",
        "ones",
        "rand",
        "randn",
        "tensor",
        "zeros",
    ):
        setattr(torch, name, on_cpu(getattr(torch, name)))
    # Pinned host memory needs an accelerator.
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self  # ty: ignore[invalid-assignment]


def load_dump(dump_dir, name, shape):
    path = os.path.join(dump_dir, f"crop000_{name}.f32")
    return np.fromfile(path, np.float32).reshape(shape)


def compare(what, ref, ours):
    rel = np.linalg.norm(ref - ours) / np.linalg.norm(ref)
    print(
        f"{what}: relative L2 error {rel:.3g}, "
        f"max abs difference {np.abs(ref - ours).max():.3g}"
    )


def box_iou(a, b):
    """IoU of inclusive xyxy boxes, as torchvision computes it."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rgb_path = min(glob.glob(os.path.join(args.dump_dir, "crop000_rgb_*.u8")))
    size = re.search(r"_(\d+)x(\d+)\.u8$", rgb_path)
    if size is None:
        raise ValueError(f"no crop size in the dump name {rgb_path}")
    h, w = map(int, size.groups())
    rgb = np.fromfile(rgb_path, dtype=np.uint8).reshape(h, w, 3)
    inp = load_dump(args.dump_dir, "input", (3, 1008, 1008))
    det = load_dump(args.dump_dir, "detector", (72, 72, 256))
    txt = np.fromfile(os.path.join(args.dump_dir, "crop000_text.f32"), np.float32)
    txt = txt.reshape(-1, 256)

    cpu_only()
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, load_from_HF=False, device="cpu"
    )
    proc = Sam3Processor(model, device="cpu", confidence_threshold=args.threshold)
    image = Image.fromarray(rgb)
    ours = np.loadtxt(
        os.path.join(args.dump_dir, "masks.csv"), delimiter=",", skiprows=1, ndmin=2
    )

    with torch.inference_mode():
        ref_inp = proc.transform(v2.functional.to_image(image)).numpy()
        print(f"encoder input: max abs difference {np.abs(ref_inp - inp).max():.3g}")
        state = proc.set_image(image)
        ref_det = state["backbone_out"]["backbone_fpn"][-1][0].permute(1, 2, 0)
        compare("detector features", ref_det.numpy(), det)
        text = model.backbone.forward_text([args.prompts[0]], device="cpu")
        ref_txt = text["language_features"][: len(txt), 0].numpy()
        compare("text features", ref_txt, txt)

        for k, prompt in enumerate(args.prompts):
            out = proc.set_text_prompt(prompt, state)
            masks = out["masks"][:, 0].numpy()
            scores = out["scores"].numpy()
            ref = []
            for mask, score in zip(masks, scores):
                ys, xs = np.nonzero(mask)
                if len(xs):
                    ref.append(([xs.min(), ys.min(), xs.max(), ys.max()], float(score)))
            mine = ours[ours[:, 7] == k] if len(ours) else ours
            print(
                f"prompt <{prompt}>: reference {len(ref)} detections, "
                f"i.sam3.opencl {len(mine)}"
            )
            matched = 0
            for row in mine:
                ious = [box_iou(row[:4], r[0]) for r in ref]
                best = int(np.argmax(ious)) if ious else -1
                if ious and ious[best] > 0.9:
                    matched += 1
                    print(
                        f"  box {row[:4].astype(int).tolist()}: IoU "
                        f"{ious[best]:.3f}, score {row[5]:.4f} vs "
                        f"{ref[best][1]:.4f}"
                    )
                else:
                    print(f"  box {row[:4].astype(int).tolist()}: unmatched")
            print(f"  matched {matched}/{len(mine)}")


if __name__ == "__main__":
    main()
