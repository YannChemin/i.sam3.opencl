#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Compare i.sam3.opencl debug dumps with the PyTorch sam3 code.

Run i.sam3.opencl with the environment variable I_SAM3_OPENCL_DUMP set to
a directory; it then writes, for each crop, the 8-bit crop it segmented,
the normalized encoder input, the image embedding and the two
high-resolution feature maps, plus the final mask list (masks.csv). This
script runs the reference implementation on the same 8-bit pixels and
reports:

1. the maximum difference between the reference preprocessing
   (Sam3Processor.set_image) and the dumped encoder input,
2. the relative error of the image embedding and of the high-resolution
   features of the point-prompt head (Sam3Image.predict_inst),
3. how the automatically generated masks match (box IoU pairing). sam3
   has no automatic mask generator: the single-crop steps of
   SAM2AutomaticMaskGenerator (point grid, IoU and stability filtering,
   box NMS) are run here on SAM3InteractiveImagePredictor._predict.

Run i.sam3.opencl with minsize=0 and one crop for the mask comparison.

Requires torch, torchvision and sam3 (not needed by the module itself).

Usage:
    reference_compare.py DUMP_DIR CHECKPOINT [--points N] [--iou 0.8]
        [--stability 0.95]
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from torchvision.ops import batched_nms


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
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self


def load_dump(dump_dir, name, shape):
    path = os.path.join(dump_dir, f"crop000_{name}.f32")
    return np.fromfile(path, np.float32).reshape(shape)


def compare(what, ref, ours):
    rel = np.linalg.norm(ref - ours) / np.linalg.norm(ref)
    cos = float((ref * ours).sum() / (np.linalg.norm(ref) * np.linalg.norm(ours)))
    print(
        f"{what}: relative L2 error {rel:.3g}, cosine {cos:.6f}, "
        f"max abs difference {np.abs(ref - ours).max():.3g}"
    )


def box_iou(a, b):
    """IoU of inclusive xyxy boxes, as torchvision computes it."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_box(mask):
    """Inclusive xyxy box of a binary mask (batched_mask_to_box)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def generate(predictor, h, w, points_per_side, iou_thresh, stab_thresh):
    """Single-crop SAM2AutomaticMaskGenerator on the set image."""
    off = 1 / (2 * points_per_side)
    side = np.linspace(off, 1 - off, points_per_side)
    grid = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
    points = torch.as_tensor(grid * np.array([w, h]), dtype=torch.float32)
    keep_masks, keep_iou, keep_stab = [], [], []
    for b in range(0, len(points), 64):
        pts = predictor._transforms.transform_coords(
            points[b : b + 64], normalize=True, orig_hw=(h, w)
        )
        labels = torch.ones(len(pts), dtype=torch.int)
        masks, iou, _ = predictor._predict(
            pts[:, None, :], labels[:, None], multimask_output=True, return_logits=True
        )
        masks, iou = masks.flatten(0, 1), iou.flatten(0, 1)
        keep = iou > iou_thresh
        masks, iou = masks[keep], iou[keep]
        inter = (masks > 1.0).sum((-1, -2))
        union = (masks > -1.0).sum((-1, -2))
        stab = inter / union
        keep = stab >= stab_thresh
        keep_masks.append(masks[keep] > 0)
        keep_iou.append(iou[keep])
        keep_stab.append(stab[keep])
    masks = torch.cat(keep_masks)
    iou = torch.cat(keep_iou)
    stab = torch.cat(keep_stab)
    boxes = torch.tensor([mask_box(m.numpy()) for m in masks], dtype=torch.float32)
    if not len(boxes):
        return []
    keep = batched_nms(boxes, iou, torch.zeros(len(boxes)), 0.7)
    return [
        (boxes[i].tolist(), float(iou[i]), float(stab[i]), int(masks[i].sum()))
        for i in keep
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--iou", type=float, default=0.8)
    parser.add_argument("--stability", type=float, default=0.95)
    args = parser.parse_args()

    rgb_path = min(glob.glob(os.path.join(args.dump_dir, "crop000_rgb_*.u8")))
    h, w = map(int, re.search(r"_(\d+)x(\d+)\.u8$", rgb_path).groups())
    rgb = np.fromfile(rgb_path, dtype=np.uint8).reshape(h, w, 3)
    inp = load_dump(args.dump_dir, "input", (3, 1008, 1008))
    emb = load_dump(args.dump_dir, "embedding", (72, 72, 256))
    s0 = load_dump(args.dump_dir, "feat_s0", (288, 288, 32))
    s1 = load_dump(args.dump_dir, "feat_s1", (144, 144, 64))

    cpu_only()
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        enable_inst_interactivity=True,
        device="cpu",
    )
    processor = Sam3Processor(model, device="cpu")
    image = Image.fromarray(rgb)

    with torch.inference_mode():
        from torchvision.transforms import v2

        ref_inp = processor.transform(v2.functional.to_image(image)).numpy()
        state = processor.set_image(image)
        pred = model.inst_interactive_predictor
        _, feats, _, _ = pred.model._prepare_backbone_features(
            state["backbone_out"]["sam2_backbone_out"]
        )
        feats[-1] = feats[-1] + pred.model.no_mem_embed
        feats = [
            f.permute(1, 2, 0).view(1, -1, *size)
            for f, size in zip(feats[::-1], pred._bb_feat_sizes[::-1])
        ][::-1]
        pred._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        pred._is_image_set = True
        pred._orig_hw = [(h, w)]

    print(f"encoder input: max abs difference {np.abs(ref_inp - inp).max():.3g}")
    compare("embedding", feats[-1][0].permute(1, 2, 0).numpy(), emb)
    compare("features s0", feats[0][0].permute(1, 2, 0).numpy(), s0)
    compare("features s1", feats[1][0].permute(1, 2, 0).numpy(), s1)

    with torch.inference_mode():
        ref = generate(pred, h, w, args.points, args.iou, args.stability)
    ours = np.loadtxt(
        os.path.join(args.dump_dir, "masks.csv"), delimiter=",", skiprows=1, ndmin=2
    )
    print(f"masks: reference {len(ref)}, i.sam3.opencl {len(ours)}")
    matched = 0
    ious = []
    for row in ours:
        scores = [box_iou(row[:4], r[0]) for r in ref]
        best = int(np.argmax(scores)) if scores else -1
        ious.append(scores[best] if scores else 0.0)
        if scores and scores[best] > 0.9:
            matched += 1
            print(
                f"  box {row[:4].astype(int).tolist()}: IoU {scores[best]:.3f}, "
                f"area {int(row[4])} vs {ref[best][3]}, "
                f"predicted_iou {row[5]:.4f} vs {ref[best][1]:.4f}, "
                f"stability {row[6]:.4f} vs {ref[best][2]:.4f}"
            )
    if ious:
        print(
            f"masks with a reference box IoU > 0.9: {matched}/{len(ours)} "
            f"(median best IoU {np.median(ious):.3f})"
        )


if __name__ == "__main__":
    main()
