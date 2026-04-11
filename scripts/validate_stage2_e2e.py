"""Stage-2 validation script for the current hybrid model.

Features:
- Runs numeric validation via train_stage2.validate_stage2
- Saves summary metrics JSON
- Optionally writes GT vs prediction visualizations for N samples

CLI is compatible with submit_scripts/submit_stage2_validation.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_DIR, DATA_DIR, DEVICE, IMAGE_SIZE
from train_stage2 import build_stage2_model, load_vocab, validate_stage2
from utils import KuzushijiDataset
from utils.text_normalization import render_tokens
from utils.training_helpers.helper_stage1 import collate_fn


def _load_stage2_weights(model: torch.nn.Module, stage2_ckpt_path: Path) -> int:
    checkpoint = torch.load(stage2_ckpt_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = int(checkpoint.get("epoch", -1))
    else:
        state_dict = checkpoint
        epoch = -1
    model.load_state_dict(state_dict, strict=True)
    return epoch


def _strip_special_tokens(ids: list[int], pad_id: int, sos_id: int, eos_id: int) -> list[int]:
    out: list[int] = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id in (pad_id, sos_id):
            continue
        if token_id == eos_id:
            break
        out.append(token_id)
    return out


def _ids_to_text(ids: list[int], vocab) -> str:
    chars = vocab.decode([int(x) for x in ids], remove_special=True)
    return render_tokens(chars)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _sequence_cer(pred_text: str, gt_text: str) -> float:
    return _edit_distance(pred_text, gt_text) / max(1, len(gt_text))


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        path = Path(font_path)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), font_size)
            except Exception:
                pass

    candidates = [
        "/usr/share/fonts/truetype/hanazono/HanaMinA.ttf",
        "/usr/share/fonts/truetype/hanazono/HanaMinB.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), font_size)
            except Exception:
                continue

    return ImageFont.load_default()


def _draw_text_block(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, text: str, font, color) -> int:
    draw.text((x, y), title, fill=(255, 220, 120), font=font)
    y += 20
    max_chars = 60
    text = text if len(text) <= 200 else text[:200] + " ..."
    for i in range(0, len(text), max_chars):
        draw.text((x, y), text[i:i + max_chars], fill=color, font=font)
        y += 18
    return y + 6


def _save_visualization(
    out_path: Path,
    image_tensor: torch.Tensor,
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    gt_text: str,
    pred_text: str,
    cer: float,
    orientation: str,
    font,
) -> None:
    rgb = _denormalize_image(image_tensor)
    base = Image.fromarray(rgb).convert("RGB")
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)

    for box in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(40, 220, 80), width=1)

    for box in pred_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(230, 40, 40), width=2)

    panel_w = max(base.width, overlay.width)
    panel_h = base.height + overlay.height + 230
    panel = Image.new("RGB", (panel_w, panel_h), color=(26, 26, 26))
    panel.paste(base, (0, 0))
    panel.paste(overlay, (0, base.height))

    d = ImageDraw.Draw(panel)
    d.text((8, 6), "Original", fill=(235, 235, 235), font=font)
    d.text((8, base.height + 6), f"GT+Pred boxes | ori={orientation} | CER={cer:.4f}", fill=(235, 235, 235), font=font)

    y = base.height + overlay.height + 10
    y = _draw_text_block(d, 8, y, "GT:", gt_text, font, (200, 255, 200))
    _draw_text_block(d, 8, y, "PRED:", pred_text, font, (255, 220, 180))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def _filter_batch_with_text(batch: dict[str, Any]) -> dict[str, Any] | None:
    text_ids = batch.get("text_ids", None)
    if text_ids is None:
        return None

    present = batch.get("text_ids_present", None)
    if present is None:
        return {
            "images": batch["image"].to(DEVICE),
            "text_ids": text_ids.to(DEVICE),
            "boxes": batch.get("boxes", []),
            "orientations": batch.get("orientations", []),
        }

    valid_idx = present.to(DEVICE).nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return None

    idx_list = valid_idx.detach().cpu().tolist()
    return {
        "images": batch["image"].to(DEVICE).index_select(0, valid_idx),
        "text_ids": text_ids.to(DEVICE).index_select(0, valid_idx),
        "boxes": [batch.get("boxes", [])[i] for i in idx_list],
        "orientations": [batch.get("orientations", [])[i] for i in idx_list],
    }


def _collect_plot_samples(
    *,
    model,
    loader: DataLoader,
    vocab,
    max_decode_len: int,
    num_plot_samples: int,
    plots_dir: Path,
    font,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    count = 0

    with torch.no_grad():
        for batch in loader:
            prepared = _filter_batch_with_text(batch)
            if prepared is None:
                continue

            images = prepared["images"]
            text_ids = prepared["text_ids"]
            boxes_batch = prepared["boxes"]
            orientations = prepared["orientations"]

            outputs = model(
                images=images,
                orientations=orientations,
                targets=None,
                teacher_forcing_ratio=0.0,
                input_seq=None,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                max_len=max_decode_len,
            )
            pred_ids_batch = outputs["decoder_logits"].argmax(dim=-1)

            for i in range(images.size(0)):
                if count >= num_plot_samples:
                    return rows

                gt_ids = text_ids[i].detach().cpu().tolist()
                pred_ids = pred_ids_batch[i].detach().cpu().tolist()

                gt_clean = _strip_special_tokens(gt_ids, vocab.pad_id, vocab.sos_id, vocab.eos_id)
                pred_clean = _strip_special_tokens(pred_ids, vocab.pad_id, vocab.sos_id, vocab.eos_id)

                gt_text = _ids_to_text(gt_clean, vocab)
                pred_text = _ids_to_text(pred_clean, vocab)
                cer = _sequence_cer(pred_text, gt_text)

                gt_boxes_tensor = boxes_batch[i]
                if gt_boxes_tensor is None:
                    gt_boxes = []
                elif gt_boxes_tensor.numel() == 0:
                    gt_boxes = []
                else:
                    gt_boxes = gt_boxes_tensor.detach().cpu().tolist()

                ordered_mask = outputs["ordered_mask"][i].bool()
                pred_boxes = outputs["ordered_boxes"][i][ordered_mask].detach().cpu().tolist()

                orientation = str(orientations[i]) if i < len(orientations) else "unknown"
                out_img = plots_dir / f"sample_{count:04d}.png"
                _save_visualization(
                    out_path=out_img,
                    image_tensor=images[i],
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    gt_text=gt_text,
                    pred_text=pred_text,
                    cer=cer,
                    orientation=orientation,
                    font=font,
                )

                rows.append(
                    {
                        "index": count,
                        "plot": str(out_img),
                        "orientation": orientation,
                        "gt_text": gt_text,
                        "pred_text": pred_text,
                        "cer": cer,
                        "gt_boxes": len(gt_boxes),
                        "pred_boxes": len(pred_boxes),
                    }
                )
                count += 1

    return rows


def _print_summary(metrics: dict[str, Any]) -> None:
    prop = metrics["proposal_summary"]
    dec_tf = metrics["decoder_summary"]["teacher_forcing"]
    dec_free = metrics["decoder_summary"]["free_decoding"]

    print("Loss summary | "
          f"val={metrics['val_loss']:.4f}, dec={metrics['val_decoder']:.4f}, "
          f"stop={metrics['val_stop']:.4f}, action={metrics['val_action']:.4f}, "
          f"box={metrics['val_box']:.4f}, delta={metrics['val_delta']:.4f}, "
          f"score={metrics['val_score']:.4f}, aux={metrics['val_aux']:.4f}")
    print("Proposal summary | "
          f"cov={prop['positive_coverage_ratio']:.3f}, uniq_cov={prop['unique_coverage_ratio']:.3f}, "
          f"aux_top1={prop['aux_accuracy_on_positives']:.3f}, aux_top5={prop['aux_top5_on_positives']:.3f}, "
          f"mean_iou+={prop['avg_matched_iou_on_positives']:.3f}")
    print("Decoder TF | "
            f"CER={dec_tf['mean_cer']:.4f}, EOS={dec_tf['eos_hit_fraction']:.3f}, "
            f"len_ratio={dec_tf['mean_length_ratio']:.3f}, "
            f"len={dec_tf['mean_pred_len']:.2f}/{dec_tf['mean_gt_len']:.2f}, "
            f"exact={dec_tf['exact_match_fraction']:.3f}, "
            f"action stay/adv/stop="
            f"{dec_tf['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{dec_tf['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{dec_tf['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"ptr adv/stay/reg="
            f"{dec_tf['pointer_summary']['advance_fraction']:.2f}/"
            f"{dec_tf['pointer_summary']['stay_fraction']:.2f}/"
            f"{dec_tf['pointer_summary']['regress_fraction']:.2f}")
    print("Decoder free | "
            f"CER={dec_free['mean_cer']:.4f}, EOS={dec_free['eos_hit_fraction']:.3f}, "
            f"len_ratio={dec_free['mean_length_ratio']:.3f}, "
            f"len={dec_free['mean_pred_len']:.2f}/{dec_free['mean_gt_len']:.2f}, "
            f"exact={dec_free['exact_match_fraction']:.3f}, "
            f"action stay/adv/stop="
            f"{dec_free['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{dec_free['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{dec_free['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"ptr adv/stay/reg="
            f"{dec_free['pointer_summary']['advance_fraction']:.2f}/"
            f"{dec_free['pointer_summary']['stay_fraction']:.2f}/"
            f"{dec_free['pointer_summary']['regress_fraction']:.2f}")
    print(f"Pointer examples (free): {dec_free.get('pointer_examples', [])}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Stage-2 hybrid end-to-end style.")
    p.add_argument("--stage1_ckpt", type=str, default=None)
    p.add_argument("--stage2_ckpt", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=0, help="0 means full split")
    p.add_argument("--batch_size", type=int, default=2)

    # Compatibility args from deprecated flow (accepted, partially unused)
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=300)
    p.add_argument("--nms_iou", type=float, default=0.5)
    p.add_argument("--det_iou_thr", type=float, default=0.5)
    p.add_argument("--text_iou_thr", type=float, default=0.5)

    p.add_argument("--job_id", type=str, default=None)
    p.add_argument("--stage2_decode_boxes", type=str, default="both", choices=["pred", "gt", "both"])
    p.add_argument("--phase", type=str, default="B", choices=["A", "B"])
    p.add_argument("--max_decode_len", type=int, default=320)

    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--num_plot_samples", type=int, default=20)
    p.add_argument("--plot_font_path", type=str, default=None)
    p.add_argument("--plot_font_size", type=int, default=12)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    detector_ckpt = Path(args.stage1_ckpt) if args.stage1_ckpt else CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    stage2_ckpt = Path(args.stage2_ckpt) if args.stage2_ckpt else CHECKPOINT_DIR / "stage2_hybrid_phase_B" / "stage2_hybrid_best.pt"

    if not detector_ckpt.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {detector_ckpt}")
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Stage2 checkpoint not found: {stage2_ckpt}")

    run_job = args.job_id or os.environ.get("SLURM_JOB_ID") or "manual"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_job{run_job}"
    out_dir = CHECKPOINT_DIR / "stage2_validation" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    vocab = load_vocab()
    dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split=args.split,
    )

    if int(args.num_samples) > 0:
        n = min(int(args.num_samples), len(dataset))
        dataset = Subset(dataset, list(range(n)))

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id),
        pin_memory=(DEVICE == "cuda"),
    )

    model = build_stage2_model(detector_ckpt_path=detector_ckpt, vocab=vocab, overrides=None)
    loaded_epoch = _load_stage2_weights(model, stage2_ckpt)

    metrics = validate_stage2(
        model=model,
        val_loader=loader,
        vocab=vocab,
        phase=args.phase,
        max_batches=None,
        runtime_overrides={"val_max_decode_len": int(args.max_decode_len)},
    )

    artifacts: dict[str, Any] = {
        "plots_dir": None,
        "num_plots_written": 0,
        "samples_jsonl": None,
    }

    if args.save_plots:
        font = _load_font(args.plot_font_path, int(args.plot_font_size))
        rows = _collect_plot_samples(
            model=model,
            loader=loader,
            vocab=vocab,
            max_decode_len=int(args.max_decode_len),
            num_plot_samples=max(0, int(args.num_plot_samples)),
            plots_dir=plots_dir,
            font=font,
        )
        samples_path = out_dir / "samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        artifacts["plots_dir"] = str(plots_dir)
        artifacts["num_plots_written"] = len(rows)
        artifacts["samples_jsonl"] = str(samples_path)

    result = {
        "run": {
            "split": args.split,
            "num_samples": int(args.num_samples),
            "batch_size": int(args.batch_size),
            "device": DEVICE,
            "job_id": run_job,
            "loaded_epoch": loaded_epoch,
            "phase": args.phase,
            "stage2_decode_boxes": args.stage2_decode_boxes,
            "stage1_ckpt": str(detector_ckpt),
            "stage2_ckpt": str(stage2_ckpt),
        },
        "metrics": metrics,
        "artifacts": artifacts,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("STAGE2 VALIDATION")
    print("=" * 72)
    print(f"checkpoint: {stage2_ckpt}")
    print(f"loaded_epoch: {loaded_epoch}")
    _print_summary(metrics)
    print(f"saved metrics: {metrics_path}")
    if artifacts["plots_dir"] is not None:
        print(f"saved plots: {artifacts['plots_dir']} ({artifacts['num_plots_written']})")
        print(f"saved samples: {artifacts['samples_jsonl']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
