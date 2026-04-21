"""Validate Stage 2 hybrid model with concise prediction diagnostics.

This script reuses the validation logic from train_stage2.py and prints:
- loss breakdown (including action loss)
- proposal quality summary
- decoder summary (teacher forcing + free decoding)
- top token predictions and common confusion pairs
- a few GT vs prediction examples

Example:
    python validate_stage2.py \
        --phase B \
        --stage2-ckpt checkpoints/stage2_hybrid_phaseA/stage2_hybrid_best.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import CHECKPOINT_DIR, DATA_DIR, DEVICE, IMAGE_SIZE
from train_stage2 import (
    build_stage2_dataloaders,
    build_stage2_model,
    load_vocab,
    validate_stage2,
)
from utils import KuzushijiDataset
from utils.stage2_targets import build_decoder_targets
from utils.stage2_losses import build_decoder_roi_targets_monotonic
from utils.training_helpers.helper_stage1 import collate_fn
from utils.text_normalization import render_tokens


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
    pred_label: str,
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
    d.text(
        (8, base.height + 6),
        f"GT+Pred boxes | ori={orientation} | CER={cer:.4f} | {pred_label}",
        fill=(235, 235, 235),
        font=font,
    )

    y = base.height + overlay.height + 10
    y = _draw_text_block(d, 8, y, "GT:", gt_text, font, (200, 255, 200))
    _draw_text_block(d, 8, y, f"{pred_label}:", pred_text, font, (255, 220, 180))

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
    loader,
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

            encoded = model.encode_images(images=images, orientations=orientations)
            ordered_mask = encoded["ordered_mask"]
            ordered_boxes = encoded["ordered_boxes"]

            dec_targets = build_decoder_targets(
                text_ids=text_ids,
                pad_id=vocab.pad_id,
            )
            oracle_pointer_positions = build_decoder_roi_targets_monotonic(
                target_tokens=dec_targets["target_tokens"],
                target_mask=dec_targets["target_mask"],
                enc_mask=encoded.get("context_mask", None),
                eos_id=int(vocab.eos_id),
            )

            tf_outputs = model.decode_from_encoded(
                encoded,
                input_seq=dec_targets["decoder_inputs"],
                targets=dec_targets["target_tokens"],
                teacher_forcing_ratio=1.0,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                max_len=None,
                oracle_pointer_positions=oracle_pointer_positions,
            )
            pred_ids_tf = tf_outputs["decoder_logits"].argmax(dim=-1)

            free_outputs = model.decode_from_encoded(
                encoded,
                targets=None,
                teacher_forcing_ratio=0.0,
                input_seq=None,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                max_len=max_decode_len,
            )
            pred_ids_free = free_outputs.get("emitted_token_ids", None)
            if pred_ids_free is None:
                pred_ids_free = free_outputs["decoder_logits"].argmax(dim=-1)

            for i in range(images.size(0)):
                if count >= num_plot_samples:
                    return rows

                gt_ids = text_ids[i].detach().cpu().tolist()
                pred_ids_tf_row = pred_ids_tf[i].detach().cpu().tolist()
                pred_ids_free_row = pred_ids_free[i].detach().cpu().tolist()

                gt_clean = _strip_special_tokens(gt_ids, vocab.pad_id, vocab.sos_id, vocab.eos_id)
                pred_clean_tf = _strip_special_tokens(pred_ids_tf_row, vocab.pad_id, vocab.sos_id, vocab.eos_id)
                pred_clean_free = _strip_special_tokens(pred_ids_free_row, vocab.pad_id, vocab.sos_id, vocab.eos_id)

                gt_text = _ids_to_text(gt_clean, vocab)
                pred_text_tf = _ids_to_text(pred_clean_tf, vocab)
                pred_text_free = _ids_to_text(pred_clean_free, vocab)
                cer_tf = _sequence_cer(pred_text_tf, gt_text)
                cer_free = _sequence_cer(pred_text_free, gt_text)

                gt_boxes_tensor = boxes_batch[i]
                if gt_boxes_tensor is None:
                    gt_boxes = []
                elif gt_boxes_tensor.numel() == 0:
                    gt_boxes = []
                else:
                    gt_boxes = gt_boxes_tensor.detach().cpu().tolist()

                ordered_mask_i = ordered_mask[i].bool()
                pred_boxes = ordered_boxes[i][ordered_mask_i].detach().cpu().tolist()

                orientation = str(orientations[i]) if i < len(orientations) else "unknown"
                out_img_tf = plots_dir / f"sample_{count:04d}_tf.png"
                out_img_free = plots_dir / f"sample_{count:04d}_free.png"
                _save_visualization(
                    out_path=out_img_tf,
                    image_tensor=images[i],
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    gt_text=gt_text,
                    pred_text=pred_text_tf,
                    cer=cer_tf,
                    orientation=orientation,
                    font=font,
                    pred_label="TF",
                )
                _save_visualization(
                    out_path=out_img_free,
                    image_tensor=images[i],
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    gt_text=gt_text,
                    pred_text=pred_text_free,
                    cer=cer_free,
                    orientation=orientation,
                    font=font,
                    pred_label="FREE",
                )

                rows.append(
                    {
                        "index": count,
                        "plot_tf": str(out_img_tf),
                        "plot_free": str(out_img_free),
                        "orientation": orientation,
                        "gt_text": gt_text,
                        "pred_text_tf": pred_text_tf,
                        "pred_text_free": pred_text_free,
                        "cer_tf": cer_tf,
                        "cer_free": cer_free,
                        "gt_boxes": len(gt_boxes),
                        "pred_boxes": len(pred_boxes),
                    }
                )
                count += 1

    return rows
def _print_loss_summary(metrics: dict[str, Any]) -> None:
    print(
        "Loss summary | "
        f"val={metrics['val_loss']:.4f}, "
        f"dec={metrics['val_decoder']:.4f}, "
        f"stop={metrics['val_stop']:.4f}, "
        f"action={metrics['val_action']:.4f}, "
        f"box={metrics['val_box']:.4f}, "
        f"delta={metrics['val_delta']:.4f}, "
        f"score={metrics['val_score']:.4f}, "
        f"aux={metrics['val_aux']:.4f}, "
        f"token_acc={metrics['val_token_acc']:.4f}"
    )


def _print_proposal_summary(metrics: dict[str, Any]) -> None:
    prop = metrics["proposal_summary"]
    print(
        "Proposal summary | "
        f"props/img={prop['avg_proposals_per_image']:.2f}, "
        f"pos/img={prop['avg_positives_per_image']:.2f}, "
        f"gt/img={prop['avg_gt_tokens_per_image']:.2f}, "
        f"cov={prop['positive_coverage_ratio']:.3f}, "
        f"uniq_cov={prop['unique_coverage_ratio']:.3f}, "
        f"dup+={prop['duplicate_positive_rate']:.3f}, "
        f"zero/img={prop['images_with_zero_valid_props_ratio']:.3f}, "
        f"prec_proxy={prop['positive_precision_proxy']:.3f}, "
        f"mean IoU+={prop['avg_matched_iou_on_positives']:.3f}, "
        f"aux top1={prop['aux_accuracy_on_positives']:.3f}, "
        f"aux top5={prop['aux_top5_on_positives']:.3f}"
    )


def _print_decoder_summary(metrics: dict[str, Any]) -> None:
    tf = metrics["decoder_summary"]["teacher_forcing"]
    free = metrics["decoder_summary"]["free_decoding"]

    print(
        "Decoder TF | "
        f"EOS hit={tf['eos_hit_fraction']:.3f}, "
        f"len_ratio={tf['mean_length_ratio']:.3f}, "
        f"CER={tf['mean_cer']:.4f}, "
        f"len={tf['mean_pred_len']:.2f}/{tf['mean_gt_len']:.2f}, "
        f"exact={tf['exact_match_fraction']:.3f}, "
        f"action stay/adv/stop="
        f"{tf['action_pred_distribution']['fractions']['stay']:.2f}/"
        f"{tf['action_pred_distribution']['fractions']['advance']:.2f}/"
        f"{tf['action_pred_distribution']['fractions']['stop']:.2f}, "
        f"action_match={tf['action_match_fraction']:.3f}, "
        f"ptr adv/stay/reg="
        f"{tf['pointer_summary']['advance_fraction']:.2f}/"
        f"{tf['pointer_summary']['stay_fraction']:.2f}/"
        f"{tf['pointer_summary']['regress_fraction']:.2f}"
    )
    print(
        "Decoder free | "
        f"EOS hit={free['eos_hit_fraction']:.3f}, "
        f"len_ratio={free['mean_length_ratio']:.3f}, "
        f"CER={free['mean_cer']:.4f}, "
        f"len={free['mean_pred_len']:.2f}/{free['mean_gt_len']:.2f}, "
        f"exact={free['exact_match_fraction']:.3f}, "
        f"action stay/adv/stop="
        f"{free['action_pred_distribution']['fractions']['stay']:.2f}/"
        f"{free['action_pred_distribution']['fractions']['advance']:.2f}/"
        f"{free['action_pred_distribution']['fractions']['stop']:.2f}, "
        f"ptr adv/stay/reg="
        f"{free['pointer_summary']['advance_fraction']:.2f}/"
        f"{free['pointer_summary']['stay_fraction']:.2f}/"
        f"{free['pointer_summary']['regress_fraction']:.2f}"
    )


def _print_aux_predictions(metrics: dict[str, Any]) -> None:
    prop = metrics["proposal_summary"]
    aux_all = prop["aux_summary"]

    for branch_name in ("without_context_encode", "with_context_encode"):
        branch = aux_all[branch_name]
        if not branch.get("available", False):
            continue
        print(
            f"Aux branch={branch_name} | top1={branch['top1']:.3f}, "
            f"top5={branch['top5']:.3f}, n={branch['total']}"
        )
        print(f"Top predictions: {branch['top_predictions']}")
        print(f"Top errors: {branch['top_errors']}")


def _print_examples(metrics: dict[str, Any], max_examples: int) -> None:
    tf_examples = metrics["decoder_summary"]["teacher_forcing"].get("examples", [])
    free_examples = metrics["decoder_summary"]["free_decoding"].get("examples", [])
    free_pointer_examples = metrics["decoder_summary"]["free_decoding"].get("pointer_examples", [])

    if tf_examples:
        print("\nTeacher-forcing examples")
        for idx, ex in enumerate(tf_examples[:max_examples], start=1):
            print(
                f"[{idx}] CER={ex['cer']:.4f} | EOS={ex['eos_hit']} | "
                f"len pred/gt={ex['pred_len']}/{ex['gt_len']}"
            )
            print(f"  GT  : {ex['gt']}")
            print(f"  PRED: {ex['pred']}")

    if free_examples:
        print("\nFree-decoding examples")
        for idx, ex in enumerate(free_examples[:max_examples], start=1):
            print(
                f"[{idx}] CER={ex['cer']:.4f} | EOS={ex['eos_hit']} | "
                f"len pred/gt={ex['pred_len']}/{ex['gt_len']}"
            )
            print(f"  GT  : {ex['gt']}")
            print(f"  PRED: {ex['pred']}")

    if free_pointer_examples:
        print("\nFree-decoding pointer traces")
        for idx, ex in enumerate(free_pointer_examples[:max_examples], start=1):
            print(
                f"[{idx}] ptr start/end={ex['start']}/{ex['end']} | "
                f"len={ex['len']} | trace_head={ex['trace_head']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage-2 hybrid model and print key predictions.")
    parser.add_argument("--phase", type=str, default="B", choices=["A", "B"], help="Validation phase profile.")
    parser.add_argument(
        "--detector-ckpt",
        type=Path,
        default=CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt",
        help="Stage-1 detector checkpoint used for model construction.",
    )
    parser.add_argument(
        "--stage2-ckpt",
        type=Path,
        default=CHECKPOINT_DIR / "stage2_hybrid_phase_B" / "stage2_hybrid_best.pt",
        help="Stage-2 checkpoint to validate.",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for quick validation.")
    parser.add_argument("--val-max-decode-len", type=int, default=320, help="Max decode length for free decoding.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to store full validation metrics as JSON.",
    )
    parser.add_argument("--print-examples", type=int, default=3, help="How many decoder examples to print.")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save TF/FREE prediction plot panels and a samples.jsonl.",
    )
    parser.add_argument(
        "--num-plot-samples",
        type=int,
        default=8,
        help="How many TF/FREE plot samples to save.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional output directory for plots/samples.jsonl. Defaults to checkpoints/stage2_validation/<timestamp>.",
    )
    parser.add_argument(
        "--plot-font-path",
        type=str,
        default=None,
        help="Optional font path for rendering Japanese text in plots.",
    )
    parser.add_argument(
        "--plot-font-size",
        type=int,
        default=18,
        help="Font size for plot text overlays.",
    )
    args = parser.parse_args()

    if not args.detector_ckpt.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_ckpt}")
    if not args.stage2_ckpt.exists():
        raise FileNotFoundError(f"Stage2 checkpoint not found: {args.stage2_ckpt}")

    vocab = load_vocab()
    _, val_loader = build_stage2_dataloaders(vocab)

    model = build_stage2_model(
        detector_ckpt_path=args.detector_ckpt,
        vocab=vocab,
        overrides=None,
    )
    loaded_epoch = _load_stage2_weights(model, args.stage2_ckpt)

    runtime_overrides = {
        "val_max_decode_len": int(args.val_max_decode_len),
    }

    metrics = validate_stage2(
        model=model,
        val_loader=val_loader,
        vocab=vocab,
        phase=args.phase,
        max_batches=args.max_batches,
        runtime_overrides=runtime_overrides,
    )

    print("=" * 72)
    print(f"Stage2 validation done | ckpt={args.stage2_ckpt} | loaded_epoch={loaded_epoch}")
    print("=" * 72)
    _print_loss_summary(metrics)
    _print_proposal_summary(metrics)
    _print_decoder_summary(metrics)
    _print_aux_predictions(metrics)
    _print_examples(metrics, max_examples=max(0, int(args.print_examples)))

    if args.save_plots:
        if args.plots_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = CHECKPOINT_DIR / "stage2_validation" / f"{stamp}_validation"
        else:
            plots_dir = args.plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)

        font = _load_font(args.plot_font_path, int(args.plot_font_size))
        rows = _collect_plot_samples(
            model=model,
            loader=val_loader,
            vocab=vocab,
            max_decode_len=int(args.val_max_decode_len),
            num_plot_samples=int(args.num_plot_samples),
            plots_dir=plots_dir,
            font=font,
        )
        samples_path = plots_dir / "samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(rows)} plot samples to {plots_dir}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics JSON to {args.out_json}")


if __name__ == "__main__":
    main()
