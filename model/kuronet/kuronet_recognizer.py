"""KuroNet-style per-ROI character recognizer.

Pipeline:
    image
    -> frozen backbone (Stage 1 UNet)
    -> frozen detector (Stage 1 detector) -> coarse proposals
    -> feature projector
    -> ROI pool encoder
    -> ROI refinement head
    -> reading-order sorting (deterministic heuristic)
    -> ROI token projector  (fuses visual + geometry + quality score)
    -> [optional] BiGRU context encoder
    -> MLP classifier  -> (B, T, vocab_size) per-ROI character logits

At inference: argmax over classifier logits, sorted by reading order,
decoded with vocabulary -> transcription string.

This replaces the seq2seq HybridKuroNetRecognizer decoder with a direct
per-ROI classification head. All ROI pipeline modules are reused unchanged.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from model.kuronet.backbone.feature_projector import FeatureProjector
from model.kuronet.context.roi_context import ROIContextEncoder
from model.kuronet.detection.proposal_utils import extract_coarse_proposals
from model.kuronet.roi.roi_ordering import ROIReadingOrder
from model.kuronet.roi.roi_pool import ROIPoolEncoder
from model.kuronet.roi.roi_refinement import ROIRefinementHead
from model.kuronet.roi.roi_tokens import ROITokenProjector


class KuroNetRecognizer(nn.Module):
    """
    KuroNet-style character recognizer.

    Detects character boxes, classifies each one independently,
    then assembles the transcription in reading order.

    Key differences from HybridKuroNetRecognizer:
    - No SeqDecoderAttention, no pointer mechanism, no action/stop heads
    - Auxiliary classification head is promoted to PRIMARY output
    - Single-phase training, no teacher-forcing schedule
    - Inference: argmax per ROI -> sort by reading order -> text
    """

    def __init__(
        self,
        backbone: nn.Module,
        detector: nn.Module,
        backbone_out_channels: int,
        vocab_size: int,

        # Feature projection
        proj_dim: int = 256,

        # ROI pooling
        roi_size: tuple[int, int] = (8, 8),
        roi_feat_dim: int = 384,

        # ROI refinement
        refine_hidden_dim: int = 256,

        # ROI token projection
        token_dim: int = 256,
        token_hidden_dim: int = 512,
        token_use_score_branch: bool = True,

        # Context encoder (optional BiGRU)
        use_context: bool = True,
        context_hidden_dim: int = 384,
        context_num_layers: int = 2,

        # Classifier head
        classifier_hidden_dim: int = 512,

        # Detector proposal thresholds
        det_score_thresh: float = 0.26,
        det_top_k: int = 896,
        det_nms_iou: float = 0.50,
        det_min_box_size: float = 1.66,

        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = backbone
        self.detector = detector

        self.det_score_thresh = float(det_score_thresh)
        self.det_top_k = int(det_top_k)
        self.det_nms_iou = float(det_nms_iou)
        self.det_min_box_size = float(det_min_box_size)

        self.vocab_size = vocab_size
        self.use_context = bool(use_context)

        # --- ROI pipeline (identical to HybridKuroNetRecognizer) ---

        self.feature_projector = FeatureProjector(
            in_channels=backbone_out_channels,
            out_channels=proj_dim,
            hidden_channels=proj_dim,
            dropout=dropout,
        )

        self.roi_pool = ROIPoolEncoder(
            in_channels=proj_dim,
            roi_size=roi_size,
            conv_channels=proj_dim,
            out_dim=roi_feat_dim,
            dropout=dropout,
            predict_aux_logits=False,   # classifier is at top level, not in pool
            vocab_size=None,
        )

        self.roi_refine = ROIRefinementHead(
            feat_dim=roi_feat_dim,
            hidden_dim=refine_hidden_dim,
            dropout=dropout,
        )

        self.roi_order = ROIReadingOrder(line_merge_thresh_ratio=0.6)

        self.roi_tokens = ROITokenProjector(
            roi_feat_dim=roi_feat_dim,
            token_dim=token_dim,
            hidden_dim=token_hidden_dim,
            dropout=dropout,
            use_score_branch=bool(token_use_score_branch),
        )

        # --- Context encoder (optional) ---

        if self.use_context:
            self.context_encoder = ROIContextEncoder(
                in_dim=token_dim,
                hidden_dim=context_hidden_dim,
                out_dim=context_hidden_dim,
                num_layers=int(context_num_layers),
                dropout=dropout,
                mode="bigru",
                use_layernorm=True,
                use_residual=True,
            )
            classifier_in_dim = context_hidden_dim
        else:
            self.context_encoder = None
            classifier_in_dim = token_dim

        # --- Primary classification head ---

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, vocab_size),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_proposals(
        self,
        shared_feats: torch.Tensor,
        image_size: tuple[int, int],
    ):
        """Run frozen detector and extract coarse proposals."""
        det_out = self.detector(shared_feats)

        if "heatmap" not in det_out or "bbox" not in det_out:
            raise ValueError("Detector output must contain 'heatmap' and 'bbox'.")

        coarse_boxes_list, coarse_scores_list = extract_coarse_proposals(
            heat_logits=det_out["heatmap"],
            bbox_reg=det_out["bbox"],
            image_size=image_size,
            score_thresh=self.det_score_thresh,
            top_k=self.det_top_k,
            nms_iou=self.det_nms_iou,
            min_size=self.det_min_box_size,
        )

        return det_out, coarse_boxes_list, coarse_scores_list

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(
        self,
        images: torch.Tensor,           # (B, 3, H, W)
        orientations: List[str],
        coarse_boxes_list: Optional[List[torch.Tensor]] = None,
        coarse_scores_list: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """
        Run the full ROI pipeline and produce per-ROI feature vectors.

        Returns a dict with all intermediate tensors needed for loss
        computation and for the classifier head.
        """
        if images.dim() != 4:
            raise ValueError(f"images must be (B, C, H, W), got {tuple(images.shape)}")

        bsz, _, h_img, w_img = images.shape
        image_size = (h_img, w_img)

        # --- Backbone + detector proposals ---
        shared_feats = self.backbone(images)

        if coarse_boxes_list is None or coarse_scores_list is None:
            det_out, coarse_boxes_list, coarse_scores_list = self._extract_proposals(
                shared_feats=shared_feats,
                image_size=image_size,
            )
        else:
            det_out = None

        # --- Feature projection ---
        proj_feats = self.feature_projector(shared_feats)

        # --- ROI pooling ---
        roi_out = self.roi_pool(
            feat_2d=proj_feats,
            proposal_boxes_list=coarse_boxes_list,
            proposal_scores_list=coarse_scores_list,
            image_size=image_size,
        )

        # --- ROI refinement ---
        refine_out = self.roi_refine(
            roi_feats=roi_out["roi_feats"],
            roi_boxes=roi_out["roi_boxes"],
            roi_scores=roi_out["roi_scores"],
            roi_mask=roi_out["roi_mask"],
            image_size=image_size,
        )

        # --- Reading-order sorting ---
        # sort_batch needs aux_logits placeholder; use zeros since we have none at pool stage
        aux_placeholder = torch.zeros(
            (*roi_out["roi_feats"].shape[:2], self.vocab_size),
            device=roi_out["roi_feats"].device,
            dtype=roi_out["roi_feats"].dtype,
        )

        ordered = self.roi_order.sort_batch(
            boxes=refine_out["refined_boxes"],
            mask=roi_out["roi_mask"],
            orientations=orientations,
            roi_feats=roi_out["roi_feats"],
            refined_feats=refine_out["refined_feats"],
            refine_scores=refine_out["refine_scores"],
            aux_logits=aux_placeholder,
        )

        # --- ROI token projection ---
        token_feats, token_mask = self.roi_tokens(
            roi_feats=ordered["roi_feats"],
            refined_feats=ordered["refined_feats"],
            refined_boxes=ordered["boxes"],
            refine_scores=ordered["refine_scores"],
            roi_mask=ordered["mask"],
            image_size=image_size,
        )

        # --- Context encoder (optional BiGRU) ---
        if self.use_context and self.context_encoder is not None:
            context_feats, context_mask = self.context_encoder(
                seq=token_feats,
                mask=token_mask,
            )
        else:
            context_feats = token_feats
            context_mask = token_mask

        return {
            # Raw proposal info (for loss building with original unordered boxes)
            "coarse_boxes_list": coarse_boxes_list,
            "coarse_scores_list": coarse_scores_list,
            "detector_outputs": det_out,
            "proj_feats": proj_feats,

            # Unordered ROI outputs (used for refinement target matching)
            "roi_feats": roi_out["roi_feats"],
            "roi_boxes": roi_out["roi_boxes"],
            "roi_scores": roi_out["roi_scores"],
            "roi_mask": roi_out["roi_mask"],

            # Refinement outputs (unordered, for loss computation)
            "refined_feats": refine_out["refined_feats"],
            "refined_boxes": refine_out["refined_boxes"],
            "refine_scores": refine_out["refine_scores"],
            "box_deltas": refine_out["box_deltas"],

            # Reading-order sorted outputs
            "ordered_boxes": ordered["boxes"],
            "ordered_mask": ordered["mask"],
            "sort_indices": ordered["sort_indices"],
            "ordering_diagnostics": ordered.get("ordering_diagnostics", None),

            # Token and context features (sorted order)
            "token_feats": token_feats,
            "token_mask": token_mask,
            "context_feats": context_feats,
            "context_mask": context_mask,
        }

    def forward(
        self,
        images: torch.Tensor,
        orientations: List[str],
        coarse_boxes_list: Optional[List[torch.Tensor]] = None,
        coarse_scores_list: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """
        Full forward pass.

        Returns the encode() dict plus:
            char_logits: (B, T, vocab_size)  per-ROI character logits (sorted order)
        """
        encoded = self.encode(
            images=images,
            orientations=orientations,
            coarse_boxes_list=coarse_boxes_list,
            coarse_scores_list=coarse_scores_list,
        )

        char_logits = self.classifier(encoded["context_feats"])  # (B, T, V)

        return {**encoded, "char_logits": char_logits}

    @torch.no_grad()
    def transcribe(
        self,
        images: torch.Tensor,
        orientations: List[str],
        vocab,
        score_thresh: float = 0.0,
    ) -> List[str]:
        """
        Inference: returns a transcription string per image.

        For each image:
          1. Run forward pass
          2. Argmax over char_logits to get predicted char IDs
          3. Filter by roi_mask (and optionally refine_score threshold)
          4. ROIs are already in reading order from sort_batch
          5. Decode IDs with vocab -> list of chars -> join as string

        Args:
            score_thresh: minimum refine_score (sigmoid) to include an ROI.
                          0.0 = include all valid ROIs (roi_mask only).
        """
        self.eval()
        outputs = self.forward(images, orientations)

        char_logits = outputs["char_logits"]        # (B, T, V)
        ordered_mask = outputs["ordered_mask"]      # (B, T)
        refine_scores = outputs["refine_scores"]    # (B, T) - unordered
        sort_indices = outputs["sort_indices"]      # (B, T) - maps ordered -> original

        bsz = images.size(0)
        results: List[str] = []

        for b in range(bsz):
            mask_b = ordered_mask[b]  # (T,)

            if score_thresh > 0.0 and sort_indices is not None:
                # Reorder refine_scores into sorted order for filtering
                si = sort_indices[b]  # (T,)
                valid_si = si[mask_b]
                scores_ordered = refine_scores[b].index_select(0, valid_si)
                score_mask = torch.sigmoid(scores_ordered) >= score_thresh
                valid_positions = mask_b.nonzero(as_tuple=True)[0][score_mask]
            else:
                valid_positions = mask_b.nonzero(as_tuple=True)[0]

            if valid_positions.numel() == 0:
                results.append("")
                continue

            logits_b = char_logits[b, valid_positions]  # (N_valid, V)
            pred_ids = logits_b.argmax(dim=-1).tolist()  # (N_valid,)

            # Decode with vocab (skip special tokens)
            chars = vocab.decode(pred_ids, remove_special=True)
            results.append("".join(chars))

        return results
