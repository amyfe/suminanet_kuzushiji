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

from config import KURONET_BG_SCORE_GATE, KURONET_CROP_ENCODER_CHUNK_SIZE, KURONET_CROP_ENCODER_SIZE
from model.kuronet.backbone.feature_projector import FeatureProjector
from model.kuronet.backbone.roi_crop_encoder import ROICropEncoder
from model.kuronet.context.roi_context import ROIContextEncoder
from model.kuronet.detection.proposal_utils import extract_coarse_proposals
from model.kuronet.roi.roi_ordering import ROIReadingOrder
from model.kuronet.roi.roi_pool import ROIPoolEncoder
from model.kuronet.roi.roi_refinement import ROIRefinementHead
from model.kuronet.roi.roi_tokens import ROITokenProjector
from typing import Literal

def _compute_neighbor_features(
    boxes: torch.Tensor,   # (B, T, 4) xyxy, sorted by reading order
    mask: torch.Tensor,    # (B, T) bool
) -> torch.Tensor:
    """
    Per-ROI relative geometry to adjacent neighbors in reading order.

    Returns (B, T, 6):
        [dcx_prev/w, dcy_prev/h, log_area_ratio_prev,
         dcx_next/w, dcy_next/h, log_area_ratio_next]

    A very small neighbor (log_area_ratio << 0) signals a potential dakuten mark.
    Boundary and invalid positions are zero-filled.
    """
    B, T, _ = boxes.shape
    device, dtype = boxes.device, boxes.dtype

    cx       = (boxes[..., 0] + boxes[..., 2]) * 0.5       # (B, T)
    cy       = (boxes[..., 1] + boxes[..., 3]) * 0.5
    w        = (boxes[..., 2] - boxes[..., 0]).clamp(min=1e-6)
    h        = (boxes[..., 3] - boxes[..., 1]).clamp(min=1e-6)
    log_area = (w * h).clamp(min=1e-6).log()

    # Which positions have a valid previous / next neighbor
    prev_valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    next_valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    if T > 1:
        prev_valid[:, 1:] = mask[:, :-1]
        next_valid[:, :-1] = mask[:, 1:]
    prev_valid &= mask
    next_valid &= mask

    # Shifted copies (only meaningful where *_valid is True)
    cx_prev = torch.zeros_like(cx)
    cy_prev = torch.zeros_like(cy)
    la_prev = torch.zeros_like(log_area)
    cx_prev[:, 1:] = cx[:, :-1]
    cy_prev[:, 1:] = cy[:, :-1]
    la_prev[:, 1:] = log_area[:, :-1]

    cx_next = torch.zeros_like(cx)
    cy_next = torch.zeros_like(cy)
    la_next = torch.zeros_like(log_area)
    cx_next[:, :-1] = cx[:, 1:]
    cy_next[:, :-1] = cy[:, 1:]
    la_next[:, :-1] = log_area[:, 1:]

    dcx_prev  = torch.zeros_like(cx)
    dcy_prev  = torch.zeros_like(cy)
    dlog_prev = torch.zeros_like(log_area)
    dcx_prev[prev_valid]  = ((cx - cx_prev) / w)[prev_valid]
    dcy_prev[prev_valid]  = ((cy - cy_prev) / h)[prev_valid]
    dlog_prev[prev_valid] = (la_prev - log_area)[prev_valid]

    dcx_next  = torch.zeros_like(cx)
    dcy_next  = torch.zeros_like(cy)
    dlog_next = torch.zeros_like(log_area)
    dcx_next[next_valid]  = ((cx_next - cx) / w)[next_valid]
    dcy_next[next_valid]  = ((cy_next - cy) / h)[next_valid]
    dlog_next[next_valid] = (la_next - log_area)[next_valid]

    return torch.stack(
        [dcx_prev, dcy_prev, dlog_prev, dcx_next, dcy_next, dlog_next], dim=-1
    )  # (B, T, 6)


def _compute_block_ids(
    ordered_boxes: torch.Tensor,  # (B, T, 4) xyxy, sorted by reading order
    ordered_mask: torch.Tensor,   # (B, T) bool
    gap_factor: float = 2.5,
) -> torch.Tensor:
    """
    Assign monotonically increasing block IDs per image in the batch.

    Returns (B, T) LongTensor. Each valid position gets an integer ≥ 0 indicating
    which independent text block it belongs to. Invalid (padded) positions get -1.

    A block boundary is detected wherever the Euclidean distance between consecutive
    sorted box centres exceeds gap_factor × median inter-box distance. This catches
    transitions from main text to marginal annotations, headers, or column labels
    without requiring explicit layout analysis.
    """
    B, T, _ = ordered_boxes.shape
    block_ids = torch.full((B, T), -1, dtype=torch.long, device=ordered_boxes.device)

    cx = (ordered_boxes[..., 0] + ordered_boxes[..., 2]) * 0.5  # (B, T)
    cy = (ordered_boxes[..., 1] + ordered_boxes[..., 3]) * 0.5

    for b in range(B):
        mask_b = ordered_mask[b]   # (T,) bool
        if not mask_b.any():
            continue
        valid_idx = mask_b.nonzero(as_tuple=True)[0]   # indices of valid positions
        n = len(valid_idx)
        if n == 1:
            block_ids[b, valid_idx[0]] = 0
            continue

        vcx = cx[b, valid_idx]   # (N,)
        vcy = cy[b, valid_idx]
        dx  = vcx[1:] - vcx[:-1]
        dy  = vcy[1:] - vcy[:-1]
        dist = (dx * dx + dy * dy).sqrt()   # (N-1,)

        med = dist.median().clamp(min=1e-6)
        is_boundary = dist > gap_factor * med   # (N-1,) bool

        bid = 0
        for i in range(n):
            block_ids[b, valid_idx[i]] = bid
            if i < n - 1 and is_boundary[i]:
                bid += 1

    return block_ids


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
        roi_pool_output_size: tuple[int, int] = (4, 4),
        roi_feat_dim: int = 384,

        # ROI refinement
        refine_hidden_dim: int = 256,
        residual_scale_init: float = 0.5,

        # ROI token projection
        token_dim: int = 256,
        token_hidden_dim: int = 512,
        token_use_score_branch: bool = True,

        # Context encoder (optional GRU/BiGRU)
        use_context: bool = True,
        context_hidden_dim: int = 384,
        context_num_layers: int = 2,
        context_mode: Literal["bigru", "gru"] = "bigru",
        context_block_gap_factor: float = 2.5,

        # Classifier head
        classifier_hidden_dim: int = 512,

        # Detector proposal thresholds
        det_score_thresh: float = 0.26,
        det_top_k: int = 896,
        det_nms_iou: float = 0.50,
        det_min_box_size: float = 1.66,

        # Per-cell cap on coarse proposals (suppresses illustration/noise FP
        # clusters without discarding genuinely dense text columns)
        density_grid: int = 8,
        density_factor: float = 3.0,
        avg_gt_per_image: int = 236,

        dropout: float = 0.1,

        # Background class ID (last vocab token). Predictions equal to this
        # are filtered out of the transcription at inference time.
        bg_id: Optional[int] = None,

        # Pretrained EfficientNet-B0 on raw image crops (Clanuwat VGG-16 equivalent).
        # When enabled, each ROI crop is fed through a frozen pretrained network and
        # the resulting features are added to roi_feats before refinement.
        use_crop_encoder: bool = True,
        crop_encoder_size: tuple[int, int] = KURONET_CROP_ENCODER_SIZE,
        freeze_crop_encoder: bool = True,
        crop_encoder_chunk_size: int = KURONET_CROP_ENCODER_CHUNK_SIZE,
    ):
        super().__init__()

        self.backbone = backbone
        self.detector = detector
        self.bg_id = bg_id

        self.det_score_thresh = float(det_score_thresh)
        self.det_top_k = int(det_top_k)
        self.det_nms_iou = float(det_nms_iou)
        self.det_min_box_size = float(det_min_box_size)

        self.density_grid = int(density_grid)
        self.density_factor = float(density_factor)
        self.avg_gt_per_image = int(avg_gt_per_image)

        self.vocab_size = vocab_size
        self.use_context = bool(use_context)
        self.context_block_gap_factor = float(context_block_gap_factor)

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
            pool_output_size=roi_pool_output_size,
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
            residual_scale=residual_scale_init,
        )

        self.roi_order = ROIReadingOrder(line_merge_thresh_ratio=0.6)

        # --- Pretrained ROI crop encoder (EfficientNet-B0, Clanuwat VGG-16 equivalent) ---
        # Fused via concat+project rather than simple addition so the network can
        # learn to weight UNet features vs. ImageNet features independently.
        if use_crop_encoder:
            self.roi_crop_encoder: Optional[ROICropEncoder] = ROICropEncoder(
                out_dim=roi_feat_dim,
                crop_size=crop_encoder_size,
                freeze_encoder=freeze_crop_encoder,
                pretrained=True,
                chunk_size=crop_encoder_chunk_size,
            )
            # Projects [roi_feats || crop_feats] -> roi_feat_dim
            _fusion = nn.Linear(roi_feat_dim * 2, roi_feat_dim, bias=True)
            with torch.no_grad():
                nn.init.zeros_(_fusion.weight)
                nn.init.zeros_(_fusion.bias)
                # First roi_feat_dim columns → identity, second half → zero
                _fusion.weight[:, :roi_feat_dim].copy_(torch.eye(roi_feat_dim))
            self.crop_fusion = _fusion

        else:
            self.roi_crop_encoder = None
            self.crop_fusion = None

        self.roi_tokens = ROITokenProjector(
            roi_feat_dim=roi_feat_dim,
            token_dim=token_dim,
            hidden_dim=token_hidden_dim,
            dropout=dropout,
            use_score_branch=bool(token_use_score_branch),
        )

        # --- Neighbor feature projection (dakuten discriminator) ---
        # Projects 6 relative-geometry scalars (prev/next neighbor Δcx, Δcy, log_area_ratio)
        # into token space and adds to token_feats before the BiGRU.
        # Zero-init so it starts neutral and learns incrementally.
        _nbr = nn.Linear(6, token_dim, bias=True)
        nn.init.zeros_(_nbr.weight)
        nn.init.zeros_(_nbr.bias)
        self.neighbor_proj: nn.Linear = _nbr

        # --- Context encoder (optional) ---

        if self.use_context:
            self.context_encoder = ROIContextEncoder(
                in_dim=token_dim,
                hidden_dim=context_hidden_dim,
                out_dim=context_hidden_dim,
                num_layers=int(context_num_layers),
                dropout=dropout,
                mode=context_mode,
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

        # --- Script-type auxiliary head ---
        # Predicts one of 4 script classes per ROI: hiragana / katakana / kanji / other.
        # Forces context features to encode script boundaries, directly targeting
        # hiragana↔katakana confusions (は↔ハ, み↔ミ).
        # Linear only — shares no parameters with the main classifier so the
        # signal is injected at the representation level, not the output level.
        self.script_classifier: nn.Linear = nn.Linear(classifier_in_dim, 4, bias=True)
        nn.init.normal_(self.script_classifier.weight, std=0.01)
        nn.init.zeros_(self.script_classifier.bias)

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
            density_grid=self.density_grid,
            density_factor=self.density_factor,
            avg_gt_per_image=self.avg_gt_per_image,
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

        # --- ROI pooling + crop encoder in parallel on separate CUDA streams ---
        # Both branches only need (coarse_boxes_list, images) which are already
        # available. ROI pool reads proj_feats; crop encoder reads raw images.
        # They are data-independent until the fusion step.
        _crop_stream: Optional[torch.cuda.Stream] = None
        _crop_feats: Optional[torch.Tensor] = None

        if self.roi_crop_encoder is not None and self.crop_fusion is not None:
            # Pre-pad proposal boxes into (B, T, 4) / roi_mask so the crop encoder
            # can start before roi_pool finishes (roi_pool produces the same layout).
            _t_max = max((b.size(0) for b in coarse_boxes_list), default=0)
            _bsz   = len(coarse_boxes_list)
            _pre_boxes = images.new_zeros((_bsz, _t_max, 4))
            _pre_mask  = torch.zeros((_bsz, _t_max), dtype=torch.bool, device=images.device)
            for _b, _boxes_b in enumerate(coarse_boxes_list):
                _n = _boxes_b.size(0)
                if _n > 0:
                    _pre_boxes[_b, :_n] = _boxes_b.to(images.dtype)
                    _pre_mask[_b, :_n]  = True

            if torch.cuda.is_available():
                _crop_stream = torch.cuda.Stream()
                _crop_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(_crop_stream):
                    _crop_feats = self.roi_crop_encoder(
                        images=images,
                        roi_boxes=_pre_boxes,
                        roi_mask=_pre_mask,
                    )
            else:
                # CPU: no streams needed, run sequentially
                _crop_feats = self.roi_crop_encoder(
                    images=images,
                    roi_boxes=_pre_boxes,
                    roi_mask=_pre_mask,
                )

        roi_out = self.roi_pool(
            feat_2d=proj_feats,
            proposal_boxes_list=coarse_boxes_list,
            proposal_scores_list=coarse_scores_list,
            image_size=image_size,
        )

        roi_feats_for_refine = roi_out["roi_feats"]
        if _crop_stream is not None and _crop_feats is not None and self.crop_fusion is not None:
            # Wait for crop encoder stream before fusing (CUDA only)
            if torch.cuda.is_available():
                torch.cuda.current_stream().wait_stream(_crop_stream)
            # Align T dimension if roi_pool used a different cap (safety, rarely fires)
            T_out = roi_out["roi_feats"].size(1)
            T_pre = _crop_feats.size(1)
            if T_out < T_pre:
                _crop_feats = _crop_feats[:, :T_out, :]
            elif T_out > T_pre:
                _crop_feats = torch.cat(
                    [_crop_feats,
                     _crop_feats.new_zeros((_crop_feats.size(0), T_out - T_pre, _crop_feats.size(2)))],
                    dim=1,
                )
            roi_feats_for_refine = self.crop_fusion(
                torch.cat([roi_feats_for_refine, _crop_feats], dim=-1)
            )

        # --- ROI refinement ---
        refine_out = self.roi_refine(
            roi_feats=roi_feats_for_refine,
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
            refined_feats=ordered["refined_feats"],
            refined_boxes=ordered["boxes"],
            refine_scores=ordered["refine_scores"],
            roi_mask=ordered["mask"],
            image_size=image_size,
        )

        # --- Neighbor features (relative geometry to adjacent reading-order neighbors) ---
        # Adds dakuten-discriminating signal: a tiny neighbor (log_area_ratio << 0)
        # at a small spatial offset is a strong cue for voiced marks (び vs ひ, etc.).
        nbr_feats = _compute_neighbor_features(ordered["boxes"], ordered["mask"])
        token_feats = token_feats + self.neighbor_proj(nbr_feats)

        # --- Context encoder (optional GRU, per-block) ---
        if self.use_context and self.context_encoder is not None:
            # Normalized (cx, cy) from refined boxes in sorted order.
            # ordered["boxes"] is (B, T, 4) in image xyxy coordinates.
            sorted_boxes = ordered["boxes"]
            cx = (sorted_boxes[..., 0] + sorted_boxes[..., 2]) * 0.5 / float(w_img)
            cy = (sorted_boxes[..., 1] + sorted_boxes[..., 3]) * 0.5 / float(h_img)
            spatial_coords = torch.stack([cx, cy], dim=-1)  # (B, T, 2)

            # Compute block IDs so the GRU resets at text-block boundaries.
            # Characters from unrelated blocks (main text vs. marginal annotations)
            # should not share context — the GRU hidden state is reset per block.
            block_ids = _compute_block_ids(
                ordered["boxes"], ordered["mask"], self.context_block_gap_factor
            )

            context_feats, context_mask = self.context_encoder(
                seq=token_feats,
                mask=token_mask,
                spatial_coords=spatial_coords,
                refine_scores=ordered["refine_scores"],
                block_ids=block_ids,
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
            "col_ids": ordered.get("col_ids", None),
            "isolation_mask": ordered.get("isolation_mask", None),
            "furigana_mask":  ordered.get("furigana_mask",  None),
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

        char_logits   = self.classifier(encoded["context_feats"])         # (B, T, V)
        script_logits = self.script_classifier(encoded["context_feats"])   # (B, T, 4)

        return {**encoded, "char_logits": char_logits, "script_logits": script_logits}

    @torch.no_grad()
    def transcribe(
        self,
        images: torch.Tensor,
        orientations: List[str],
        vocab,
        score_thresh: float = 0.0,
        bg_score_gate: float = KURONET_BG_SCORE_GATE,
    ) -> List[str]:
        """
        Inference: returns a transcription string per image.

        For each image:
          1. Run forward pass
          2. Argmax over char_logits to get predicted char IDs
          3. Filter by roi_mask (and optionally refine_score threshold)
          4. Score-gated BG suppression: if sigmoid(refine_score) > bg_score_gate
             AND pred == bg_id, override with best non-BG class.
          5. ROIs are already in reading order from sort_batch
          6. Decode IDs with vocab -> list of chars -> join as string

        Args:
            score_thresh: minimum refine_score (sigmoid) to include an ROI.
                          0.0 = include all valid ROIs (roi_mask only).
            bg_score_gate: minimum refine_score (sigmoid) above which a BG
                           prediction is suppressed and replaced with the best
                           non-BG class.  0.0 disables the gate.
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
                if refine_scores is None:
                    raise ValueError("refine_scores must be provided for score_thresh filtering.")
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

            # Score-gated BG suppression: a high-quality proposal (high refine_score)
            # predicted as BG is likely a real character misclassified.
            # Override its prediction with the best non-BG class.
            if (
                self.bg_id is not None
                and bg_score_gate > 0.0
                and sort_indices is not None
                and refine_scores is not None
            ):
                si_b = sort_indices[b]
                orig_positions = si_b[valid_positions]   # map sorted -> original order
                raw_scores = refine_scores[b].index_select(0, orig_positions)
                roi_quality = torch.sigmoid(raw_scores)  # (N_valid,)

                suppressed: List[int] = []
                for i, p_id in enumerate(pred_ids):
                    if p_id == self.bg_id and float(roi_quality[i]) > bg_score_gate:
                        lgt = logits_b[i].clone()
                        lgt[self.bg_id] = float("-inf")
                        suppressed.append(int(lgt.argmax().item()))
                    else:
                        suppressed.append(p_id)
                pred_ids = suppressed

            # Filter remaining background predictions
            if self.bg_id is not None:
                pred_ids = [p for p in pred_ids if p != self.bg_id]

            chars = vocab.decode(pred_ids, remove_special=True)
            results.append("".join(chars))

        return results
