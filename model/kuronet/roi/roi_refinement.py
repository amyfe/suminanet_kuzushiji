"""Herzstück von Option "Hybrid"

Aufgabe

Nimmt coarse ROI-Features und coarse boxes und lernt:

refined boxes
ROI quality / validity
optional local character logits

Input
roi_feats_padded: (B, T, D)
roi_boxes_padded: (B, T, 4)
roi_mask: (B, T)
optional detector scores

Output
refined_boxes:   (B, T, 4)
refine_scores:   (B, T)      # wie vertrauenswürdig / gültig ist die ROI
refined_feats:   (B, T, D)
aux_logits:      optional (B, T, V_local_or_vocab)
Was dieses Modul tun sollte

Mindestens:

Box delta prediction relativ zur coarse box
quality head
feature refinement

Optional später:

split/merge hint
ambiguity score
Wichtig

Refinement sollte nicht absolute Boxen direkt regressieren, sondern eher:

dx, dy, dw, dh relativ zur coarse ROI

Designentscheidungen/ ToDos:

1. Relative Boxdeltas statt absolute Boxen
    Das ist für Stabilität fast sicher besser.

2. refine_scores als Logits
    Ich würde sie als rohe Logits ausgeben, nicht direkt sigmoidieren.
    Dann kannst du im Loss sauber BCEWithLogitsLoss verwenden.

3. Noch keine Split/Merge-Entscheidung

Das Modul verfeinert im Moment:

Position
Größe
Qualität

aber spaltet keine ROIs auf und merged auch nichts.
Das ist Absicht. Für eine erste echte Option-C-Version reicht das.
Split/Merge wäre der nächste Forschungs-Schritt, nicht der erste.

"""



# model/kuronet/roi/roi_refinement.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIRefinementHead(nn.Module):
    """
    Refine coarse ROI proposals into better-aligned ROI representations.

    Inputs:
        roi_feats:   (B, T, D)
        roi_boxes:   (B, T, 4)   coarse boxes in image coordinates
        roi_scores:  (B, T)      detector proposal scores
        roi_mask:    (B, T)      valid ROI mask
        image_size:  (H, W)

    Outputs:
        refined_feats:   (B, T, D)
        refined_boxes:   (B, T, 4)
        refine_scores:   (B, T)      learned validity / quality
        aux_logits:      optional (B, T, vocab_size)
        box_deltas:      (B, T, 4)
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        predict_aux_logits: bool = False,
        vocab_size: int | None = None,
    ):
        super().__init__()

        if predict_aux_logits and (vocab_size is None or vocab_size <= 0):
            raise ValueError("If predict_aux_logits=True, vocab_size must be a positive integer.")

        self.predict_aux_logits = predict_aux_logits
        self.vocab_size = vocab_size

        # geometry: cx, cy, bw, bh (normalized)
        self.geom_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # detector score embedding
        self.score_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # fuse visual + geometry + detector confidence
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # box delta prediction relative to coarse box
        # predicts normalized deltas: dx, dy, dw, dh
        self.box_delta_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

        # learned ROI validity / quality
        self.refine_score_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        if self.predict_aux_logits:
            self.aux_head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, vocab_size),
            )
        else:
            self.aux_head = None

    @staticmethod
    def _boxes_to_geom(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """
        Convert xyxy boxes to normalized geometry features.

        Args:
            boxes: (B, T, 4)
            image_size: (H, W)

        Returns:
            geom: (B, T, 4) = [cx, cy, bw, bh], normalized to image size
        """
        h_img, w_img = image_size

        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]

        cx = (x1 + x2) * 0.5 / max(float(w_img), 1.0)
        cy = (y1 + y2) * 0.5 / max(float(h_img), 1.0)
        bw = (x2 - x1).clamp_min(1e-6) / max(float(w_img), 1.0)
        bh = (y2 - y1).clamp_min(1e-6) / max(float(h_img), 1.0)

        return torch.stack([cx, cy, bw, bh], dim=-1)

    @staticmethod
    def _apply_box_deltas(
        boxes: torch.Tensor,      # (B, T, 4)
        deltas: torch.Tensor,     # (B, T, 4) = dx, dy, dw, dh
        image_size: tuple[int, int],
        delta_scale_xy: float = 0.25,
        delta_scale_wh: float = 0.20,
        min_box_size: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply relative box deltas to coarse boxes.

        Parameterization:
            dx, dy are relative to current box width/height
            dw, dh are multiplicative log-space-like updates via exp(clamped)

        This is more stable than predicting absolute image-space boxes directly.
        """
        h_img, w_img = image_size

        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1).clamp_min(min_box_size)
        bh = (y2 - y1).clamp_min(min_box_size)

        dx = deltas[..., 0].tanh() * delta_scale_xy
        dy = deltas[..., 1].tanh() * delta_scale_xy
        dw = deltas[..., 2].tanh() * delta_scale_wh
        dh = deltas[..., 3].tanh() * delta_scale_wh

        new_cx = cx + dx * bw
        new_cy = cy + dy * bh
        new_bw = bw * torch.exp(dw)
        new_bh = bh * torch.exp(dh)

        new_x1 = new_cx - 0.5 * new_bw
        new_y1 = new_cy - 0.5 * new_bh
        new_x2 = new_cx + 0.5 * new_bw
        new_y2 = new_cy + 0.5 * new_bh

        new_x1 = new_x1.clamp(0, float(w_img))
        new_y1 = new_y1.clamp(0, float(h_img))
        new_x2 = new_x2.clamp(0, float(w_img))
        new_y2 = new_y2.clamp(0, float(h_img))

        # enforce ordering and minimum size again
        x1f = torch.minimum(new_x1, new_x2)
        y1f = torch.minimum(new_y1, new_y2)
        x2f = torch.maximum(new_x1, new_x2)
        y2f = torch.maximum(new_y1, new_y2)

        x2f = torch.maximum(x2f, x1f + min_box_size).clamp(0, float(w_img))
        y2f = torch.maximum(y2f, y1f + min_box_size).clamp(0, float(h_img))

        return torch.stack([x1f, y1f, x2f, y2f], dim=-1)

    def forward(
        self,
        roi_feats: torch.Tensor,   # (B, T, D)
        roi_boxes: torch.Tensor,   # (B, T, 4)
        roi_scores: torch.Tensor,  # (B, T)
        roi_mask: torch.Tensor,    # (B, T)
        image_size: tuple[int, int],
    ) -> dict:
        if roi_feats.dim() != 3:
            raise ValueError(f"roi_feats must have shape (B, T, D), got {tuple(roi_feats.shape)}")
        if roi_boxes.dim() != 3 or roi_boxes.size(-1) != 4:
            raise ValueError(f"roi_boxes must have shape (B, T, 4), got {tuple(roi_boxes.shape)}")
        if roi_scores.dim() != 2:
            raise ValueError(f"roi_scores must have shape (B, T), got {tuple(roi_scores.shape)}")
        if roi_mask.dim() != 2:
            raise ValueError(f"roi_mask must have shape (B, T), got {tuple(roi_mask.shape)}")

        geom = self._boxes_to_geom(roi_boxes, image_size=image_size)        # (B, T, 4)
        geom_feat = self.geom_proj(geom)                                    # (B, T, H)
        score_feat = self.score_proj(roi_scores.unsqueeze(-1))              # (B, T, H)

        fused_in = torch.cat([roi_feats, geom_feat, score_feat], dim=-1)    # (B, T, D+H+H)
        refined_feats = self.fuse(fused_in)                                 # (B, T, D)

        box_deltas = self.box_delta_head(refined_feats)                     # (B, T, 4)
        refined_boxes = self._apply_box_deltas(
            boxes=roi_boxes,
            deltas=box_deltas,
            image_size=image_size,
        )                                                                   # (B, T, 4)

        refine_scores = self.refine_score_head(refined_feats).squeeze(-1)   # (B, T)

        # Mask invalid positions explicitly to avoid accidental training leakage
        refined_feats = refined_feats * roi_mask.unsqueeze(-1).to(refined_feats.dtype)
        refine_scores = refine_scores.masked_fill(~roi_mask, -1e4)
        refined_boxes = torch.where(
            roi_mask.unsqueeze(-1),
            refined_boxes,
            roi_boxes
        )

        aux_logits = None
        if self.aux_head is not None:
            aux_logits = self.aux_head(refined_feats)                       # (B, T, V)

        return {
            "refined_feats": refined_feats,
            "refined_boxes": refined_boxes,
            "refine_scores": refine_scores,
            "aux_logits": aux_logits,
            "box_deltas": box_deltas,
        }