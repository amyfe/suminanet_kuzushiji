"""Macht aus rohen und refined ROI-Features + Geometrie + Scores eine Sequenz von Token-Embeddings für den Kontextencoder.

Input
roi features (B, T, D)
refined ROI features (B, T, D)
refined boxes (B, T, 4)
refine scores (B, T)
roi_mask (B, T)
image_size


Output
token embeddings (B, T, D_tok)
token mask (B, T)

Hier: Feature + Geometrie + Score fusionieren
zB token = MLP([visual_feat ; geometry_feat ; score_feat])

Ziel: aus rohen ROI-Features, refined ROI-Features, Geometrie und Scores eine saubere Token-Sequenz machen, die in den Kontextencoder geht.

Wichtig:

kein ROI pooling mehr
kein Refinement mehr
nur Tokenrepräsentation

ToDos:
1. refine_scores als Input ist sinnvoll, aber nicht automatisch hilfreich

Wenn die Quality-Logits schlecht kalibriert sind, können sie auch schaden.
Deshalb habe ich use_score_branch=True als Option gelassen.

2. Hier wird Geometrie erneut benutzt

Das ist kein Fehler.
Refinement und Tokenisierung nutzen Geometrie auf unterschiedliche Weise:

Refinement: Box verbessern
Tokenisierung: Positions-/Strukturinformation für Kontextdecoder
3. Noch keine relative Geometrie zwischen Tokens

Das Modul nutzt nur absolute normalisierte Boxmerkmale.
Später könnte man relative Features zwischen Nachbartokens ergänzen, aber noch nicht jetzt.

"""


# model/kuronet/roi/roi_tokens.py

from __future__ import annotations

import torch
import torch.nn as nn


class ROITokenProjector(nn.Module):
    """
    Convert refined ROI representations into sequence token embeddings.

    Inputs:
        refined_feats:  (B, T, D_feat)
        refined_boxes:  (B, T, 4)
        refine_scores:  (B, T)    raw logits or scores
        roi_mask:       (B, T)
        image_size:     (H, W)

    Outputs:
        token_feats:    (B, T, D_token)
        token_mask:     (B, T)
    """

    def __init__(
        self,
        roi_feat_dim: int,
        token_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_score_branch: bool = True,
    ):
        super().__init__()

        self.use_score_branch = bool(use_score_branch)

        self.geom_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.use_score_branch:
            self.score_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            fuse_in_dim = roi_feat_dim + hidden_dim + hidden_dim
        else:
            self.score_proj = None
            fuse_in_dim = roi_feat_dim + hidden_dim

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.token_dim = token_dim

    @staticmethod
    def _boxes_to_geom(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """
        Convert xyxy boxes into geometry features: [cx, cy, log(bw), log(bh)].

        Log-space for width/height: at 256×256 a typical 10px character gives
        bw≈0.04 in linear space vs log(0.04)≈-3.2 in log space.  The larger
        magnitude gives geom_proj a clearer gradient signal to distinguish
        small (furigana/dakuten) from large characters.
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

        return torch.stack([cx, cy, bw.log(), bh.log()], dim=-1)

    def forward(
        self,
        refined_feats: torch.Tensor,   # (B, T, D_feat)  residual-enhanced ROI features
        refined_boxes: torch.Tensor,   # (B, T, 4)
        refine_scores: torch.Tensor,   # (B, T)
        roi_mask: torch.Tensor,        # (B, T)
        image_size: tuple[int, int],
        roi_feats: torch.Tensor | None = None,  # unused, kept for API compatibility
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if refined_feats.dim() != 3:
            raise ValueError(
                f"refined_feats must have shape (B, T, D), got {tuple(refined_feats.shape)}"
            )
        if refined_boxes.dim() != 3 or refined_boxes.size(-1) != 4:
            raise ValueError(
                f"refined_boxes must have shape (B, T, 4), got {tuple(refined_boxes.shape)}"
            )
        if refine_scores.dim() != 2:
            raise ValueError(
                f"refine_scores must have shape (B, T), got {tuple(refine_scores.shape)}"
            )
        if roi_mask.dim() != 2:
            raise ValueError(
                f"roi_mask must have shape (B, T), got {tuple(roi_mask.shape)}"
            )

        geom = self._boxes_to_geom(refined_boxes, image_size=image_size)
        geom_feat = self.geom_proj(geom)

        if self.use_score_branch:
            score_prob = torch.sigmoid(refine_scores.unsqueeze(-1))
            score_proj = self.score_proj
            if score_proj is None:
                raise RuntimeError("score_proj is unavailable although use_score_branch=True.")
            score_feat = score_proj(score_prob)
            fused_in = torch.cat([refined_feats, geom_feat, score_feat], dim=-1)
        else:
            fused_in = torch.cat([refined_feats, geom_feat], dim=-1)

        token_feats = self.fuse(fused_in)
        token_feats = token_feats * roi_mask.unsqueeze(-1).to(token_feats.dtype)

        return token_feats, roi_mask