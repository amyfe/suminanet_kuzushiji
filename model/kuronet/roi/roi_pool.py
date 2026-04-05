"""
ToDo:
A:
Annahme isotrope Skalierung absichtlich hart und im Fehlerfall explizit abstürzen.
Warum? Weil stillschweigende Geometriefehler in ROI-Pipelines sehr teuer sind.

B:
Die empty_token-Lösung ist technisch praktisch, aber semantisch nicht schön.
Für den Start okay. Später überlegen, leere Samples expliziter zu behandeln.


Extrahiert pro coarse ROI lokale visuelle Features aus der 2D-Featuremap.
Aufgabe:

Extrahiert pro coarse ROI lokale visuelle Features aus der 2D-Featuremap.

Rolle:

Das ist der saubere Nachfolger des brauchbaren Teils aus deinem alten ROISequenceEncoder.

Input:
projected feature map (B, C, Hf, Wf)
proposal boxes pro Bild in Bildkoordinaten
image size
Output

Pro ROI:

lokaler Featurevektor
optional lokales ROI-Featurepatch
Proposal-Maskierung/Padding


Outputs: 
roi_feats_padded: (B, T, D)
roi_local_maps:   optional (B, T, C, Hp, Wp)
roi_mask:         (B, T)
roi_boxes_padded: (B, T, 4)
roi_scores:       (B, T)
"""
# model/kuronet/roi/roi_pool.py

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.ops import roi_align as tv_roi_align

from model.kuronet.utils import make_gn


class ROIPoolEncoder(nn.Module):
    """
    Extract local ROI features from projected 2D feature maps.

    Input:
        feat_2d: (B, C, Hf, Wf)
        proposal_boxes_list: list of length B, each tensor (N_i, 4) in image coordinates
        proposal_scores_list: list of length B, each tensor (N_i,)

    Output:
        dict with padded ROI tensors:
            roi_feats:   (B, T, D)
            roi_local:   (B, T, C_mid, Hp, Wp)
            roi_boxes:   (B, T, 4)
            roi_scores:  (B, T)
            roi_mask:    (B, T)
    """

    def __init__(
        self,
        in_channels: int,
        roi_size: Tuple[int, int] = (8, 6),
        conv_channels: int | None = None,
        out_dim: int = 256,
        dropout: float = 0.1,
        predict_aux_logits: bool = False,
        vocab_size: int | None = None,
    ):
        super().__init__()

        if predict_aux_logits and (vocab_size is None or vocab_size <= 0):
            raise ValueError("If predict_aux_logits=True, vocab_size must be a positive integer.")

        if conv_channels is None:
            conv_channels = in_channels

        self.roi_size = roi_size
        self.out_dim = out_dim
        self.conv_channels = conv_channels

        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            make_gn(conv_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            make_gn(conv_channels),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * roi_size[0] * roi_size[1], out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.empty_token = nn.Parameter(torch.zeros(1, out_dim))
        self.predict_aux_logits = bool(predict_aux_logits)
        self.vocab_size = int(vocab_size) if vocab_size is not None else None

        self.aux_head = None
        if self.predict_aux_logits:
            if vocab_size is None:
                raise ValueError("If predict_aux_logits=True, vocab_size must be a positive integer.")
            aux_vocab_size: int = vocab_size
            self.aux_head = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(out_dim, aux_vocab_size),
            )

    def forward(
        self,
        feat_2d: torch.Tensor,                    # (B, C, Hf, Wf)
        proposal_boxes_list: List[torch.Tensor], # list[(N_i,4)]
        proposal_scores_list: List[torch.Tensor],# list[(N_i,)]
        image_size: Tuple[int, int],
    ) -> dict:
        if feat_2d.dim() != 4:
            raise ValueError(f"feat_2d must have shape (B, C, Hf, Wf), got {tuple(feat_2d.shape)}")

        device = feat_2d.device
        dtype = feat_2d.dtype
        bsz, _, hf, wf = feat_2d.shape
        h_img, w_img = image_size

        if len(proposal_boxes_list) != bsz:
            raise ValueError(
                f"proposal_boxes_list length {len(proposal_boxes_list)} does not match batch size {bsz}"
            )
        if len(proposal_scores_list) != bsz:
            raise ValueError(
                f"proposal_scores_list length {len(proposal_scores_list)} does not match batch size {bsz}"
            )

        spatial_scale_w = wf / float(w_img)
        spatial_scale_h = hf / float(h_img)
        if abs(spatial_scale_w - spatial_scale_h) > 1e-6:
            # In your current setup IMAGE_SIZE is fixed, so scales should usually match.
            # We still make the mismatch explicit instead of silently assuming isotropy.
            raise ValueError(
                f"Non-uniform spatial scale detected: "
                f"scale_h={spatial_scale_h:.6f}, scale_w={spatial_scale_w:.6f}. "
                "Current ROIPoolEncoder assumes isotropic image-to-feature scaling."
            )
        spatial_scale = spatial_scale_w

        rois = []
        roi_scores_cat = []
        counts = []

        for b in range(bsz):
            boxes_b = proposal_boxes_list[b]
            scores_b = proposal_scores_list[b]

            if boxes_b is None:
                boxes_b = torch.empty((0, 4), device=device, dtype=dtype)
            if scores_b is None:
                scores_b = torch.empty((0,), device=device, dtype=dtype)

            boxes_b = boxes_b.to(device=device, dtype=dtype)
            scores_b = scores_b.to(device=device, dtype=dtype)

            if boxes_b.numel() == 0:
                counts.append(0)
                continue

            valid = (
                (boxes_b[:, 2] - boxes_b[:, 0] > 1e-3) &
                (boxes_b[:, 3] - boxes_b[:, 1] > 1e-3)
            )
            boxes_b = boxes_b[valid]
            scores_b = scores_b[valid]

            counts.append(int(boxes_b.size(0)))

            if boxes_b.numel() == 0:
                continue

            batch_idx = torch.full((boxes_b.size(0), 1), float(b), device=device, dtype=dtype)
            rois_b = torch.cat([batch_idx, boxes_b], dim=1)  # (N_b, 5)

            rois.append(rois_b)
            roi_scores_cat.append(scores_b)

        if len(rois) > 0:
            rois_cat = torch.cat(rois, dim=0)                           # (N_all, 5)
            roi_scores_flat = torch.cat(roi_scores_cat, dim=0)          # (N_all,)

            pooled = tv_roi_align(  # type: ignore[operator]
                feat_2d,
                rois_cat,
                output_size=self.roi_size,
                spatial_scale=spatial_scale,
                aligned=True,
            )                                                           # (N_all, C, Hr, Wr)

            roi_local_flat = self.roi_conv(pooled)                      # (N_all, C_mid, Hr, Wr)
            roi_feats_flat = self.proj(roi_local_flat)                  # (N_all, D)
            roi_boxes_flat = rois_cat[:, 1:]                            # (N_all, 4)
        else:
            roi_scores_flat = torch.empty((0,), device=device, dtype=dtype)
            roi_local_flat = torch.empty(
                (0, self.conv_channels, self.roi_size[0], self.roi_size[1]),
                device=device,
                dtype=dtype,
            )
            roi_feats_flat = torch.empty((0, self.out_dim), device=device, dtype=dtype)
            roi_boxes_flat = torch.empty((0, 4), device=device, dtype=dtype)

        t_max = max(1, max(counts) if counts else 0)

        roi_feats = torch.zeros((bsz, t_max, self.out_dim), device=device, dtype=dtype)
        roi_boxes = torch.zeros((bsz, t_max, 4), device=device, dtype=dtype)
        roi_scores = torch.zeros((bsz, t_max), device=device, dtype=dtype)
        roi_mask = torch.zeros((bsz, t_max), device=device, dtype=torch.bool)
        roi_local = torch.zeros(
            (bsz, t_max, self.conv_channels, self.roi_size[0], self.roi_size[1]),
            device=device,
            dtype=dtype,
        )

        offset = 0
        for b, cnt in enumerate(counts):
            if cnt > 0:
                roi_feats[b, :cnt] = roi_feats_flat[offset:offset + cnt]
                roi_boxes[b, :cnt] = roi_boxes_flat[offset:offset + cnt]
                roi_scores[b, :cnt] = roi_scores_flat[offset:offset + cnt]
                roi_local[b, :cnt] = roi_local_flat[offset:offset + cnt]
                roi_mask[b, :cnt] = True
                offset += cnt
            else:
                roi_feats[b, 0] = self.empty_token[0]
                roi_mask[b, 0] = True
                # Important caveat:
                # this keeps tensor shapes valid, but semantically empty samples should later
                # still be handled carefully via dedicated valid-instance logic if needed.

        return {
            "roi_feats": roi_feats,     # (B, T, D)
            "roi_local": roi_local,     # (B, T, C_mid, Hr, Wr)
            "roi_boxes": roi_boxes,     # (B, T, 4)
            "roi_scores": roi_scores,   # (B, T)
            "roi_mask": roi_mask,       # (B, T)
            "aux_logits": self.aux_head(roi_feats) if self.aux_head is not None else None,
        }
