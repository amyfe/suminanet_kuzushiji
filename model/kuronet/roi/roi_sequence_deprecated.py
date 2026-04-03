import torch
import torch.nn as nn
from torchvision.ops import roi_align as tv_roi_align


class ROISequenceEncoder(nn.Module):
    """Extract ordered ROI embeddings from 2D feature maps and box sequences."""

    def __init__(self, in_dim: int, roi_size=(4, 4), out_dim: int = 256, use_geo_positional_encoding: bool = False):
        super().__init__()
        self.roi_size = roi_size
        self.use_geo_positional_encoding = bool(use_geo_positional_encoding)
        pool_h = 2 if int(roi_size[0]) >= 2 else 1
        pool_w = 2 if int(roi_size[1]) >= 2 else 1
        self.pool_out = (pool_h, pool_w)
        num_groups = 8 if in_dim % 8 == 0 else 1
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, in_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.pool_out),
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim * self.pool_out[0] * self.pool_out[1], out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim
        self.empty_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, feat_2d: torch.Tensor, boxes_list, image_size):
        """
        Args:
            feat_2d: (B, C, Hf, Wf)
            boxes_list: list of length B, each tensor (N_i, 4) in image coordinates
            image_size: (H_img, W_img)
        Returns:
            seq_padded: (B, T_max, D)
            seq_mask: (B, T_max)
        """
        device = feat_2d.device
        dtype = feat_2d.dtype
        bsz, _, hf, wf = feat_2d.shape
        h_img, w_img = image_size

        if boxes_list is None:
            boxes_list = [torch.empty((0, 4), device=device, dtype=dtype) for _ in range(bsz)]

        spatial_scale = wf / float(w_img)

        rois = []
        geom_features = []
        counts = []
        for i in range(bsz):
            boxes_i = boxes_list[i]
            if boxes_i is None:
                boxes_i = torch.empty((0, 4), device=device, dtype=dtype)
            boxes_i = boxes_i.to(device=device, dtype=dtype)

            if boxes_i.numel() == 0:
                counts.append(0)
                continue

            x1 = boxes_i[:, 0].clamp(0, float(w_img))
            y1 = boxes_i[:, 1].clamp(0, float(h_img))
            x2 = boxes_i[:, 2].clamp(0, float(w_img))
            y2 = boxes_i[:, 3].clamp(0, float(h_img))
            boxes_i = torch.stack([torch.minimum(x1, x2), torch.minimum(y1, y2), x2, y2], dim=1)

            valid = (boxes_i[:, 2] - boxes_i[:, 0] > 1e-3) & (boxes_i[:, 3] - boxes_i[:, 1] > 1e-3)
            boxes_i = boxes_i[valid]
            counts.append(int(boxes_i.size(0)))

            if boxes_i.numel() == 0:
                continue

            batch_idx = torch.full((boxes_i.size(0), 1), float(i), device=device, dtype=dtype)
            rois.append(torch.cat([batch_idx, boxes_i], dim=1))
            cx = ((boxes_i[:, 0] + boxes_i[:, 2]) * 0.5) / max(float(w_img), 1.0)
            cy = ((boxes_i[:, 1] + boxes_i[:, 3]) * 0.5) / max(float(h_img), 1.0)
            bw = (boxes_i[:, 2] - boxes_i[:, 0]) / max(float(w_img), 1.0)
            bh = (boxes_i[:, 3] - boxes_i[:, 1]) / max(float(h_img), 1.0)
            geom_features.append(torch.stack([cx, cy, bw, bh], dim=1))

        if len(rois) > 0:
            rois_cat = torch.cat(rois, dim=0)
            roi_feats = tv_roi_align(  # type: ignore[operator]
                feat_2d,
                rois_cat,
                output_size=self.roi_size,
                spatial_scale=spatial_scale,
                aligned=True,
            )
            roi_emb = self.roi_conv(roi_feats).flatten(1)
            roi_emb = self.proj(roi_emb)
            if self.use_geo_positional_encoding and len(geom_features) > 0:
                geom_cat = torch.cat(geom_features, dim=0).to(device=device, dtype=roi_emb.dtype)
                roi_emb = roi_emb + self.pos_proj(geom_cat)
        else:
            roi_emb = torch.empty((0, self.out_dim), device=device, dtype=dtype)

        t_max = max(1, max(counts) if counts else 0)
        d_out = self.out_dim
        seq_padded = torch.zeros((bsz, t_max, d_out), device=device, dtype=roi_emb.dtype if roi_emb.numel() > 0 else dtype)
        seq_mask = torch.zeros((bsz, t_max), device=device, dtype=torch.bool)

        offset = 0
        for i, cnt in enumerate(counts):
            if cnt > 0:
                seq_padded[i, :cnt] = roi_emb[offset: offset + cnt]
                seq_mask[i, :cnt] = True
                offset += cnt
            else:
                seq_padded[i, 0] = self.empty_token.to(device=device, dtype=seq_padded.dtype)[0, 0]
                seq_mask[i, 0] = True

        return seq_padded, seq_mask


