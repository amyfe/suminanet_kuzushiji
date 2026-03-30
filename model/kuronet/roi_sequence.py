import torch
import torch.nn as nn
from torchvision.ops import roi_align


class ROISequenceEncoder(nn.Module):
    """Extract ordered ROI embeddings from 2D feature maps and box sequences."""

    def __init__(self, in_dim: int, roi_size=(4, 4), out_dim: int = 256):
        super().__init__()
        self.roi_size = roi_size
        num_groups = 8 if in_dim % 8 == 0 else 1
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, in_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
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

        if len(rois) > 0:
            rois_cat = torch.cat(rois, dim=0)
            roi_feats = roi_align(
                feat_2d,
                rois_cat,
                output_size=self.roi_size,
                spatial_scale=spatial_scale,
                aligned=True,
            )
            roi_emb = self.roi_conv(roi_feats).flatten(1)
            roi_emb = self.proj(roi_emb)
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


class ROIContextEncoder(nn.Module):
    """Contextualize ROI token sequence before decoding."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor):
        lengths = mask.sum(dim=1).clamp(min=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq.size(1),
        )
        out = self.proj(out)
        return out, mask