# model/kuronet/decoder/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def build_roi_attention_mask(attention_weights, boxes, image_size, enc_grid_size, top_k=5, threshold=0.1):
    """
    Build ROI mask from attention weights to predict bounding boxes.
    
    Args:
        attention_weights: (B, T_dec, T_enc) attention from decoder
        boxes: Ground truth boxes (B, N, 4) for training supervision
        image_size: (H, W) original image size
        enc_grid_size: (H_enc, W_enc) encoder grid dimensions
        top_k: Number of top attention positions to consider per timestep
        threshold: Minimum attention weight to consider
        
    Returns:
        predicted_boxes: (B, T_dec, 4) predicted boxes from attention
        roi_mask: (B, T_dec, T_enc) mask highlighting ROI regions
    """
    B, T_dec, T_enc = attention_weights.shape
    H_enc, W_enc = enc_grid_size
    H_img, W_img = image_size
    
    # Reshape attention to 2D grid: (B, T_dec, T_enc) -> (B, T_dec, H_enc, W_enc)
    attn_2d = attention_weights.view(B, T_dec, H_enc, W_enc)
    
    # For each decoding step, find bounding box from attention peaks
    predicted_boxes = []
    roi_masks = []
    
    for t in range(T_dec):
        attn_t = attn_2d[:, t]  # (B, H_enc, W_enc)
        
        # Find top-k positions
        attn_flat = attn_t.view(B, -1)
        topk_vals, topk_idx = torch.topk(attn_flat, k=min(top_k, T_enc), dim=1)
        
        # Convert to 2D coordinates
        y_coords = (topk_idx // W_enc).float()
        x_coords = (topk_idx % W_enc).float()
        
        # Compute bounding box from top-k positions (min/max)
        # Scale to image coordinates
        scale_y = H_img / H_enc
        scale_x = W_img / W_enc
        
        x1 = (x_coords.min(dim=1)[0] * scale_x).clamp(0, W_img)
        y1 = (y_coords.min(dim=1)[0] * scale_y).clamp(0, H_img)
        x2 = ((x_coords.max(dim=1)[0] + 1) * scale_x).clamp(0, W_img)
        y2 = ((y_coords.max(dim=1)[0] + 1) * scale_y).clamp(0, H_img)
        
        boxes_t = torch.stack([x1, y1, x2, y2], dim=1)  # (B, 4)
        predicted_boxes.append(boxes_t)
        
        # Create ROI mask: threshold attention
        mask_t = (attn_t > threshold).float()
        roi_masks.append(mask_t.view(B, -1))  # (B, T_enc)
    
    predicted_boxes = torch.stack(predicted_boxes, dim=1)  # (B, T_dec, 4)
    roi_masks = torch.stack(roi_masks, dim=1)  # (B, T_dec, T_enc)
    
    return predicted_boxes, roi_masks


class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.project_dec = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: (B, dec_dim)
        # enc_outputs: (B, T_enc, enc_dim)
        dec_proj = self.project_dec(dec_hidden).unsqueeze(2)  # (B, enc_dim, 1)
        scores = torch.bmm(enc_outputs, dec_proj).squeeze(2)  # (B, T_enc)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-1e9"))
        attn = F.softmax(scores, dim=1)  # (B, T_enc)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # (B, enc_dim)
        return context, attn


class SeqDecoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        enc_dim,
        num_layers=1,
        init_from_encoder=True,
        sampling_method="multinomial",  # "argmax"|"multinomial"|"gumbel"
        use_roi_attention=False  # NEW: enable ROI-based box prediction
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + enc_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attn = LuongAttention(enc_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.init_from_encoder = init_from_encoder
        self.sampling_method = sampling_method
        self.use_roi_attention = use_roi_attention
        
        if init_from_encoder:
            self.enc2hidden = nn.Sequential(
                nn.Linear(enc_dim, hidden_dim),
                nn.Tanh()
            )
        
        # Optional: box prediction head from attention
        if use_roi_attention:
            self.box_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4)  # predict (x1, y1, x2, y2)
            )

    def _sample_from_logits(self, logits):
        # logits: (B, V)
        if self.sampling_method == "argmax":
            return logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)
        if self.sampling_method == "multinomial":
            return torch.multinomial(probs, num_samples=1).squeeze(1)
        if self.sampling_method == "gumbel":
            g = -torch.log(-torch.log(torch.rand_like(probs)))
            return (torch.log(probs + 1e-12) + g).argmax(dim=-1)
        return logits.argmax(dim=-1)

    def forward(
        self,
        input_seq,
        enc_outputs,
        enc_mask=None,
        hidden=None,
        teacher_forcing_ratio=0.0,
        targets=None,
        sos_id=None,
        eos_id=None,
        max_len=None,
    ):
        device = enc_outputs.device
        # Batch size: derive from input_seq if provided, else from encoder outputs
        B = input_seq.size(0) if input_seq is not None else enc_outputs.size(0)

        if targets is not None:
            T_dec = targets.size(1)
        elif input_seq is not None:
            T_dec = input_seq.size(1)
        elif max_len is not None:
            T_dec = max_len
        else:
            raise ValueError("Provide targets or input_seq or max_len to determine decoding length")

        if hidden is None:
            if self.init_from_encoder:
                if enc_mask is not None:
                    lengths = enc_mask.sum(dim=1).clamp(min=1).unsqueeze(1).to(enc_outputs.dtype)
                    pooled = enc_outputs.sum(dim=1) / lengths
                else:
                    pooled = enc_outputs.mean(dim=1)
                h0 = self.enc2hidden(pooled)
                hidden = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
            else:
                hidden = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)

        if input_seq is not None:
            input_tok = input_seq[:, 0]
        else:
            if sos_id is None:
                raise ValueError("sos_id must be given when input_seq is None")
            # initialize all sequences with SOS token
            input_tok = torch.full((B,), sos_id, dtype=torch.long, device=device)

        outputs = []
        attn_weights = []

        for t in range(T_dec):
            emb = self.embed(input_tok)
            dec_hidden_top = hidden[-1]
            context, attn = self.attn(dec_hidden_top, enc_outputs, mask=enc_mask)
            rnn_in = torch.cat([emb, context], dim=1).unsqueeze(1)
            out_rnn, hidden = self.rnn(rnn_in, hidden)
            logits = self.out(out_rnn.squeeze(1))

            outputs.append(logits.unsqueeze(1))
            attn_weights.append(attn.unsqueeze(1))

            use_teacher = (targets is not None) and (random.random() < teacher_forcing_ratio)
            if use_teacher:
                # mypy/linters: targets guaranteed non-None in this branch
                assert targets is not None
                next_tok = targets[:, t]
            else:
                next_tok = self._sample_from_logits(logits)

            input_tok = next_tok

            if (targets is None or teacher_forcing_ratio == 0.0) and (eos_id is not None):
                if (input_tok == eos_id).all():
                    break

        logits_cat = torch.cat(outputs, dim=1)
        attn_cat = torch.cat(attn_weights, dim=1) if len(attn_weights) > 0 else None
        
        # ROI mode: predict boxes from attention or box head
        predicted_boxes = None
        if self.use_roi_attention and attn_cat is not None:
            if hasattr(self, 'box_head'):
                # Predict boxes from decoder hidden states
                # Use final hidden state per timestep
                box_logits = []
                for t in range(len(outputs)):
                    box_pred = self.box_head(hidden[-1])  # (B, 4)
                    box_logits.append(box_pred.unsqueeze(1))
                predicted_boxes = torch.cat(box_logits, dim=1)  # (B, T_dec, 4)
            # Could also use build_roi_attention_mask here if needed
        
        return logits_cat, hidden, attn_cat, predicted_boxes
