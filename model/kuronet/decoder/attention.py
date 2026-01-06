# model/kuronet/decoder/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def _clamp_boxes(boxes, H_img, W_img):
    """Clamp boxes to image bounds and enforce x1<x2, y1<y2."""
    if boxes is None:
        return None
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1 = x1.clamp(0, W_img)
    y1 = y1.clamp(0, H_img)
    x2 = x2.clamp(0, W_img)
    y2 = y2.clamp(0, H_img)
    # Ensure proper ordering
    x1 = torch.minimum(x1, x2)
    y1 = torch.minimum(y1, y2)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _attention_weighted_centroids(attn, enc_mask, H_img, W_img):
    """
    Compute attention-weighted 1D centroids (seq over width) and expand to boxes.
    Assumes encoder sequence corresponds to width positions (horizontal reading).
    """
    B, T_dec, T_enc = attn.shape
    # Apply mask if provided
    if enc_mask is not None:
        # enc_mask: (B, T_enc)
        mask = enc_mask.unsqueeze(1).expand(-1, T_dec, -1)
        attn = attn * mask.float()
    attn_sum = attn.sum(dim=2, keepdim=True).clamp(min=1e-6)
    attn_norm = attn / attn_sum

    positions = torch.arange(T_enc, device=attn.device).float()  # 0..T_enc-1
    x_cent = (attn_norm * positions).sum(dim=2)  # (B, T_dec)

    cell_w = W_img / float(T_enc)
    x1 = (x_cent * cell_w - 0.5 * cell_w).clamp(0, W_img)
    x2 = (x_cent * cell_w + 0.5 * cell_w).clamp(0, W_img)
    # Without vertical info from encoder sequence, cover full height
    y1 = torch.zeros_like(x1)
    y2 = torch.full_like(x1, H_img)

    return torch.stack([x1, y1, x2, y2], dim=-1)  # (B, T_dec, 4)


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
            # Use finite negative to avoid FP16 overflow
            neg_fill = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask.bool(), neg_fill)
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
        use_roi_attention=False,  # NEW: enable ROI-based box prediction
        roi_blend_alpha=0.7,
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
        self.roi_blend_alpha = roi_blend_alpha
        
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
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        # If distribution collapsed, fall back to uniform to avoid CUDA assert
        probs = torch.where(
            probs_sum > 0,
            probs / probs_sum,
            probs.new_full(probs_sum.shape, 1.0 / probs.size(-1))
        )

        if self.sampling_method == "multinomial":
            return torch.multinomial(probs, num_samples=1).squeeze(1)
        if self.sampling_method == "gumbel":
            g = -torch.log(-torch.log(torch.rand_like(probs)))
            return (torch.log(probs + 1e-12) + g).argmax(dim=-1)
        return probs.argmax(dim=-1)

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
        image_size=(512, 512),
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
        dec_hidden_states = []  # store per-step decoder top hidden for box head

        for t in range(T_dec):
            emb = self.embed(input_tok)
            dec_hidden_top = hidden[-1]
            context, attn = self.attn(dec_hidden_top, enc_outputs, mask=enc_mask)
            rnn_in = torch.cat([emb, context], dim=1).unsqueeze(1)
            out_rnn, hidden = self.rnn(rnn_in, hidden)
            logits = self.out(out_rnn.squeeze(1))

            outputs.append(logits.unsqueeze(1))
            attn_weights.append(attn.unsqueeze(1))
            dec_hidden_states.append(hidden[-1].unsqueeze(1))

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
            # Box from decoder hidden states (per timestep)
            boxes_head = None
            # Attention-derived box (centroid over encoder positions)
            H_img, W_img = image_size
            if hasattr(self, 'box_head') and len(dec_hidden_states) > 0:
                boxes_head = torch.cat([self.box_head(h.squeeze(1)).unsqueeze(1) for h in dec_hidden_states], dim=1)
                # Normalize to image scale so predictions stay bounded
                boxes_head = torch.sigmoid(boxes_head)
                scale = torch.tensor([W_img, H_img, W_img, H_img], device=boxes_head.device, dtype=boxes_head.dtype)
                boxes_head = boxes_head * scale.view(1, 1, 4)

            boxes_attn = None
            # attn_cat: (B, T_dec, T_enc)
            boxes_attn = _attention_weighted_centroids(attn_cat, enc_mask, H_img, W_img)

            # Blend boxes: alpha * attn + (1-alpha) * head (when both available)
            if boxes_head is not None and boxes_attn is not None:
                predicted_boxes = self.roi_blend_alpha * boxes_attn + (1 - self.roi_blend_alpha) * boxes_head
            elif boxes_head is not None:
                predicted_boxes = boxes_head
            else:
                predicted_boxes = boxes_attn

            predicted_boxes = _clamp_boxes(predicted_boxes, H_img, W_img)
        
        return logits_cat, hidden, attn_cat, predicted_boxes
