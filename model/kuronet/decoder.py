"""Sequence decoder (RNN) that can be trained with teacher forcing.
This is an optional component for sequence modeling of reading-order labels.
"""
import random
import torch
import torch.nn as nn


class SeqDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, hidden=None, teacher_forcing_ratio=0.0, targets=None):
        # input_seq: (B, T) initial tokens (e.g. sos tokens or previous outputs)
        # if teacher_forcing_ratio>0 and targets provided, use teacher forcing
        B, T = input_seq.shape
        embeddings = self.embed(input_seq)
        outputs, hn = self.rnn(embeddings, hidden)
        logits = self.out(outputs)
        return logits, hn

    def generate_with_teacher_forcing(self, sos_token, max_len, encoder_ctx, teacher_forcing_ratio=0.5, targets=None):
        # Simple step-by-step decode to demonstrate teacher forcing
        device = next(self.parameters()).device
        input_tok = torch.full((encoder_ctx.size(0),), sos_token, dtype=torch.long, device=device)
        hidden = None
        outputs = []
        for t in range(max_len):
            emb = self.embed(input_tok).unsqueeze(1)  # (B,1,embed)
            out, hidden = self.rnn(emb, hidden)
            logits = self.out(out.squeeze(1))  # (B, vocab)
            outputs.append(logits.unsqueeze(1))
            # decide next input
            if targets is not None and random.random() < teacher_forcing_ratio:
                input_tok = targets[:, t]
            else:
                input_tok = logits.argmax(dim=-1)
        return torch.cat(outputs, dim=1)