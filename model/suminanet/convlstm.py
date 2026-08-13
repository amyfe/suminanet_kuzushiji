# model/suminanet/convlstm.py
"""
ConvLSTM implementation (cell + stacked module).
Simple, dependency-free implementation.
"""
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        # x: (B, C, H, W); h_prev/c_prev: (B, hidden, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        i, f, g, o = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1, bidirectional=False):
        super().__init__()
        # Normalize and validate hidden_dim
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        elif isinstance(hidden_dim, (list, tuple)):
            hidden_dim = list(hidden_dim)
            if len(hidden_dim) == 1 and num_layers > 1:
                hidden_dim = hidden_dim * num_layers
        else:
            raise TypeError(f"hidden_dim must be int or list/tuple of ints, got {type(hidden_dim)}")

        if len(hidden_dim) != num_layers:
            raise ValueError(
                f"Length of hidden_dim ({len(hidden_dim)}) must match num_layers ({num_layers})."
            )

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # Keep references to expected input and hidden channel sizes per layer for validation
        in_dims = [input_dim] + hidden_dim[:-1]
        self._in_dims_per_layer = in_dims
        self._hidden_dims_per_layer = hidden_dim
        for i in range(num_layers):
            self.layers.append(ConvLSTMCell(in_dims[i], hidden_dim[i], kernel_size=kernel_size))
        self.bidirectional = bidirectional

    def forward(self, x, hidden=None):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        device = x.device
        if hidden is None:
            hidden = []
            for layer_idx in range(self.num_layers):
                hidden_ch = self._hidden_dims_per_layer[layer_idx]
                h = torch.zeros(B, hidden_ch, H, W, device=device)
                c = torch.zeros(B, hidden_ch, H, W, device=device)
                hidden.append((h, c))

        layer_input = x
        last_states = []
        for layer_idx, cell in enumerate(self.layers):
            h, c = hidden[layer_idx]
            # Runtime checks to catch shape/channel mismatches early
            expected_in_ch = self._in_dims_per_layer[layer_idx]
            expected_hidden_ch = self._hidden_dims_per_layer[layer_idx]
            if layer_input.size(2) != expected_in_ch:
                raise ValueError(
                    f"ConvLSTM layer {layer_idx} expected input channels {expected_in_ch}, "
                    f"but got {layer_input.size(2)}."
                )
            if h.size(1) != expected_hidden_ch or c.size(1) != expected_hidden_ch:
                raise ValueError(
                    f"Hidden state channel mismatch at layer {layer_idx}: expected {expected_hidden_ch}, "
                    f"got h={h.size(1)}, c={c.size(1)}."
                )
            outputs = []
            for t in range(T):
                h, c = cell(layer_input[:, t], h, c)
                outputs.append(h.unsqueeze(1))  # (B,1,hidden,H,W)
            layer_output = torch.cat(outputs, dim=1)  # (B,T,hidden,H,W)
            last_states.append((h, c))
            layer_input = layer_output
        # layer_input holds the output of the last layer at this point
        return layer_input, last_states
