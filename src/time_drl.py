import math

import torch
import torch.nn as nn
from torch import Tensor



class AttentionInspector:
    def __init__(self, model: nn.Module):
        self.outputs = []

        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                AttentionInspector.patch_attention(module)
                module.register_forward_hook(self)

    @staticmethod
    def patch_attention(m):
        forward_orig = m.forward
        def wrap(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False

            return forward_orig(*args, **kwargs)
        m.forward = wrap

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


class TokenEmbedding(nn.Module):
    def __init__(self, input_channels: int, d_model: int, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2  # `same` padding
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        # Weight initalization
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, x: Tensor):
        x = self.conv(x.permute(0, 2, 1))
        return torch.transpose(x, 1, 2)


class TimeDRL(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        input_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        token_embedding_kernel_size: int,
        dropout: float,
        pos_embed_type = "learnable"
    ) -> None:
        super().__init__()

        self.sequence_len = sequence_len
        self.input_channels = input_channels
        self.d_model = d_model

        # NOTE: The code implementing the paper just prepends the [CLS]
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, input_channels, dtype=torch.float), requires_grad=True
        )
        self.token_embeddings = TokenEmbedding(
            self.input_channels, d_model, token_embedding_kernel_size
        )

        if pos_embed_type == "learnable":
            self.positional_embeddings = nn.Parameter(
                # +1 Positional Embedding for the [CLS] token
                torch.randn(self.sequence_len + 1, d_model, dtype=torch.float),
                requires_grad=True,
            )
        elif pos_embed_type == "fixed":
            pe = torch.zeros(sequence_len + 1, d_model, requires_grad=False).float()
            position = torch.arange(0, sequence_len + 1).float().unsqueeze(1)
            div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            ).exp()

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("positional_embeddings", pe)
        elif pos_embed_type == "none":
            self.positional_embeddings = None
        else:
            raise ValueError(
                f"Unknown pos_embed_type: {pos_embed_type}. "
                "Supported values are 'learnable', 'fixed' and 'none'"
            )

        self.dropout = nn.Dropout(p=dropout)

        # self.save_output = SaveOutput()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                batch_first=True,
                norm_first=True,
                dropout=dropout,
                dim_feedforward=4 * d_model,
                activation="gelu",
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        self.reconstructor = nn.Sequential(
            nn.Dropout(dropout),
            # nn.Linear(2 * self.d_model, self.input_channels),
            nn.Linear(1 * self.d_model, self.input_channels),
        )
        self.contrastive_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.BatchNorm1d(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, self.d_model),
        )

    def forward(self, x: Tensor):
        # self.save_output.clear()
        B, T, C = x.shape
        assert T == self.sequence_len and C == self.input_channels, (
            f"Input tensor shape {x.shape} does not match expected dimensions. "
            f"Expected sequence length: {self.sequence_len}, got {T}. "
            f"Expected input channels: {self.input_channels}, got {C}."
        )

        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # Preprend [CLS]
        x = self.token_embeddings(x)  # Create token embeddings

        if self.positional_embeddings is not None:  # Apply positional embeddings
            x = x + self.positional_embeddings[None, : T + 1]

        x = self.dropout(x)
        assert x.shape == (B, T+1, self.d_model), (
            f"Unexpected shape after applying positional embeddings. "
            f"Expected: {(B, T + 1, self.d_model)}, got: {x.shape}"
        )

        x = self.encoder(x)
        return x


if __name__ == "__main__":
    model = TimeDRL(
        sequence_len=168,
        input_channels=3 * 4,
        d_model=1408,
        n_heads=11,
        n_layers=8,
        token_embedding_kernel_size=5,
        dropout=0.2,
    )
    print(model)
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Token embedder parameters: {sum(p.numel() for p in model.token_embeddings.parameters()):,}")
    print(
        f"Reconstructor parameters: {sum(p.numel() for p in model.reconstructor.parameters()):,}"
    )
    print(
        f"Contrastive predictor parameters: {sum(p.numel() for p in model.contrastive_predictor.parameters()):,}"
    )
