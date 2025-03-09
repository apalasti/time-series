from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


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
        x = torch.transpose(self.conv(x.permute(0, 2, 1)), (1, 2))
        return x


class TimeDRL(nn.Module):
    def __init__(
        self,
        ts_len: int,
        input_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        token_embedding_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.ts_len = ts_len
        self.input_channels = input_channels
        self.d_model = d_model

        # NOTE: The code implementing the paper just prepends the [CLS]
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, input_channels, dtype=torch.float), requires_grad=True
        )
        self.token_embeddings = TokenEmbedding(
            self.input_channels, d_model, token_embedding_kernel_size
        )
        self.positional_embeddings = nn.Parameter(
            # +1 Positional Embedding for the [CLS] token
            torch.randn(self.ts_len + 1, d_model, dtype=torch.float),
            requires_grad=True,
        )
        self.dropout = nn.Dropout(p=dropout)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                batch_first=True,
                dropout=dropout,
                dim_feedforward=4*d_model,
                activation=nn.GELU,
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        self.reconstructor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.input_channels),
        )
        self.contrastive_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.BatchNorm1d(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, self.d_model),
        )

    def forward(self, x: Tensor):
        B, T, C = x.shape
        assert T == self.ts_len and C == self.input_channels

        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # Preprend [CLS]
        x = self.token_embeddings(x)  # Create token embeddings
        x = x + self.positional_embeddings[None, : T + 1]  # Apply positional embeddings
        x = self.dropout(x)
        assert x.shape == (B, T+1, self.d_model)

        x = self.encoder(x)
        return x


if __name__ == "__main__":
    model = TimeDRL(
        ts_len=168,
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
