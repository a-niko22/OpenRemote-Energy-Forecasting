"""
Energy price forecasting model architectures.

Models (all PyTorch nn.Module):
  - LSTM        : Stacked LSTM -> linear head
  - GRU         : Stacked GRU -> linear head
  - BiLSTM      : Bidirectional LSTM -> linear head
  - Transformer : Vanilla Transformer encoder (time steps as tokens)
  - iTransformer: Inverted Transformer (variates as tokens, Liu et al. 2024)
"""

import math
import torch
import torch.nn as nn


# -- Positional Encoding (used by Transformer) --------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# -- Model classes -------------------------------------------------------------

class LSTMModel(nn.Module):
    """Stacked LSTM encoder with a linear forecasting head."""

    def __init__(self, input_size: int, horizon: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])          # (B, horizon)


class GRUModel(nn.Module):
    """Stacked GRU encoder with a linear forecasting head."""

    def __init__(self, input_size: int, horizon: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM with a linear forecasting head."""

    def __init__(self, input_size: int, horizon: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    """
    Vanilla Transformer encoder for time-series forecasting.
    Time steps are treated as sequence tokens; multi-head self-attention
    captures temporal dependencies across the lookback window.
    """

    def __init__(self, input_size: int, horizon: int,
                 d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)     # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)        # (B, T, d_model)
        return self.fc(x[:, -1, :])


class _NBEATSxBlock(nn.Module):
    """Single NBEATSx block: FC stack -> backcast + forecast heads."""

    def __init__(self, input_dim: int, hidden_size: int,
                 lookback: int, horizon: int,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_size, lookback)
        self.forecast_head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor):
        h = self.fc(x)
        return self.backcast_head(h), self.forecast_head(h)


class NBEATSxModel(nn.Module):
    """
    NBEATSx: Neural Basis Expansion Analysis with Exogenous Variables.

    Extends N-BEATS by incorporating exogenous (external) features alongside
    the target time series.  Exogenous inputs are projected into a context
    vector and concatenated with the backcast residual at each block.

    Reference: Olivares et al., "NBEATSx: Neural basis expansion analysis
    with exogenous variables" (Int. J. Forecasting, 2023).
    """

    def __init__(self, input_size: int, seq_len: int, horizon: int,
                 hidden_size: int = 256, n_stacks: int = 2,
                 n_blocks: int = 3, n_fc_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        exog_size = input_size - 1  # everything except price (col 0)

        # Project flattened exogenous series -> context vector
        if exog_size > 0:
            self.exog_proj = nn.Linear(seq_len * exog_size, hidden_size)
        else:
            self.exog_proj = None

        block_input_dim = seq_len + (hidden_size if exog_size > 0 else 0)

        self.stacks = nn.ModuleList([
            nn.ModuleList([
                _NBEATSxBlock(block_input_dim, hidden_size, seq_len, horizon,
                              n_fc_layers, dropout)
                for _ in range(n_blocks)
            ])
            for _ in range(n_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V)  where V = 1 (price) + exog features
        B = x.size(0)
        price = x[:, :, 0]                         # (B, seq_len)

        exog_ctx = None
        if self.exog_proj is not None and x.size(2) > 1:
            exog = x[:, :, 1:].reshape(B, -1)      # (B, seq_len * exog_size)
            exog_ctx = self.exog_proj(exog)         # (B, hidden_size)

        forecast = torch.zeros(B, self.horizon, device=x.device)
        residual = price

        for blocks in self.stacks:
            for block in blocks:
                if exog_ctx is not None:
                    block_in = torch.cat([residual, exog_ctx], dim=-1)
                else:
                    block_in = residual
                bc, fc = block(block_in)
                residual = residual - bc
                forecast = forecast + fc

        return forecast


class iTransformerModel(nn.Module):
    """
    Inverted Transformer (iTransformer).

    Unlike the standard Transformer, each *variate* (feature) is treated as a
    token whose embedding is the projection of its full time series.
    Self-attention then models cross-variate dependencies.

    Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective
    for Time Series Forecasting" (ICLR 2024).
    """

    def __init__(self, input_size: int, seq_len: int, horizon: int,
                 d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        # Project each variate's time series of length seq_len -> d_model embedding
        self.variate_proj = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        # All variate embeddings -> horizon forecast
        self.fc = nn.Linear(d_model * input_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V) -> invert -> (B, V, T)
        x = x.permute(0, 2, 1)
        x = self.variate_proj(x)    # (B, V, d_model)
        x = self.encoder(x)         # (B, V, d_model)
        x = x.flatten(1)            # (B, V * d_model)
        return self.fc(x)           # (B, horizon)


# -- Registry & factory --------------------------------------------------------

ALL_MODELS = ['lstm', 'gru', 'bilstm', 'transformer', 'itransformer', 'nbeatsx']


def build_model(name: str, input_size: int, seq_len: int, horizon: int,
                hidden_size: int = 128, num_layers: int = 2,
                d_model: int = 128, nhead: int = 8,
                dropout: float = 0.1) -> nn.Module:
    """Instantiate a model by name with the given hyperparameters."""
    name = name.lower()
    rnn_kwargs = dict(input_size=input_size, horizon=horizon,
                      hidden_size=hidden_size, num_layers=num_layers,
                      dropout=dropout)
    attn_kwargs = dict(input_size=input_size, horizon=horizon,
                       d_model=d_model, nhead=nhead,
                       num_layers=num_layers, dropout=dropout)
    if name == 'lstm':
        return LSTMModel(**rnn_kwargs)
    elif name == 'gru':
        return GRUModel(**rnn_kwargs)
    elif name == 'bilstm':
        return BiLSTMModel(**rnn_kwargs)
    elif name == 'transformer':
        return TransformerModel(**attn_kwargs)
    elif name == 'itransformer':
        return iTransformerModel(seq_len=seq_len, **attn_kwargs)
    elif name == 'nbeatsx':
        return NBEATSxModel(
            input_size=input_size, seq_len=seq_len, horizon=horizon,
            hidden_size=hidden_size, dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: {ALL_MODELS}")
