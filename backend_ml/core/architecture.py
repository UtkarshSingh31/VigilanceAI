import torch
import torch.nn as nn
from backend_ml.logger import get_logger

logger = get_logger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, seq_len: int = 30):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 2-layer Encoder matching your trained checkpoint (hidden_dim=32)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 2-layer Decoder matching your trained checkpoint (hidden_dim=32)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Linear output layer (32 -> 6)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        logger.info(
            f"Initialized LSTMAutoencoder (input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, num_layers=2, seq_len={seq_len})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(2) != self.input_dim:
            logger.error(f"Invalid tensor shape: {x.shape}. Expected (*, {self.seq_len}, {self.input_dim})")
            raise ValueError(f"Expected tensor shape (*, {self.seq_len}, {self.input_dim}), got {x.shape}")

        _, (hn, _) = self.encoder_lstm(x)
        hidden_summary = hn[-1]
        decoder_input = hidden_summary.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input)
        predictions = self.output_layer(decoder_out)
        return predictions