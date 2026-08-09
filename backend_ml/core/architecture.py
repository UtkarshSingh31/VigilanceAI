import torch
import torch.nn as nn
from backend_ml.logger import get_logger


logger = get_logger(__name__)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        """
        The structural blueprint for the Anomaly Detection model.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        try:
            logger.debug(
                f"Initializing LSTMAutoencoder: input_dim={input_dim}, "
                f"hidden_dim={hidden_dim}, num_layers={num_layers}"
            )
            
            #  ENCODER 
            self.encoder_lstm = nn.LSTM(
                input_size=self.input_dim, 
                hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, 
                batch_first=True
            )
            
            #  DECODER 
            self.decoder_lstm = nn.LSTM(
                input_size=self.hidden_dim, 
                hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, 
                batch_first=True
            )
            
            self.decoder_output_layer = nn.Linear(
                in_features=self.hidden_dim, 
                out_features=self.input_dim
            )
            
            logger.debug("LSTMAutoencoder architecture initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize LSTMAutoencoder architecture: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference. Expects shape: (batch_size, sequence_length, input_dim)
        """
        try:
            # Check input dimensions before running matrix operations
            if x.dim() != 3 or x.size(-1) != self.input_dim:
                logger.error(
                    f"Tensor dimension mismatch! Expected (batch, seq_len, {self.input_dim}), "
                    f"got shape {tuple(x.shape)}"
                )
                raise ValueError(f"Invalid tensor shape for forward pass: {x.shape}")

            #  Encode
            _, (hidden, _) = self.encoder_lstm(x)
            
            #  Repeat context vector for sequence length
            seq_len = x.shape[1]
            hidden_reshaped = hidden[-1].unsqueeze(1).expand(-1, seq_len, -1)
            
            #  Decode
            decoder_out, _ = self.decoder_lstm(hidden_reshaped)
            
            #  Map to output feature space
            reconstructed = self.decoder_output_layer(decoder_out)
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"Error during LSTM forward pass execution: {e}")
            raise

        