"""Longitudinal 1D CNN + GRU Sequence Model for EHR Cognitive Trajectory Classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LongitudinalCNNGRU(nn.Module):
    """Compact 1D CNN + GRU neural architecture for multi-step clinical EHR trajectories.
    
    Architecture:
      1. Conv1D layers extract local multi-step patterns across adjacent clinical observations.
      2. Batch normalization and dropout stabilize feature representations.
      3. GRU layer captures long-range temporal dependencies and event ordering.
      4. Masked temporal pooling combines final trajectory state and mean trajectory representation.
      5. Multi-layer classification head outputs 4-class logits.
    """

    def __init__(
        self,
        input_dim=41,
        conv_dim=64,
        gru_dim=64,
        num_classes=4,
        dropout=0.2,
        kernel_size=3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conv_dim = conv_dim
        self.gru_dim = gru_dim
        self.num_classes = num_classes

        # 1D CNN Feature Extractor (Local pattern recognition)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(conv_dim)
        
        self.conv2 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(conv_dim)
        self.conv_dropout = nn.Dropout(p=dropout * 0.75)

        # Temporal Sequence Modeler (Long-term order and progression)
        self.gru = nn.GRU(
            input_size=conv_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Aggregation: Last valid step representation + Masked average pooling
        pooling_dim = gru_dim * 2

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(pooling_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            lengths: LongTensor of shape (batch_size,) with valid sequence lengths
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Transpose for 1D convolution: (B, D, T)
        x_conv = x.transpose(1, 2)
        h_conv = F.relu(self.bn1(self.conv1(x_conv)))
        h_conv = self.conv_dropout(h_conv)
        h_conv = F.relu(self.bn2(self.conv2(h_conv))) + h_conv  # Residual skip
        
        # Transpose back for GRU: (B, T, conv_dim)
        h_gru_in = h_conv.transpose(1, 2)
        gru_out, _ = self.gru(h_gru_in)  # (B, T, gru_dim)

        # 1. Last valid step representation
        last_indices = (lengths - 1).clamp(min=0, max=seq_len - 1)
        batch_indices = torch.arange(batch_size, device=x.device)
        last_step_feat = gru_out[batch_indices, last_indices, :]  # (B, gru_dim)

        # 2. Masked mean pooling across valid time steps
        step_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, T)
        mask = (step_indices < lengths.unsqueeze(1)).float().unsqueeze(-1)   # (B, T, 1)
        
        masked_sum = (gru_out * mask).sum(dim=1)  # (B, gru_dim)
        lengths_expanded = lengths.unsqueeze(1).clamp(min=1).float()  # (B, 1)
        mean_step_feat = masked_sum / lengths_expanded  # (B, gru_dim)

        # Combined representation
        pooled = torch.cat([last_step_feat, mean_step_feat], dim=-1)  # (B, 2 * gru_dim)

        # Classification logits
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x, lengths):
        """Inference probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probs = F.softmax(logits, dim=-1)
        return probs
