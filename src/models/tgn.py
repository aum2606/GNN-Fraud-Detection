import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import GRUCell
from typing import Optional, Tuple

from src.config import MODEL_CONFIG

import logging
logger = logging.getLogger(__name__)

class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for fraud detection.
    Combines GAT layers with GRU for temporal dynamics.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        residual: bool = True,
        use_batch_norm: bool = True,
    ):
        super(TemporalGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_batch_norm = use_batch_norm
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=8, concat=False, dropout=dropout))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels, hidden_channels, heads=8, concat=False, dropout=dropout))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        if num_layers > 1:
            self.gat_layers.append(GATConv(hidden_channels, hidden_channels, heads=8, concat=False, dropout=dropout))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # GRU for temporal dynamics
        self.gru = GRUCell(hidden_channels, hidden_channels)
        
        # Output layer
        self.out = nn.Linear(hidden_channels, out_channels)
        
        logger.info(f"Initialized TemporalGNN with {num_layers} layers, {hidden_channels} hidden channels")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TemporalGNN.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for nodes [num_nodes]
            hidden_state: Previous hidden state for GRU [num_nodes, hidden_channels]
            
        Returns:
            out: Output predictions [num_nodes, out_channels]
            hidden_state: Updated hidden state [num_nodes, hidden_channels]
        """
        # Initial hidden state
        if hidden_state is None:
            hidden_state = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        
        # GAT layers
        h = x
        for i, gat in enumerate(self.gat_layers):
            h_new = gat(h, edge_index)
            
            if self.use_batch_norm:
                h_new = self.batch_norms[i](h_new)
            
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            if self.residual and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Update hidden state with GRU
        hidden_state = self.gru(h, hidden_state)
        
        # Output layer
        out = self.out(hidden_state)
        
        return out, hidden_state
    
    def predict(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = True
    ) -> torch.Tensor:
        """
        Make predictions using the model.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for nodes [num_nodes]
            hidden_state: Previous hidden state for GRU [num_nodes, hidden_channels]
            apply_sigmoid: Whether to apply sigmoid to the output
            
        Returns:
            predictions: Predicted probabilities [num_nodes, out_channels]
        """
        out, _ = self.forward(x, edge_index, batch, hidden_state)
        
        if apply_sigmoid:
            out = torch.sigmoid(out)
            
        return out
