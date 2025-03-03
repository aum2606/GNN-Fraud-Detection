import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple

import logging
logger = logging.getLogger(__name__)

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) for fraud detection.
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
        super(GAT, self).__init__()
        
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
        
        # Output layer
        self.out = nn.Linear(hidden_channels, out_channels)
        
        logger.info(f"Initialized GAT with {num_layers} layers, {hidden_channels} hidden channels")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the GAT.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for nodes [num_nodes]
            
        Returns:
            out: Output predictions [num_nodes, out_channels]
        """
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
        
        # Output layer
        out = self.out(h)
        
        return out
    
    def predict(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = True
    ) -> torch.Tensor:
        """
        Make predictions using the model.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for nodes [num_nodes]
            apply_sigmoid: Whether to apply sigmoid to the output
            
        Returns:
            predictions: Predicted probabilities [num_nodes, out_channels]
        """
        out = self.forward(x, edge_index, batch)
        
        if apply_sigmoid:
            out = torch.sigmoid(out)
            
        return out
