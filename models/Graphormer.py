import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from models.moe_module import ExpertModule

class GraphormerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                beta=True
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
            x = norm(x + residual)
        return self.proj_out(x)
    

class GraphormerEncoderMoE(nn.Module):
    """
    Graphormer encoder with Mixture-of-Experts (MoE) FFN layers.
    """
    def __init__(self, hidden_dim, num_heads=8, num_layers=3, dropout=0.1,
                 num_experts=4, moe_hidden_dim=None, top_k=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        moe_hidden_dim = moe_hidden_dim or hidden_dim * 4

        # TransformerConv layers
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                beta=True
            )
            for _ in range(num_layers)
        ])

        # MoE layers (FFN replacement)
        self.moe_layers = nn.ModuleList([
            ExpertModule(input_dim=hidden_dim,
                         hidden_dim=moe_hidden_dim,
                         num_experts=num_experts,
                         k=top_k,
                         dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norms_1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms_2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        total_aux_loss = 0.0

        for conv, moe, norm1, norm2 in zip(self.layers, self.moe_layers, self.norms_1, self.norms_2):
            # === 1. Graph Transformer layer ===
            residual = x
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
            x = norm1(x + residual)

            # === 2. MoE FFN layer ===
            residual = x
            ffn_out, aux_loss = moe(x)
            total_aux_loss += aux_loss
            x = self.dropout(ffn_out)
            x = norm2(x + residual)

        return self.proj_out(x), total_aux_loss