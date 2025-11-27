import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from typing import Optional

from .GNN import GNN


# -----------------------------
# 1️⃣ Mixture-of-Experts Layer
# -----------------------------
class MoELayer(nn.Module):
    """Mixture-of-Experts Layer with Top-K Gating and Load Balancing Loss."""
    def __init__(self, input_dim, output_dim, hidden_dim, top_k, num_experts):
        super(MoELayer, self).__init__()
        assert input_dim == output_dim, "Input and output dims must match for MoE update."
        self.top_k = top_k
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        batch_size, _ = x.shape
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x)
        experts_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1)

        for i, expert in enumerate(self.experts):
            mask_topk = (top_k_indices == i)
            expert_weights = torch.sum(top_k_weights * mask_topk, dim=1)
            selected = torch.where(expert_weights > 0)[0]

            if selected.numel() > 0:
                input_expert = x[selected]
                expert_out = expert(input_expert)
                weighted_out = expert_out * expert_weights[selected].unsqueeze(1)
                output.index_add_(0, selected, weighted_out)

        Prob = gate_probs.mean(dim=0)
        Frac = experts_mask.float().mean(dim=0)
        load_balancing_loss = self.num_experts * torch.sum(Prob * Frac)

        return output, load_balancing_loss


# -----------------------------
# 2️⃣ GCN with integrated MoE
# -----------------------------
class GCN_MoE(GNN):
    """GCN with integrated Mixture-of-Experts module"""
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 tasks,
                 causal=False,
                 top_k=1,
                 num_experts=4):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, dropout, tasks, causal)
        self.top_k = top_k
        self.num_experts = num_experts

    # -----------------------
    # Define GCN layers
    # -----------------------
    def get_layers(self):
        layers = nn.ModuleList()
        layers.append(GraphConv(self.in_dim, self.hidden_dim, activation=self.activation))
        for _ in range(self.n_layers - 1):
            layers.append(GraphConv(self.hidden_dim, self.hidden_dim, activation=self.activation))
        return layers

    # -----------------------
    # Forward propagation with MoE
    # -----------------------
    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        total_moe_loss = 0.0

        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

            # Apply MoE after GCN transformation (per layer)
            moe = MoELayer(
                input_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim * 2,
                top_k=self.top_k,
                num_experts=self.num_experts
            ).to(h.device)

            h, moe_loss = moe(h)
            total_moe_loss += moe_loss

        self.set_embeddings(h)
        return h, total_moe_loss

    # -----------------------
    # Encode and return embeddings
    # -----------------------
    def encode(self, g, h, causal=False, return_moe=False):
        z, moe_loss = self.get_logit(g, h, causal)
        if return_moe:
            return z, moe_loss
        return z

    # -----------------------
    # Edge-level forward (link prediction)
    # -----------------------
    def forward_edges(self, g, h, predictor, eids=None, uv=None, causal: bool = False,
                      fixed_nid: Optional[int] = None, use_interaction: bool = False):
        z, moe_loss = self.encode(g, h, causal=causal, return_moe=True)
        logits = predictor(g, z, eids=eids, uv=uv, fixed_nid=fixed_nid)
        return logits, moe_loss

    # -----------------------
    # Node-level forward (classification)
    # -----------------------
    def forward(self, g, h, task=None, person_node_type_id=None):
        z, moe_loss = self.get_logit(g, h)
        if task is None or task == "link_pred":
            return z, moe_loss
        node_logits = self.out[task](z)
        if person_node_type_id is not None:
            node_logits = node_logits[g.ndata["_TYPE"] == person_node_type_id]
        return node_logits, moe_loss
