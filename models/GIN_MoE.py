import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GINConv
from typing import Optional

from .GNN import GNN


# -----------------------------
# 1️⃣ Mixture-of-Experts Layer
# -----------------------------
class MoELayer(nn.Module):
    """
    Mixture-of-Experts Layer with Top-K Gating and Load Balancing Loss.
    """
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

        # Gating logits → probabilities
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
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

        # Load Balancing Loss
        Prob = gate_probs.mean(dim=0)
        Frac = experts_mask.float().mean(dim=0)
        load_balancing_loss = self.num_experts * torch.sum(Prob * Frac)

        return output, load_balancing_loss


# -----------------------------
# 2️⃣ ApplyNodeFunc_MoE
# -----------------------------
class ApplyNodeFunc_MoE(nn.Module):
    """MLP + BN + ReLU + MoE"""
    def __init__(self, mlp, top_k=2, num_experts=4):
        super().__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)
        self.moe = MoELayer(
            input_dim=self.mlp.output_dim,
            output_dim=self.mlp.output_dim,
            hidden_dim=self.mlp.output_dim * 2,
            top_k=top_k,
            num_experts=num_experts
        )

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        h, moe_loss = self.moe(h)
        return h, moe_loss


# -----------------------------
# 3️⃣ MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


# -----------------------------
# 4️⃣ GIN_MoE
# -----------------------------
class GIN_MoE(GNN):
    """GIN with integrated Mixture-of-Experts"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_mlp_layers,
                 final_dropout, tasks, causal, neighbor_pooling_type="mean", learn_eps=True,
                 top_k=1, num_experts=4):
        self.learn_eps = learn_eps
        self.num_mlp_layers = num_mlp_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.top_k = top_k
        self.num_experts = num_experts

        super().__init__(in_dim, hidden_dim, out_dim, num_layers, F.relu, final_dropout, tasks, causal)

    # -----------------------
    # Layers 정의
    # -----------------------
    def get_layers(self):
        layers = nn.ModuleList()
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(self.num_mlp_layers, self.in_dim, self.hidden_dim, self.hidden_dim)
            else:
                mlp = MLP(self.num_mlp_layers, self.hidden_dim, self.hidden_dim, self.hidden_dim)

            layers.append(
                GINConv(
                    ApplyNodeFunc_MoE(mlp, self.top_k, self.num_experts),
                    self.neighbor_pooling_type,
                    0,
                    self.learn_eps
                )
            )
        return layers

    # -----------------------
    # GNN Propagation + MoE Loss
    # -----------------------
    def get_logit(self, g, h, causal=False):
        total_moe_loss = 0.0
        layers = self.layers if not causal else self.rand_layers

        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)

            if isinstance(layer.apply_func, ApplyNodeFunc_MoE):
                h, moe_loss = layer.apply_func(h)
                total_moe_loss += moe_loss
            else:
                h = layer.apply_func(h)

            # if isinstance(h, tuple):
            #     h = h[0]
            # h = layer(g, h)

        self.set_embeddings(h)
        return h, total_moe_loss

    # -----------------------
    # encode(): Base GNN의 인터페이스 확장
    # -----------------------
    def encode(self, g, h, causal=False, return_moe=False):
        """Return node embeddings, optionally with MoE loss"""
        z, moe_loss = self.get_logit(g, h, causal)
        if return_moe:
            return z, moe_loss
        return z

    # -----------------------
    # forward_edges(): 기존 학습 루프 호환
    # -----------------------
    def forward_edges(self, g, h, predictor, eids=None, uv=None, causal: bool = False,
                      fixed_nid: Optional[int] = None, use_interaction: bool = False):
        z, moe_loss = self.encode(g, h, causal=causal, return_moe=True)
        logits = predictor(g, z, eids=eids, uv=uv, fixed_nid=fixed_nid)
        return logits, moe_loss

    # -----------------------
    # forward(): 노드 예측용
    # -----------------------
    def forward(self, g, h, task=None, person_node_type_id=None):
        z, moe_loss = self.get_logit(g, h)
        if task is None or task == "link_pred":
            return z, moe_loss
        node_logits = self.out[task](z)
        if person_node_type_id is not None:
            node_logits = node_logits[g.ndata["_TYPE"] == person_node_type_id]
        return node_logits, moe_loss
