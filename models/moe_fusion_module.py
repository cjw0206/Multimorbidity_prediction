import torch
import torch.nn as nn
import torch.nn.functional as F

# 1️⃣ concat 기반 fusion
class ConcatFusionExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, z_gat, z_gin):
        # z_gat, z_gin: (B, D)
        u = torch.cat([z_gat, z_gin], dim=-1)   # (B, 2D)
        h = self.act(self.fc1(u))               # (B, D)
        out = self.fc2(h)                       # (B, D)
        return out


# 2️⃣ element-wise multiplication 기반 fusion
class MulFusionExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, z_gat, z_gin):
        h = z_gat * z_gin                       # (B, D)
        h = self.act(self.fc1(h))               # (B, D)
        out = self.fc2(h)                       # (B, D)
        return out


# 3️⃣ 2-token cross/self-attention 기반 fusion
class CrossAttnFusionExpert(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.fc = nn.Linear(dim, dim)

    def forward(self, z_gat, z_gin):
        # z_gat, z_gin: (B, D)
        z_stack = torch.stack([z_gat, z_gin], dim=1)  # (B, 2, D)
        z_attn, _ = self.attn(z_stack, z_stack, z_stack)  # (B, 2, D)

        # 두 토큰을 평균하거나, 첫 토큰만 쓰는 등 선택 가능
        z_fused = z_attn.mean(dim=1)  # (B, D)
        out = self.fc(z_fused)        # (B, D)
        return out


# 4️⃣ 가중합 기반 fusion: alpha * z_gat + (1-alpha) * z_gin
class WeightedSumFusionExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_alpha = nn.Linear(dim * 2, dim)  # 채널별 alpha
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, z_gat, z_gin):
        u = torch.cat([z_gat, z_gin], dim=-1)      # (B, 2D)
        alpha = torch.sigmoid(self.fc_alpha(u))    # (B, D), 0~1
        h = alpha * z_gat + (1.0 - alpha) * z_gin  # (B, D)
        out = self.fc_out(h)                       # (B, D)
        return out


class FusionMoE(nn.Module):
    """
    z_gat, z_gin 두 view를 입력으로 받아서
    4가지 서로 다른 fusion expert 중 top-k를 선택하는 MoE.
    Expert:
      0: ConcatFusion
      1: MulFusion
      2: CrossAttnFusion
      3: WeightedSumFusion
    """
    def __init__(self, dim, num_experts=4, k=1, dropout=0.1, num_heads=4):
        super().__init__()
        assert num_experts == 4, "현재 구현은 4개 expert에 맞춰져 있음"

        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            ConcatFusionExpert(dim),           # expert 0
            MulFusionExpert(dim),              # expert 1
            CrossAttnFusionExpert(dim, num_heads=num_heads),  # expert 2
            WeightedSumFusionExpert(dim),      # expert 3
        ])

        # 게이트는 concat(z_gat, z_gin)를 보고 어떤 fusion expert를 쓸지 결정
        self.gate = nn.Linear(dim * 2, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_gat, z_gin):
        """
        z_gat, z_gin: (B, D)
        return: fused_z (B, D), aux_loss
        """
        x = torch.cat([z_gat, z_gin], dim=-1)    # (B, 2D)
        B, _ = x.size()

        gate_logits = self.gate(x)              # (B, E)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, E)

        # Top-k expert selection
        topk_probs, topk_idx = torch.topk(gate_probs, self.k, dim=-1)  # (B, k), (B, k)

        output = torch.zeros_like(z_gat)        # (B, D)
        expert_counter = torch.zeros(self.num_experts, device=z_gat.device)

        for expert_id in range(self.num_experts):
            # 이 expert가 선택된 샘플 mask
            mask = (topk_idx == expert_id).any(dim=-1)  # (B,)
            if mask.sum() == 0:
                continue

            zg_i = z_gat[mask]   # (B_i, D)
            zi_i = z_gin[mask]   # (B_i, D)

            out_i = self.experts[expert_id](zg_i, zi_i)  # (B_i, D)

            # 선택된 샘플 중 이 expert에 해당하는 확률만 추출
            probs_i = topk_probs[mask]   # (B_i, k)
            idx_i   = topk_idx[mask]     # (B_i, k)
            mask_i  = (idx_i == expert_id)  # (B_i, k)

            selected_probs = probs_i[mask_i]  # (M_i,)
            if selected_probs.numel() > 0:
                avg_prob = selected_probs.mean()
            else:
                avg_prob = 0.0

            weighted_out = avg_prob * out_i   # (B_i, D)
            output[mask] = weighted_out

            expert_counter[expert_id] += mask.sum()

        output = self.dropout(output)

        # Load balancing auxiliary loss (네가 쓰던 방식 그대로)
        total_tokens = B
        avg_expert_usage = expert_counter / total_tokens  # (E,)
        aux_loss = (avg_expert_usage ** 2).sum() * self.num_experts

        return output, aux_loss
