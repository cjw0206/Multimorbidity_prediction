# moe_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """단일 Expert Layer"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        x: (N, D)
        return: (N, D)
        """
        return self.fc2(self.act(self.fc1(x)))


class ExpertModule(nn.Module):
    """
    Top-k Expert 선택 및 load-balancing loss를 포함한 Mixture of Experts 모듈
    기본: Top-1 / 필요 시 Top-2 이상 확장 가능
    """
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, D) 또는 (B, T, D)
        return: (output, aux_loss)
        """
        if x.dim() == 2:
            B, D = x.size()
            T = 1
            x_flat = x
        elif x.dim() == 3:
            B, T, D = x.size()
            x_flat = x.view(-1, D)
        else:
            raise ValueError("Input tensor must be 2D or 3D")

        # Gating network
        gate_logits = self.gate(x_flat)              # (B*T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B*T, E)

        # Top-k expert selection
        topk_probs, topk_idx = torch.topk(gate_probs, self.k, dim=-1)
        output = torch.zeros_like(x_flat)
        expert_counter = torch.zeros(self.num_experts, device=x.device)

        for expert_id in range(self.num_experts):
            # (B*T, E) → 특정 expert가 top-k에 포함된 샘플만 True
            mask = (topk_idx == expert_id).any(dim=-1)   # (B*T,)
            if mask.sum() == 0:
                continue

            input_i = x_flat[mask]                       # (N_i, D)
            out_i = self.experts[expert_id](input_i)     # (N_i, D_out)

            # 해당 expert에 해당하는 확률만 추출
            probs_i = topk_probs[mask]                   # (N_i, k)
            idx_i   = topk_idx[mask]                     # (N_i, k)
            # k>1일 경우, 선택된 column 중 expert_id 위치를 찾아 평균 확률 계산
            mask_i  = (idx_i == expert_id)               # (N_i, k)
            selected_probs = probs_i[mask_i]             # (M_i,) 선택된 위치의 확률들
            if selected_probs.numel() > 0:
                avg_prob = selected_probs.mean()         # 평균 확률
            else:
                avg_prob = 0.0

            # 최종 weighted output
            weighted_out = avg_prob * out_i              # (N_i, D_out)
            output[mask] = weighted_out
            expert_counter[expert_id] += mask.sum()


        # reshape & dropout
        output = output.view(B, T, D) if T > 1 else output
        output = self.dropout(output)

        # Load balancing auxiliary loss
        total_tokens = B * T
        avg_expert_usage = expert_counter / total_tokens
        aux_loss = (avg_expert_usage ** 2).sum() * self.num_experts

        return output, aux_loss



# class ExpertModule_3loss(nn.Module):
#     """
#     Top-k Expert 선택 및 load-balancing loss를 포함한 Mixture of Experts 모듈
#     (수정됨: 3가지 보조 손실 포함)
#     """
#     def __init__(self, input_dim, hidden_dim, num_experts=4, k=1, dropout=0.1, 
#                  aux_weight=0.01, ortho_weight=0.001, var_weight=0.0001):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k
#         self.experts = nn.ModuleList([
#             ExpertLayer(input_dim, hidden_dim) for _ in range(num_experts)
#         ])
#         self.gate = nn.Linear(input_dim, num_experts)
#         self.dropout = nn.Dropout(dropout)
        
#         # 새로운 손실 계산 모듈 추가
#         self.loss_calculator = MoELosses(aux_weight, ortho_weight, var_weight)


#     def forward(self, x):
#         """
#         x: (B, D) 또는 (B, T, D)
#         return: (output, total_aux_loss)
#         """
#         if x.dim() == 2:
#             B, D = x.size()
#             T = 1
#             x_flat = x
#         elif x.dim() == 3:
#             B, T, D = x.size()
#             x_flat = x.view(-1, D)
#         else:
#             raise ValueError("Input tensor must be 2D or 3D")

#         # Gating network
#         gate_logits = self.gate(x_flat) 
#         gate_probs = F.softmax(gate_logits, dim=-1) # S: 라우팅 가중치 (B*T, E)

#         # Top-k expert selection
#         topk_probs, topk_idx = torch.topk(gate_probs, self.k, dim=-1)
#         output = torch.zeros_like(x_flat)
        
#         # L_o 계산을 위해 각 expert가 처리한 토큰의 출력을 저장할 리스트
#         expert_outputs_list = []

#         for expert_id in range(self.num_experts):
#             # (B*T, E) → 특정 expert가 top-k에 포함된 샘플만 True
#             mask = (topk_idx == expert_id).any(dim=-1)
            
#             # 전문가의 출력을 계산하고 저장
#             if mask.sum() > 0:
#                 input_i = x_flat[mask]
#                 out_i = self.experts[expert_id](input_i)  # (N_i, D_out)
#                 expert_outputs_list.append(out_i)
#             else:
#                 # 할당된 토큰이 없는 경우 빈 텐서 또는 0 텐서를 추가 (오류 방지)
#                 expert_outputs_list.append(torch.tensor([], dtype=x.dtype, device=x.device).view(0, D))
#                 continue

#             input_i = x_flat[mask]                       # (N_i, D)
#             out_i = self.experts[expert_id](input_i)     # (N_i, D_out)

#             # 해당 expert에 해당하는 확률만 추출
#             probs_i = topk_probs[mask]                   # (N_i, k)
#             idx_i   = topk_idx[mask]                     # (N_i, k)
#             # k>1일 경우, 선택된 column 중 expert_id 위치를 찾아 평균 확률 계산
#             mask_i  = (idx_i == expert_id)               # (N_i, k)
#             selected_probs = probs_i[mask_i]             # (M_i,) 선택된 위치의 확률들
            
#             # NOTE: 사용자 코드가 k>1에서 Expert별 가중치를 평균 확률로 계산하는 독특한 방식을 따르고 있어 그대로 유지
#             if selected_probs.numel() > 0:
#                 # 해당 expert에 대한 토큰의 평균 라우팅 확률 사용
#                 avg_prob = selected_probs.mean() 
#             else:
#                 avg_prob = 0.0

#             # 최종 weighted output
#             # out_i는 이미 계산되었고, expert_outputs_list의 마지막 요소와 동일
#             weighted_out = avg_prob * expert_outputs_list[-1] 
#             output[mask] = weighted_out
            
#         # reshape & dropout
#         output = output.view(B, T, D) if T > 1 else output.view(B, T, D).squeeze(1) if T==1 else output # B, D
#         output = self.dropout(output)

#         # -----------------------------------------------
#         # 새로운 3가지 보조 손실 계산
#         # -----------------------------------------------
#         total_tokens = B * T
#         total_aux_loss = self.loss_calculator(gate_probs, expert_outputs_list, self.num_experts, total_tokens)
        
#         # 최종 모델 출력과 총 보조 손실 반환
#         return output, total_aux_loss
    


    
# class MoELosses(nn.Module):
#     """
#     논문 'Advancing Expert Specialization for Better MoE'에서 제안하는
#     세 가지 보조 손실을 계산하는 클래스.
#     """
#     def __init__(self, aux_weight=0.01, ortho_weight=0.001, var_weight=0.0001):
#         super().__init__()
#         self.aux_weight = aux_weight
#         self.ortho_weight = ortho_weight
#         self.var_weight = var_weight

#     def forward(self, gate_probs, expert_outputs_list, num_experts, total_tokens):
#         """
#         gate_probs: 라우터의 Softmax 출력 (N_total, E)
#         expert_outputs_list: 각 expert의 출력을 담은 리스트 (E개의 텐서)
#         num_experts: 전문가 수
#         total_tokens: 전체 토큰 수 (B*T)
        
#         S (라우팅 가중치) 대신 gate_probs를 사용하며,
#         Expert 출력을 활용하여 Orthogonality Loss를 계산합니다.
#         """
        
#         # -----------------------------------------------
#         # 1. 기존의 부하 균형 손실 (Auxiliary Load Balancing Loss, L_aux)
#         # -----------------------------------------------
#         # 논문의 p_j (총 라우팅 점수)는 gate_probs.sum(dim=0) 입니다.
#         p_j = gate_probs.sum(dim=0) 
        
#         # p_j의 분포를 균일 분포 (f_j_ideal)와 가깝게 만드는 GShard/Switch-MoE 방식의 변형
#         p_j_normalized = p_j / total_tokens  # [E]
#         f_j_ideal = torch.ones_like(p_j_normalized) / num_experts # [E]
        
#         # KL-Divergence를 사용하여 L_aux 계산
#         # torch.kl_div는 log-확률을 기대하므로 log()를 취합니다.
#         L_aux = F.kl_div(p_j_normalized.log(), f_j_ideal, reduction='sum')
        
#         # -----------------------------------------------
#         # 2. 전문가 특화 목적 (Orthogonality Loss, L_o)
#         # -----------------------------------------------
#         # 전문가 출력 Y_expert_avg를 기반으로 직교성을 강제합니다.
        
#         # 각 expert의 출력 텐서를 리스트로 받아 평균을 계산하고 텐서로 결합
#         # 주의: expert_outputs_list에는 '선택된' 토큰에 대한 출력만 포함되어 있으므로,
#         # 이 구현은 개념적이며, 논문의 원래 구현(모든 토큰에 대한 전문가 출력)과 다를 수 있습니다.
#         # 여기서는 각 expert가 처리한 토큰의 평균 출력 벡터를 사용합니다.
#         expert_outputs_avg = []
#         for out in expert_outputs_list:
#             if out.numel() > 0:
#                 expert_outputs_avg.append(out.mean(dim=0))
#             else:
#                 # 선택된 토큰이 없는 expert를 위해 0 벡터 삽입
#                 expert_outputs_avg.append(torch.zeros_like(expert_outputs_list[0].mean(dim=0))) 
        
#         Y_expert_avg = torch.stack(expert_outputs_avg) # [E x D]

#         # 전문가 출력 간의 내적 행렬 (Gram Matrix) 계산
#         Gram_matrix = torch.matmul(Y_expert_avg, Y_expert_avg.T) # [E x E]
        
#         # 비대각선 요소만 추출하여 L2 norm 최소화 (직교성)
#         Identity = torch.eye(num_experts, device=Y_expert_avg.device)
#         L_ortho = torch.norm(Gram_matrix * (1 - Identity), p='fro')**2

#         # -----------------------------------------------
#         # 3. 라우팅 다양화 목적 (Variance Loss, L_v)
#         # -----------------------------------------------
#         # 라우팅 점수 (gate_probs)의 분산을 높여 결정이 명확해지도록 유도합니다.
#         # 토큰별 라우팅 점수 S[i, :]의 분산을 최대화합니다.
        
#         # 토큰별 분산 계산 (전문가 축에 대해)
#         token_variances = gate_probs.var(dim=1) # [N_total]

#         # L_v는 분산을 최대화하므로, 손실 함수에서는 음수(-)를 취해야 합니다.
#         L_var = -token_variances.mean()

#         # -----------------------------------------------
#         # 최종 보조 손실 계산
#         # -----------------------------------------------
        
#         L_aux_total = (self.aux_weight * L_aux) + \
#                       (self.ortho_weight * L_ortho) + \
#                       (self.var_weight * L_var)

#         return L_aux_total


class ExpertModule_3loss(nn.Module):
    """
    Top-k MoE + Load Balancing + Orthogonality Loss (Lo) + Variance Loss (Lv)
    논문 'Advancing Expert Specialization for Better MoE' 통합 구현
    """
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=1,
                 dropout=0.1, alpha=0.001, beta=0.001, gamma=0.001):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.alpha = alpha   # aux
        self.beta = beta     # Lo
        self.gamma = gamma   # Lv

        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            B, D = x.size()
            T = 1
            x_flat = x
        else:
            B, T, D = x.size()
            x_flat = x.reshape(-1, D)

        N = B * T

        # ----------- Routing ------------
        gate_logits = self.gate(x_flat)      # (N, E)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # top-k routing
        topk_probs, topk_idx = torch.topk(gate_probs, self.k, dim=-1)

        # 각 expert별 output 저장
        expert_outputs = {eid: [] for eid in range(self.num_experts)}
        expert_counter = torch.zeros(self.num_experts, device=x.device)

        output = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            mask = (topk_idx == expert_id).any(dim=-1)  # (N,)
            if mask.sum() == 0:
                continue

            input_i = x_flat[mask]                    # (n_i, D)
            out_i = self.experts[expert_id](input_i)  # (n_i, D)
            expert_outputs[expert_id].append(out_i)

            # weighted output
            probs_i = topk_probs[mask]                # (n_i, k)
            idx_i = topk_idx[mask]                    # (n_i, k)
            mask_i = (idx_i == expert_id)
            selected_probs = probs_i[mask_i]

            avg_prob = selected_probs.mean() if selected_probs.numel() > 0 else 0.0
            output[mask] = avg_prob * out_i

            expert_counter[expert_id] += mask.sum()

        output = output.view(B, T, D) if T > 1 else output
        output = self.dropout(output)

        # ----------- Auxiliary load-balancing loss (기존) -----------
        avg_expert_usage = expert_counter / N
        aux_loss = (avg_expert_usage ** 2).sum() * self.num_experts

        # ------------------------------------------------------------
        # --------------- (논문) Orthogonality Loss Lo ---------------
        # ------------------------------------------------------------
        orth_loss = 0.0
        for eid, outs in expert_outputs.items():
            if len(outs) == 0:
                continue
            E_out = torch.cat(outs, dim=0)  # (n_e, D)

            # Gram matrix (n_e, n_e)
            G = E_out @ E_out.t()

            # off-diagonal = expert output들이 얼마나 겹치는지 나타냄
            off_diag = G - torch.diag(torch.diag(G))
            orth_loss += (off_diag ** 2).mean()

        # ------------------------------------------------------------
        # --------------- (논문) Variance Loss Lv ---------------------
        # ------------------------------------------------------------
        # 전문가 j에 대한 routing score 분산을 최대화 (∴ 부호 - )
        mean_scores = gate_probs.mean(dim=0)          # (E,)
        var_loss = -((gate_probs - mean_scores)**2).mean()

        # ------------------------------------------------------------
        # -------------------- Total Loss -----------------------------
        # ------------------------------------------------------------
        total_loss = (
            self.alpha * aux_loss +
            self.beta * orth_loss +
            self.gamma * var_loss
        )

        return output, total_loss
