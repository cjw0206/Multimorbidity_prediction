

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph  # 타입 힌트 용도 (선택)

# GNN 백본은 models 패키지에서 가져오기
from models import GCN, GAT, GIN, GIN_MoE
from models.GIN_MoE import GIN_MoE
from models.GCN_MoE import GCN_MoE
from models.Graphormer import GraphormerEncoder, GraphormerEncoderMoE
from models.moe_module import ExpertModule, ExpertModule_3loss
from models.moe_fusion_module import FusionMoE

from core.config import Settings
# -----------------------------
# Predictor (엣지 분류기)
# -----------------------------

class MLPPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
        )
    # def forward(self, g: DGLGraph, z: torch.Tensor, eids: Optional[torch.Tensor] = None, uv=None):
    #     u, v = g.find_edges(eids) if uv is None else uv
    #     return self.mlp(torch.cat([z[u], z[v]], dim=-1))
        # fixed_nid와 기타 키워드는 받아서 그냥 무시
    def forward(self, g: DGLGraph, z: torch.Tensor, eids: Optional[torch.Tensor] = None, uv=None,
                fixed_nid: Optional[int] = None, **kwargs):
        u, v = g.find_edges(eids) if uv is None else uv
        return self.mlp(torch.cat([z[u], z[v]], dim=-1))
    
class EdgeMoEPredictor(nn.Module):
    """
    GAT/GIN에서 나온 z를 받아서,
    서로 다른 4가지 scoring expert를 mixture로 쓰는 링크 예측 MoE predictor.
    return: (out, aux_loss)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.2, num_experts: int = 4, top_k: int=1,):
        super().__init__()
        assert num_experts == 4, "현재 구현은 4개 expert에 맞춰져 있음"
        self.num_experts = num_experts
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k

        D = in_dim
        edge_feat_dim = 4 * D  # [z_u, z_v, |z_u-z_v|, z_u*z_v]

        # 🔹 게이트 네트워크: edge feature → expert weight
        self.gate = nn.Linear(edge_feat_dim, num_experts)

        # 🔹 Expert 1: concat 기반
        self.expert_concat = nn.Sequential(
            nn.Linear(2 * D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # 🔹 Expert 2: distance 기반 (|z_u - z_v|)
        self.expert_dist = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # 🔹 Expert 3: multiplicative 기반 (z_u * z_v)
        self.expert_mul = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # 🔹 Expert 4: all-in-one (concat + diff + mul)
        self.expert_all = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, g: DGLGraph, z: torch.Tensor,
                eids: Optional[torch.Tensor] = None, uv=None,
                fixed_nid: Optional[int] = None, **kwargs):
        # u, v 인덱스 가져오기 (기존 MLPPredictor와 동일)
        u, v = g.find_edges(eids) if uv is None else uv  # (B,)

        z_u = z[u]  # (B, D)
        z_v = z[v]  # (B, D)

        z_cat  = torch.cat([z_u, z_v], dim=-1)      # (B, 2D)
        z_diff = torch.abs(z_u - z_v)               # (B, D)
        z_mul  = z_u * z_v                          # (B, D)

        edge_feat = torch.cat([z_cat, z_diff, z_mul], dim=-1)  # (B, 4D)

        # 1) 각 expert의 score 계산
        s1 = self.expert_concat(z_cat)    # (B, out_dim)
        s2 = self.expert_dist(z_diff)     # (B, out_dim)
        s3 = self.expert_mul(z_mul)       # (B, out_dim)
        s4 = self.expert_all(edge_feat)   # (B, out_dim)

        scores = torch.stack([s1, s2, s3, s4], dim=1)  # (B, 4, out_dim)

        # 2) 게이트로 mixture weight 계산
        gate_logits = self.gate(edge_feat)           # (B, 4)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, 4)
        # gate_probs_expanded = gate_probs.unsqueeze(-1)  # (B, 4, 1)

        ################ top-K routing #################
        top_k = self.top_k
        if top_k == self.num_experts:
            # 모든 expert를 쓰는 경우: 일반 soft mixture (aux_loss만 유지)
            gate_st = gate_probs
        else:
            # 상위 k개의 prob와 index
            topk_vals, topk_idx = torch.topk(gate_probs, k=top_k, dim=-1)  # (B, k), (B, k)

            # hard_gate: 선택된 k개 위치에만 값을 두고 나머지는 0
            # 여기서는 선택된 위치에 원래 soft prob를 남김 (0/1이 아니라 "zero-out된 soft")
            hard_gate = torch.zeros_like(gate_probs)  # (B, num_experts)
            hard_gate.scatter_(1, topk_idx, topk_vals)

            # 🔹 Straight-through trick:
            # forward: hard_gate처럼 동작
            # backward: gate_probs에서 gradient를 받도록 구성
            gate_st = hard_gate + gate_probs - gate_probs.detach()  # (B, num_experts)

        gate_st_expanded = gate_st.unsqueeze(-1)                 # (B, num_experts, 1)
        ################ top-K routing #################

        # 3) 가중합
        # out = (gate_probs_expanded * scores).sum(dim=1)  # (B, out_dim)

        out = (gate_st_expanded * scores).sum(dim=1)  # (B, out_dim)
        out = self.dropout(out)

        # 4) load balancing auxiliary loss
        # 각 expert가 배치에서 얼마나 사용되는지 평균 (soft mixture 기준)
        # avg_expert_usage: (4,)
        avg_expert_usage = gate_probs.mean(dim=0)
        # usage가 한쪽으로 쏠리지 않게 L2 penalty
        aux_loss = (avg_expert_usage ** 2).sum() * self.num_experts

        return out, aux_loss


class HeteroProjectionGNN(nn.Module):
    """
    각 노드 타입(person, disease)의 피처를 타입별 프로젝션 레이어를 통해
    공통된 hidden_dim으로 맞춘 후, GNN 백본에 전달하는 모델
    """
    def __init__(self, person_in_dim: int, disease_in_dim: int, params: Dict, g_hetero: dgl.DGLHeteroGraph):
        super().__init__()
        self.model_type = params['model_type']
        hidden_dim = params['hidden_dim']
        
        # 1. 타입별 프로젝션 레이어를 ModuleDict로 관리
        # 각 타입을 hidden_dim으로 매핑합니다.
        self.projectors = nn.ModuleDict({
            'person': nn.Linear(person_in_dim, hidden_dim),
            'disease': nn.Linear(disease_in_dim, hidden_dim)
        })
        
        # 2. GNN 백본은 이제 hidden_dim을 입력으로 받습니다.
        activation, tasks, causal = F.relu, [], False
        if self.model_type == "gcn":
            self.gnn = GCN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], activation, params['dropout'], tasks, causal)
        elif self.model_type == "gat":
            self.gnn = GAT(params['n_layers'], hidden_dim, hidden_dim, hidden_dim, params['num_heads'], activation, params['dropout'], params['gat_attn_drop'], params['gat_neg_slope'], True, tasks, causal)
        elif self.model_type == "gin":
            self.gnn = GIN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], params['gin_mlp_layers'], params['dropout'], tasks, causal, "mean", True)
        elif self.model_type == "gin_moe":
            self.gnn = GIN_MoE(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], params['gin_mlp_layers'], params['dropout'], tasks, causal, "mean", True)
        elif self.model_type == "gcn_moe":
            self.gnn = GCN_MoE(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], activation, params['dropout'], tasks, causal)

        elif self.model_type in ["multi_graph","multi_graph_pred_moe"]:
            # self.gcn = GCN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], activation, params['dropout'], tasks, causal)
            # self.gat = GAT(params['n_layers'], hidden_dim, hidden_dim, hidden_dim, params['num_heads'], activation, params['dropout'], params['gat_attn_drop'], params['gat_neg_slope'], True, tasks, causal)
            self.gin = GIN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], params['gin_mlp_layers'], params['dropout'], tasks, causal, "mean", True)
        else:
            raise ValueError(f"Unknown model_type for HeteroProjectionGNN: {self.model_type}")
            
        # 3. 노드 타입 이름과 DGL이 부여한 정수 ID를 매핑
        self.ntype_map = {ntype: i for i, ntype in enumerate(g_hetero.ntypes)}
        self.in_dim = hidden_dim # GNN의 입력 차원은 hidden_dim
        # self.reduce_384To128 = nn.Linear(128*3, 128)
        # self.alpha = torch.nn.Parameter(torch.tensor(0.5))  # 초기값 0.5, 학습 가능

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        # 최종 GNN에 입력될, 프로젝션이 완료된 피처 텐서
        projected_feats = torch.zeros(g.num_nodes(), self.in_dim, device=g.device)

        # 2. 타입별로 순회하며 각각의 프로젝터를 적용
        for ntype, projector in self.projectors.items():
            type_id = self.ntype_map[ntype]
            mask = (g.ndata['_TYPE'] == type_id)
            
            # 원본 피처 슬라이싱 (제로 패딩 부분 제거)
            original_dim = projector.in_features
            original_h = features[mask, :original_dim]
            
            # 해당 타입의 프로젝터 통과
            projected_h = projector(original_h)
            
            # 결과 텐서에 저장
            projected_feats[mask] = projected_h
            
        # 3. GNN 백본에 통과시켜 최종 노드 임베딩(z)을 얻음
        if self.model_type in ["multi_graph","multi_graph_pred_moe"]:
            # z_gcn = self.gcn(g, projected_feats)
            # z_gat = self.gat(g, projected_feats)
            z = self.gin(g, projected_feats)
            # z = torch.cat([z_gat, z_gin], dim=-1)
            # z = self.reduce_384To128(z_cat)
            # z = self.alpha * z_gcn + (1-self.alpha) * z_gat
        else:
            z = self.gnn(g, projected_feats)
        return z
    
    # def forward_edges(self, g: dgl.DGLGraph, features: torch.Tensor, predictor: nn.Module, uv: Tuple) -> torch.Tensor:
    #     """ 학습 루프와의 호환성을 위한 엣지 포워드 메소드 """
    #     z = self.forward(g, features)
    #     return predictor(g, z, uv=uv)
    
    def forward_edges(self, g: dgl.DGLGraph, features: torch.Tensor, predictor: nn.Module, uv: Tuple) -> torch.Tensor:
        """ 학습 루프와의 호환성을 위한 엣지 포워드 메소드 """
        z = self.forward(g, features)
        # return predictor(g, z[0], uv=uv), z[1]        # moe 쓸 때
        return predictor(g, z, uv=uv)


class HeteroProjectionMoEGNN(nn.Module):
    """
    각 노드 타입(person, disease)의 피처를 타입별 프로젝션 레이어를 통해
    공통된 hidden_dim으로 맞춘 후, GNN + MoE 기반 백본을 통해
    노드 임베딩(z)을 계산하는 모델.
    """
    def __init__(self, person_in_dim: int, disease_in_dim: int, params: dict, g_hetero: dgl.DGLHeteroGraph):
        super().__init__()
        self.model_type = params['model_type']
        hidden_dim = params['hidden_dim']
        activation, tasks, causal = F.relu, [], False

        # 1️⃣ 타입별 projector
        self.projectors = nn.ModuleDict({
            'person': nn.Linear(person_in_dim, hidden_dim),
            'disease': nn.Linear(disease_in_dim, hidden_dim)
        })

        # 2️⃣ GNN 백본
        if self.model_type == "multi_graph":
            self.gcn = GCN(hidden_dim, hidden_dim, hidden_dim,
                           params['n_layers'], activation,
                           params['dropout'], tasks, causal)
            self.gat = GAT(params['n_layers'], hidden_dim, hidden_dim, hidden_dim,
                           params['num_heads'], activation, params['dropout'],
                           params['gat_attn_drop'], params['gat_neg_slope'],
                           True, tasks, causal)
            self.gin = GIN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], params['gin_mlp_layers'], params['dropout'], tasks, causal, "mean", True)
        else:
            raise ValueError(f"Unknown model_type for HeteroProjectionGNN: {self.model_type}")

        # 3️⃣ Expert Modules (Top-1, Expert 4개)
        self.k = params.get("topk", 1)
        self.expert_gcn = ExpertModule(hidden_dim, hidden_dim // 2,
                                       num_experts=4, k=self.k)
        self.expert_gat = ExpertModule(hidden_dim, hidden_dim // 2,
                                       num_experts=4, k=self.k)
        self.expert_gin = ExpertModule(hidden_dim, hidden_dim // 2,
                                       num_experts=4, k=self.k)

        # 타입 매핑
        self.ntype_map = {ntype: i for i, ntype in enumerate(g_hetero.ntypes)}
        self.in_dim = hidden_dim
        
        # 🔁 Gated multiplicative fusion 모듈
        # u = [z_gat; z_gin] (N, 2D) -> gate g (N, D)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # concat 쪽에서 오는 additive term u' 생성용
        self.fusion_linear = nn.Linear(hidden_dim * 2, hidden_dim)

    # -----------------------------
    # 🧠 projector 적용
    # -----------------------------
    def _project_and_embed(self, g, features):
        projected_feats = torch.zeros(g.num_nodes(), self.in_dim, device=g.device)
        for ntype, projector in self.projectors.items():
            type_id = self.ntype_map[ntype]
            mask = (g.ndata['_TYPE'] == type_id)
            original_dim = projector.in_features
            projected_feats[mask] = projector(features[mask, :original_dim])
        return projected_feats

    # -----------------------------
    # ⚙️ forward(): GNN + MoE
    # -----------------------------
    def forward(self, g: dgl.DGLGraph, features: torch.Tensor):
        """
        GNN 임베딩(z)을 계산하고, MoE layer 통과 후
        element-wise multiplication으로 최종 z를 반환.
        Load balancing loss 포함.
        """
        projected_feats = self._project_and_embed(g, features)

        # 1️⃣ GNN 기반 임베딩 계산
        # z_gcn = self.gcn(g, projected_feats)
        z_gat = self.gat(g, projected_feats)
        z_gin = self.gin(g, projected_feats)

        # 2️⃣ Expert MoE 통과
        # z_gcn_expert, aux_gcn = self.expert_gcn(z_gcn)
        z_gat_expert, aux_gat = self.expert_gat(z_gat)
        z_gin_expert, aux_gin = self.expert_gin(z_gin)

        # 3️⃣ Element-wise multiplication
        z = z_gin_expert * z_gat_expert


        ###################### Gated mul fusion ######################
        u = torch.cat([z_gat_expert, z_gin_expert], dim=-1)
        g = self.fusion_gate(u)

        u_prime = self.fusion_gate(u)
        z = g*z + (1.0 - g) * u_prime
        ###################### Gated mul fusion ######################


        # 4️⃣ load balancing loss 합산
        aux_loss_total = aux_gin + aux_gat
        # aux_loss_total = aux_gat

        # 5️⃣ forward_edges()와 호환성 유지 위해 튜플 반환
        # z = z_gat_expert
        return z, aux_loss_total

    # -----------------------------
    # 🔗 forward_edges(): 기존 구조 유지
    # -----------------------------
    def forward_edges(self, g: dgl.DGLGraph, features: torch.Tensor, predictor: nn.Module, uv: tuple):
        """
        학습 루프와의 호환성을 위한 엣지 포워드 메소드.
        forward()에서 z, aux_loss 계산 후 predictor 호출.
        """
        z, aux_loss = self.forward(g, features)
        pred = predictor(g, z, uv=uv)
        return pred, aux_loss




class HeteroProjectionGraphormer(nn.Module):
    """
    두 타입(person, disease) 피처를 각각 프로젝션 후 Graphormer 백본에 전달
    """
    def __init__(self, person_in_dim, disease_in_dim, params, g_hetero):
        super().__init__()
        hidden_dim = params['hidden_dim']
        self.in_dim = hidden_dim
        self.model_type = params['model_type']

        # 타입별 feature projection
        self.projectors = nn.ModuleDict({
            'person': nn.Linear(person_in_dim, hidden_dim),
            'disease': nn.Linear(disease_in_dim, hidden_dim)
        })

        # Graphormer backbone
        # self.encoder = GraphormerEncoder(
        #     hidden_dim=hidden_dim,
        #     num_heads=params.get('num_heads', 4),
        #     num_layers=params.get('n_layers', 2),
        #     dropout=params.get('dropout', 0.1)
        # )

        # Graphormer MoE backbone
        self.encoder = GraphormerEncoderMoE(
            hidden_dim=hidden_dim,
            num_heads=params.get('num_heads', 4),
            num_layers=params.get('n_layers', 1),
            dropout=params.get('dropout', 0.1)
        )

        # 노드 타입 매핑
        self.ntype_map = {ntype: i for i, ntype in enumerate(g_hetero.ntypes)}

    def forward(self, g, features):
        """
        g: DGLHeteroGraph (converted to homogeneous inside)
        features: [N, total_dim]
        """
        projected_feats = torch.zeros(g.num_nodes(), self.in_dim, device=g.device)

        # 타입별 projection
        for ntype, projector in self.projectors.items():
            type_id = self.ntype_map[ntype]
            mask = (g.ndata['_TYPE'] == type_id)
            original_dim = projector.in_features
            projected_feats[mask] = projector(features[mask, :original_dim])

        # DGL → edge_index 변환
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0).to(features.device)

        # Graphormer 인코딩
        # z = self.encoder(projected_feats, edge_index)
        # return z
        # GraphormerMoE 인코딩
        z, aux_loss = self.encoder(projected_feats, edge_index)
        return z, aux_loss

    # def forward_edges(self, g, features, predictor, uv):
    #     z = self.forward(g, features)
    #     logits = predictor(g, z, uv=uv)
    #     return logits
    
    def forward_edges(self, g, features, predictor, uv):
        z, aux_loss = self.forward(g, features)
        logits = predictor(g, z, uv=uv)
        return logits, aux_loss


class HeteroProjectionMoEFusion(nn.Module):
    def __init__(self, person_in_dim: int, disease_in_dim: int,
                 params: dict, g_hetero: dgl.DGLHeteroGraph):
        super().__init__()
        self.model_type = params['model_type']
        hidden_dim = params['hidden_dim']
        activation, tasks, causal = F.relu, [], False

        # 1️⃣ 타입별 projector
        self.projectors = nn.ModuleDict({
            'person': nn.Linear(person_in_dim, hidden_dim),
            'disease': nn.Linear(disease_in_dim, hidden_dim)
        })

        # 2️⃣ GNN 백본
        if self.model_type == "multi_graph_moe_fuse":
            self.gat = GAT(params['n_layers'], hidden_dim, hidden_dim, hidden_dim,
                           params['num_heads'], activation, params['dropout'],
                           params['gat_attn_drop'], params['gat_neg_slope'],
                           True, tasks, causal)
            self.gin = GIN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'],
                           params['gin_mlp_layers'], params['dropout'],
                           tasks, causal, "mean", True)
        else:
            raise ValueError(f"Unknown model_type for HeteroProjectionGNN: {self.model_type}")

        # 3️⃣ Fusion MoE
        self.k = params.get("topk", 1)
        self.fusion_moe = FusionMoE(
            dim=hidden_dim,
            num_experts=4,
            k=self.k,
            dropout=params.get("moe_dropout", 0.1),
            num_heads=params.get("fusion_num_heads", 4),
        )

        # 타입 매핑
        self.ntype_map = {ntype: i for i, ntype in enumerate(g_hetero.ntypes)}
        self.in_dim = hidden_dim

    def _project_and_embed(self, g, features):
        projected_feats = torch.zeros(g.num_nodes(), self.in_dim, device=g.device)
        for ntype, projector in self.projectors.items():
            type_id = self.ntype_map[ntype]
            mask = (g.ndata['_TYPE'] == type_id)
            original_dim = projector.in_features
            projected_feats[mask] = projector(features[mask, :original_dim])
        return projected_feats

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor):
        projected_feats = self._project_and_embed(g, features)

        # GNN 임베딩
        z_gat = self.gat(g, projected_feats)   # (N, D)
        z_gin = self.gin(g, projected_feats)   # (N, D)

        # Fusion MoE
        z, aux_loss = self.fusion_moe(z_gat, z_gin)  # (N, D), scalar

        return z, aux_loss

    def forward_edges(self, g: dgl.DGLGraph, features: torch.Tensor,
                      predictor: nn.Module, uv: tuple):
        z, aux_loss = self.forward(g, features)
        pred = predictor(g, z, uv=uv)
        return pred, aux_loss


# -----------------------------
# 모델 생성
# -----------------------------

def create_model_and_predictor(params: Dict, settings: Settings,
                               person_in_dim: int, disease_in_dim: int,
                               g_hetero: dgl.DGLHeteroGraph):
    
    # HGT는 이 설계와 맞지 않으므로 실행되지 않도록 처리
    if params['model_type'] == "hgt":
        raise ValueError(f"The HeteroProjectionGNN is not configured for HGT. Please use gcn, gat, or gin.")

    # 1. 래퍼 모델을 바로 생성합니다.
    if params['model_type'] in ["gcn", "gat", "gin", "multi_graph_pred_moe"]:
        model = HeteroProjectionGNN(person_in_dim, disease_in_dim, params, g_hetero)
    elif params['model_type'] in ["multi_graph", "gcn_moe", 'gat_moe','gin_moe'] and params["using_moe"]:
        model = HeteroProjectionMoEGNN(person_in_dim, disease_in_dim, params, g_hetero)
    elif params['model_type'] in ["multi_graph"] and not params["using_moe"]:
        model = HeteroProjectionGNN(person_in_dim, disease_in_dim, params, g_hetero)
    elif params['model_type'] in ["graphormer", "graphormer_moe"]:
        model = HeteroProjectionGraphormer(person_in_dim, disease_in_dim, params, g_hetero)
    elif params['model_type'] in ["multi_graph_moe_fuse"]:
        model = HeteroProjectionMoEFusion(person_in_dim, disease_in_dim, params, g_hetero)
    

    # 2. Predictor를 생성합니다.

    if params['model_type'] in ["multi_graph_pred_moe"]:
        predictor = EdgeMoEPredictor(params['hidden_dim'], params['pred_hidden']*2, 1, params['pred_dropout'], 4, top_k=params['top_k'])
    else:
        predictor = MLPPredictor(params['hidden_dim'], params['pred_hidden'], 1, params['pred_dropout'])
    return model, predictor