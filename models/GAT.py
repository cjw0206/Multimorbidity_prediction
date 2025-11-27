"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn

import dgl
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
from typing import Optional

from .GNN import GNN

class GAT(GNN):
    def __init__(self,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 tasks,
                 causal):

        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual

        super().__init__(in_dim, hidden_dim, 
                         out_dim, 
                         n_layers, activation, feat_drop, tasks, causal) 

    # def forward(self, g: dgl.DGLHeteroGraph, nt, task, person_node_type_id):
    #     g = dgl.to_homogeneous(g, ndata=["feat"], store_type=True)
    #     g = dgl.add_self_loop(g)

    #     h = g.ndata["feat"]
    #     logits = self.get_logit(g, h)
    #     h = self.out[task](logits)
    #     #out = h[g.ndata["_TYPE"] == 4]
    #     out = h[g.ndata["_TYPE"] == person_node_type_id]

    #     if self.causal:
    #         h = g.ndata["feat"]
    #         feat_rand = self.get_logit(g, h, True)
    #         feat_interv = logits + feat_rand
    #         out_interv = self.out[task](feat_interv)
    #         return out, feat_rand, out_interv

    #     self.set_embeddings(h)

    #     return out

    def forward(self, g, h, task=None, person_node_type_id=None):
        z = self.get_logit(g, h)  # 전체 노드 임베딩
        # 링크 예측(엣지 분류)일 때는 헤드 없이 임베딩만 반환
        if task is None or task == "link_pred":
            return z

        node_logits = self.out[task](z)
        if person_node_type_id is not None:
            node_logits = node_logits[g.ndata["_TYPE"] == person_node_type_id]
        return node_logits

    def forward_edges(self, g, h, predictor, eids=None, uv=None, causal: bool = False,
                  fixed_nid: Optional[int] = None, use_interaction: bool = False):
        z = self.encode(g, h, causal=causal)
        return predictor(g, z, eids=eids, uv=uv, fixed_nid=fixed_nid)

    # def forward_edges(self, g, h, predictor, uv=None, **kwargs):
    #     """
    #     GNN으로 노드 임베딩 z를 생성하고,
    #     predictor가 필요로 하는 모든 인자(z, uv, L_p, L_d, maps 등)를
    #     **kwargs를 통해 그대로 전달합니다.
    #     """
    #     z = self.encode(g, h) # encode 메소드는 GNN.py에 이미 정의되어 있습니다.
        
    #     # predictor 호출 시 uv와 함께 **kwargs를 넘겨주는 것이 핵심
    #     return predictor(g, z, uv=uv, **kwargs)

    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if i != len(layers) - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)

        #self.embeddings = h
        self.set_embeddings(h)

        return h

    # def get_layers(self):

    #     layers = nn.ModuleList()
    #     for l in range(self.n_layers):
    #         if l == 0:
    #             # input projection (no residual)
    #             layers.append(GATConv(
    #                 self.in_dim, self.hidden_dim, self.heads[0],
    #                 self.dor, self.attn_drop, self.negative_slope, False, self.activation))
    #         else:
    #             # due to multi-head, the in_dim = num_hidden * num_heads
    #             layers.append(GATConv(
    #                 self.hidden_dim * self.heads[l-1], self.hidden_dim, self.heads[l],
    #                 self.dor, self.attn_drop, self.negative_slope, self.residual, self.activation))

    #     return layers
    
    def get_layers(self):
    
        layers = nn.ModuleList()
        num_heads = self.num_heads

        for l in range(self.n_layers):

            # 각 레이어 입력 차원 결정
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim * num_heads

            cur_heads =1 if l == self.n_layers -1 else num_heads        # 11.21 마지막 레이어는 head 하나로 변경.
            # GATConv 추가
            layers.append(GATConv(
                in_dim,
                self.hidden_dim,
                cur_heads,
                self.dor,
                self.attn_drop,
                self.negative_slope,
                self.residual if l > 0 else False,
                self.activation
            ))

        return layers