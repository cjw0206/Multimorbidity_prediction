"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from typing import Optional

from .GNN import GNN

class GCN(GNN):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 tasks,
                 causal=False):
        super().__init__(in_dim, hidden_dim, 
                         out_dim, 
                         n_layers, activation, dropout, tasks, causal)

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

    # def forward(self, g: dgl.DGLHeteroGraph, nt, task, person_node_type_id):
    #     g = dgl.to_homogeneous(g, ndata=["feat"], store_type=True) # heterogeneous -> homogeneous
    #     g = dgl.add_self_loop(g) # 모든 노드에 본인으로 향하게 하는 edge를 추가

    #     h = g.ndata["feat"] # 그래프 초기 특징
    #     logits = self.get_logit(g, h) # GNN 레이어를 통과시킴 
    #     h = self.out[task](logits) # task에 맞는 nn.linear에 통과시킴
    #     #out = h[g.ndata["_TYPE"] == 4]
    #     out = h[g.ndata["_TYPE"] == person_node_type_id] #하드코딩된 값 전달

    #     if self.causal:
    #         h = g.ndata["feat"]
    #         feat_rand = self.get_logit(g, h, True)
    #         feat_interv = logits + feat_rand
    #         out_interv = self.out[task](feat_interv)
    #         return out, feat_rand, out_interv

    #     return out

    # def forward(self, g, h, task, person_node_type_id): # <--- 인자 변경
    #     # to_homogeneous, add_self_loop 등은 train.py로 이동
    #     x = h
    #     logits = self.get_logit(g, x)
    #     node_logits = self.out[task](logits)
    #     out = node_logits[g.ndata["_TYPE"] == person_node_type_id]

    #     if self.causal:
    #         feat_rand = self.get_logit(g, x, True)
    #         feat_interv = logits + feat_rand
    #         out_interv = self.out[task](feat_interv)
    #         return out, feat_rand, out_interv
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

    # def forward_edges(self, g, h, predictor, uv=None, **kwargs):
    #     """
    #     GNN으로 노드 임베딩 z를 생성하고,
    #     predictor가 필요로 하는 모든 인자(z, uv, L_p, L_d, maps 등)를
    #     **kwargs를 통해 그대로 전달합니다.
    #     """
    #     z = self.encode(g, h) # encode 메소드는 GNN.py에 이미 정의되어 있습니다.
        
    #     # predictor 호출 시 uv와 함께 **kwargs를 넘겨주는 것이 핵심
    #     return predictor(g, z, uv=uv, **kwargs)

    def forward_edges(self, g, h, predictor, eids=None, uv=None, causal: bool = False,
                  fixed_nid: Optional[int] = None, use_interaction: bool = False):
        z = self.encode(g, h, causal=causal)        
        return predictor(g, z, eids=eids, uv=uv, fixed_nid=fixed_nid)

    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        self.set_embeddings(h)

        return h

    # 가중치 추가 로짓
    # def get_logit(self, g, h, causal=False):
    #     layers = self.layers if not causal else self.rand_layers
    #     # 그래프 엣지에 'weight'가 있다면 사용
    #     weights = g.edata['weight'] if 'weight' in g.edata else None

    #     for i, layer in enumerate(layers):
    #         if i != 0:
    #             h = self.dropout(h)
    #         # edge_weight 인자 추가
    #         h = layer(g, h, edge_weight=weights)

    #     self.set_embeddings(h)
    #     return h

    def get_layers(self):
        layers = nn.ModuleList()
        layers.append(GraphConv(self.in_dim, self.hidden_dim, activation=self.activation))
        # hidden layers
        for i in range(self.n_layers - 1):
            layers.append(GraphConv(self.hidden_dim, self.hidden_dim, activation=self.activation))

        return layers
