import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
from typing import Optional

from .GNN import GNN


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(GNN):
    """GIN model"""
    def __init__(self, in_dim, hidden_dim,
                 out_dim, num_layers, num_mlp_layers,
                 final_dropout, tasks,
                 causal, neighbor_pooling_type="mean", learn_eps=True):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        self.learn_eps = learn_eps
        self.num_mlp_layers = num_mlp_layers
        self.neighbor_pooling_type = neighbor_pooling_type

        super().__init__(in_dim, hidden_dim, out_dim, num_layers, F.relu, final_dropout, tasks, causal)

    # def forward(self, g: dgl.DGLHeteroGraph, nt, task):
    #     g = dgl.to_homogeneous(g, ndata=["feat"], store_type=True)
    #     g = dgl.add_self_loop(g)

    #     h = g.ndata["feat"]
    #     logits = self.get_logit(g, h)
    #     h = self.out[task](logits)
    #     out = h[g.ndata["_TYPE"] == 4]

    #     if self.causal:
    #         h = g.ndata["feat"]
    #         feat_rand = self.get_logit(g, h, True)
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

    def get_layers(self):

        layers = nn.ModuleList()

        #for layer in range(self.n_layers - 1):
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(self.num_mlp_layers, self.in_dim, self.hidden_dim, self.hidden_dim)
            else:
                mlp = MLP(self.num_mlp_layers, self.hidden_dim, self.hidden_dim, self.hidden_dim)

            layers.append(
                GINConv(ApplyNodeFunc(mlp), self.neighbor_pooling_type, 0, self.learn_eps))

        return layers

    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        self.set_embeddings(h)

        return h