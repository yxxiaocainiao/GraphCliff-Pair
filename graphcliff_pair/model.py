"""薄适配：复用 GraphCliff 编码/读出，注意力由 PyTorch 实现。"""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from .vendor.model import GraphCliffRegressor

def parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class CrossInteraction(nn.Module):
    def __init__(self, hidden_size, heads=4):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size 必须能被 heads 整除")
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, q, r, qb, rb):
        q_dense, qm = to_dense_batch(q, qb)
        r_dense, rm = to_dense_batch(r, rb)
        if q_dense.size(0) != r_dense.size(0):
            raise ValueError("查询和参考图数量必须相同")
        # K/V 来自配对的另一分子；batch 维度不参与注意力混合。
        with sdpa_kernel(SDPBackend.MATH):
            qr, _ = self.attention(q_dense, r_dense, r_dense, key_padding_mask=~rm, need_weights=False)
            rq, _ = self.attention(r_dense, q_dense, q_dense, key_padding_mask=~qm, need_weights=False)
        return self.norm(q_dense + qr)[qm], self.norm(r_dense + rq)[rm]

class PairRegressor(nn.Module):
    def __init__(self, variant="global_diff", hidden_size=256, num_layers=3, heads=4, readout="sag"):
        super().__init__()
        if variant not in ("global_diff", "pair_mlp", "cross_attention"):
            raise ValueError("未知 variant")
        if readout not in ("sag", "fppool"):
            raise ValueError("未知 readout")
        self.variant, self.readout = variant, readout
        base = GraphCliffRegressor(38, 13, hidden_size=hidden_size, num_layers=num_layers, dropout=0)
        self.atom_encoder, self.encoder = base.atom_encoder, base.encoder
        if readout == "sag":
            self.sagpool = base.sagpool
        else:
            from .fppool import FPPoolReadout
            self.fppool = FPPoolReadout(hidden_size)
        h = hidden_size
        if variant == "cross_attention":
            self.interaction = CrossInteraction(h, heads)
        input_dim = 6*h if variant == "pair_mlp" else 2*h
        width = h//2
        if variant == "pair_mlp":
            # 与同读出的 attention 组匹配新增可训练参数，不声称计算量相同。
            target = parameters(base.reg_head) + 4*h*h + 6*h
            width = max(1, round((target - 1) / (input_dim + 4)))
        self.head = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))

    def encode(self, graph):
        edge_counts = torch.bincount(graph.batch[graph.edge_index[0]], minlength=graph.num_graphs)
        if (edge_counts == 0).any():
            # 官方 LongPoly 的空边分支与混合批次的零度节点分支不同。
            # 含无键分子的 batch 单图编码，避免其结果随其他分子改变。
            return torch.cat([self.encoder(self.atom_encoder(g.x), g.edge_index, g.edge_attr)
                              for g in graph.to_data_list()], dim=0)
        return self.encoder(self.atom_encoder(graph.x), graph.edge_index, graph.edge_attr)

    def pool(self, x, graph):
        if self.readout == "fppool":
            return self.fppool(x, graph.batch, graph.atom_fp)
        nodes, _, _, batch, _, _ = self.sagpool(x, graph.edge_index, graph.edge_attr, graph.batch)
        return torch.cat([global_max_pool(nodes, batch), global_mean_pool(nodes, batch)], dim=-1)

    def forward(self, query, reference):
        if query.num_graphs != reference.num_graphs:
            raise ValueError("查询和参考图数量必须相同")
        q, r = self.encode(query), self.encode(reference)
        if self.variant == "cross_attention":
            q, r = self.interaction(q, r, query.batch, reference.batch)
        gq, gr = self.pool(q, query), self.pool(r, reference)
        if self.variant == "pair_mlp":
            qr = torch.cat([gq, gr, gq-gr], dim=-1)
            rq = torch.cat([gr, gq, gr-gq], dim=-1)
        else:
            qr, rq = gq-gr, gr-gq
        # 对所有 pair 组使用同一反对称规则，无额外一致性 loss。
        delta = (self.head(qr) - self.head(rq)) / 2
        if delta.shape != (query.num_graphs, 1) or not torch.isfinite(delta).all():
            raise RuntimeError("差值预测必须为有限 [B,1]")
        return delta

    def load_shared(self, base_state):
        names = ("atom_encoder.", "encoder.", "sagpool.")
        own = self.state_dict()
        selected = {k: v for k, v in base_state.items() if k.startswith(names) and k in own}
        result = self.load_state_dict(selected, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(result.unexpected_keys)
        if self.variant != "pair_mlp":
            # 官方末层索引为4（含无参数 Dropout），本包装为3。
            head_state = {key.replace("reg_head.", "").replace("4.", "3."): value
                          for key, value in base_state.items() if key.startswith("reg_head.")}
            self.head.load_state_dict(head_state)
        return selected
