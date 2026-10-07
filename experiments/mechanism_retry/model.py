"""Two independent candidates: scalar FP correction and centered layer interaction."""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_geometric.utils import to_dense_batch, subgraph
from torch_geometric.data import Data
from graphcliff_pair.model import PairRegressor
from graphcliff_pair.fppool import FPPoolReadout
from graphcliff_pair.vendor.model import GraphCliffRegressor
from experiments.long_branch_swap.model import BranchSwap


class SingleBaseline(GraphCliffRegressor):
    """Upstream parameter names; use the existing empty-graph batching guard."""
    readout = "sag"

    def forward(self, x_f, edge_index, edge_attr, batch):
        graph = Data(x=x_f, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        counts = torch.bincount(batch[edge_index[0]], minlength=int(batch.max()) + 1)
        if (counts == 0).any():
            values = []
            for i in range(len(counts)):
                nodes = batch == i
                edges, attrs = subgraph(nodes, edge_index, edge_attr, relabel_nodes=True)
                values.append(self.encoder(self.atom_encoder(x_f[nodes]), edges, attrs))
            x = torch.cat(values)
        else:
            x = self.encoder(self.atom_encoder(x_f), edge_index, edge_attr)
        return self.reg_head(PairRegressor.pool(self, x, graph))


class ResidualFP(PairRegressor):
    """Single-molecule prediction; reference labels never enter this module."""
    def __init__(self, hidden_size=256, num_layers=3, space="output", frozen=False,
                 shuffle=False):
        if space not in ("feature", "output"):
            raise ValueError("unknown residual space")
        super().__init__(hidden_size=hidden_size, num_layers=num_layers)
        self.space, self.frozen, self.shuffle = space, frozen, shuffle
        if frozen:
            self.requires_grad_(False)
        self.fp = FPPoolReadout(hidden_size)
        if space == "feature":
            self.alpha = nn.Parameter(torch.zeros(()))
        else:
            self.correction = nn.Linear(2 * hidden_size, 1, bias=False)
            nn.init.zeros_(self.correction.weight)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            for module in (self.atom_encoder, self.encoder, self.sagpool, self.head):
                module.eval()
        return self

    def forward(self, graph, reference=None):
        x = self.encode(graph)
        original = self.pool(x, graph)
        fp = graph.atom_fp
        if self.shuffle:
            # ponytail: fixed row reversal is one null, use preregistered random permutations for confirmation.
            fp = torch.cat([fp[graph.batch == i].flip(0) for i in range(graph.num_graphs)])
        auxiliary = self.fp(x, graph.batch, fp)
        if self.space == "feature":
            return self.head(original + self.alpha * auxiliary)
        return self.head(original) + self.correction(auxiliary)

    def load_base(self, state):
        # PairRegressor maps the upstream head indices and checks shared tensor shapes.
        self.load_shared(state)


class LayerInteraction(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=0, batch_first=True)

    def forward(self, q, r, qb, rb, mode):
        if mode not in ("self", "cross", "centered"):
            raise ValueError("unknown interaction mode")
        qd, qm = to_dense_batch(q, qb)
        rd, rm = to_dense_batch(r, rb)
        if qd.shape[0] != rd.shape[0]:
            raise ValueError("unequal pair batches")
        def attend(a, b, mask):
            return self.attention(a, b, b, key_padding_mask=~mask, need_weights=False)[0]
        with sdpa_kernel(SDPBackend.MATH):
            if mode == "self":
                a, b = attend(qd, qd, qm), attend(rd, rd, rm)
            else:
                a, b = attend(qd, rd, rm), attend(rd, qd, qm)
                if mode == "centered":
                    a, b = a - attend(qd, qd, qm), b - attend(rd, rd, rm)
        return a[qm], b[rm]


class LayerSwap(BranchSwap):
    def __init__(self, variant, hidden_size=256, num_layers=3, heads=4, readout="sag"):
        mode = variant.removeprefix("retry_").removesuffix("_fp")
        if mode not in ("full", "self", "cross", "centered") or readout != "sag":
            raise ValueError("unsupported retry")
        super().__init__("branch_full", hidden_size, num_layers, heads, readout)
        self.retry_mode = mode
        if mode != "full":
            self.interactions = nn.ModuleList([LayerInteraction(hidden_size, heads) for _ in self.encoder.layers])
            for layer in self.encoder.layers:
                # Preserve LongPoly's scale, bias, RMSNorm, SiLU and dropout, remove only its polynomial coefficients.
                del layer.long.cheb_coeffs
        if variant.endswith("_fp"):
            self.fp = FPPoolReadout(hidden_size)
            self.correction = nn.Linear(2 * hidden_size, 1, bias=False)
            nn.init.zeros_(self.correction.weight)

    @staticmethod
    def transform(module, x):
        grouped = x.reshape(-1, module.groups, module.group_channels)
        scaled = grouped * module.group_scale[None, :, None] + module.group_bias[None, :, None]
        return module.activation(module.norm(scaled.flatten(1)))

    def forward(self, query, reference):
        if query.num_graphs != reference.num_graphs:
            raise ValueError("unequal pair batches")
        if self.retry_mode == "full":
            q, r = self.encode(query), self.encode(reference)
        else:
            q, r = self.atom_encoder(query.x), self.atom_encoder(reference.x)
            for layer, interaction in zip(self.encoder.layers, self.interactions):
                q2, q1, qv = self.local(layer, q, query)
                r2, r1, rv = self.local(layer, r, reference)
                a, b = interaction(q2, r2, query.batch, reference.batch, self.retry_mode)
                a, b = self.transform(layer.long, a), self.transform(layer.long, b)
                if self.retry_mode == "centered":
                    # Center the affine/nonlinear transform as well: nonzero learned bias cannot break T(0)=0.
                    a = a - self.transform(layer.long, torch.zeros_like(q2))
                    b = b - self.transform(layer.long, torch.zeros_like(r2))
                q = q + layer.long.dropout(a) * torch.sigmoid(q1) + qv
                r = r + layer.long.dropout(b) * torch.sigmoid(r1) + rv
        difference = self.pool(q, query) - self.pool(r, reference)
        result = (self.head(difference) - self.head(-difference)) / 2
        if hasattr(self, "fp"):
            result = result + self.correction(self.fp(q, query.batch, query.atom_fp) -
                                              self.fp(r, reference.batch, reference.atom_fp))
        return result
