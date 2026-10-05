"""Replace only LongPoly, retaining the upstream short path, gates and pooling."""
import torch
from torch import nn
from graphcliff_pair.model import PairRegressor, CrossInteraction

class BranchSwap(PairRegressor):
    def __init__(self, variant, hidden_size=256, num_layers=3, heads=4, readout="sag"):
        self.mode = variant.removeprefix("branch_")
        if self.mode not in ("full", "short", "cross") or readout != "sag":
            raise ValueError("unsupported branch experiment")
        super().__init__("global_diff", hidden_size, num_layers, heads, readout)
        if self.mode != "full":
            for layer in self.encoder.layers:
                layer.long = nn.Identity()
        if self.mode == "cross":
            self.branch_interactions = nn.ModuleList([CrossInteraction(hidden_size, heads) for _ in range(num_layers)])

    @staticmethod
    def local(layer, x, graph):
        return torch.chunk(layer.short(layer.proj(layer.pre_norm(x)), graph.edge_index, graph.edge_attr), 3, dim=-1)

    def forward(self, query, reference):
        if self.mode == "full":
            return super().forward(query, reference)
        if query.num_graphs != reference.num_graphs:
            raise ValueError("unequal pair batches")
        q, r = self.atom_encoder(query.x), self.atom_encoder(reference.x)
        for i, layer in enumerate(self.encoder.layers):
            q2, q1, qv = self.local(layer, q, query)
            r2, r1, rv = self.local(layer, r, reference)
            if self.mode == "cross":
                q2, r2 = self.branch_interactions[i](q2, r2, query.batch, reference.batch)
            q, r = q + q2*torch.sigmoid(q1) + qv, r + r2*torch.sigmoid(r1) + rv
        delta = self.pool(q, query) - self.pool(r, reference)
        return (self.head(delta) - self.head(-delta))/2
