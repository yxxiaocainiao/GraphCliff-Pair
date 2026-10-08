"""Preserve self response; learn a bounded signed cross-context correction."""
import torch
from .model import LayerSwap


def combine(self_response, cross_response, gain, anchor=True):
    correction = gain.tanh() * (cross_response - self_response)
    return self_response + correction if anchor else correction


class SelfAnchorSwap(LayerSwap):
    def __init__(self, hidden_size=256, num_layers=3, heads=4, readout="sag", anchor=True):
        super().__init__("retry_self", hidden_size, num_layers, heads, readout)
        self.anchor = anchor
        self.innovation_gain = torch.nn.Parameter(torch.zeros(num_layers))

    def forward(self, query, reference):
        if query.num_graphs != reference.num_graphs:
            raise ValueError("unequal pair batches")
        q, r = self.atom_encoder(query.x), self.atom_encoder(reference.x)
        for index, (layer, interaction) in enumerate(zip(self.encoder.layers, self.interactions)):
            q2, q1, qv = self.local(layer, q, query)
            r2, r1, rv = self.local(layer, r, reference)
            u, v = interaction(q2, r2, query.batch, reference.batch, "self")
            a, b = interaction(q2, r2, query.batch, reference.batch, "cross")
            uq, ur = self.transform(layer.long, u), self.transform(layer.long, v)
            cq, cr = self.transform(layer.long, a), self.transform(layer.long, b)
            q = q + layer.long.dropout(combine(uq, cq, self.innovation_gain[index], self.anchor)) * torch.sigmoid(q1) + qv
            r = r + layer.long.dropout(combine(ur, cr, self.innovation_gain[index], self.anchor)) * torch.sigmoid(r1) + rv
        difference = self.pool(q, query) - self.pool(r, reference)
        return (self.head(difference) - self.head(-difference)) / 2
