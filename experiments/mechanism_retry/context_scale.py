"""One operator change: cross and self use the same self-context scale."""
import argparse
import torch
from . import run as retry
from .model import LayerSwap


def contextual_difference(module, self_output, cross_output):
    def affine(x):
        grouped = x.reshape(-1, module.groups, module.group_channels)
        return (grouped * module.group_scale[None, :, None] + module.group_bias[None, :, None]).flatten(1)
    anchor, cross = affine(self_output), affine(cross_output)
    eps = torch.finfo(anchor.dtype).eps if module.norm.eps is None else module.norm.eps
    scale = (anchor.square().mean(-1, keepdim=True) + eps).sqrt()
    # No detach: the common self-context scale remains differentiable.
    return module.activation(module.norm.weight * cross / scale) - module.activation(module.norm.weight * anchor / scale)


class ContextScaleSwap(LayerSwap):
    def __init__(self, hidden_size=256, num_layers=3, heads=4, readout="sag"):
        super().__init__("retry_centered", hidden_size, num_layers, heads, readout)

    def forward(self, query, reference):
        if query.num_graphs != reference.num_graphs:
            raise ValueError("unequal pair batches")
        q, r = self.atom_encoder(query.x), self.atom_encoder(reference.x)
        for layer, interaction in zip(self.encoder.layers, self.interactions):
            q2, q1, qv = self.local(layer, q, query)
            r2, r1, rv = self.local(layer, r, reference)
            a, b = interaction(q2, r2, query.batch, reference.batch, "cross")
            u, v = interaction(q2, r2, query.batch, reference.batch, "self")
            q = q + layer.long.dropout(contextual_difference(layer.long, u, a)) * torch.sigmoid(q1) + qv
            r = r + layer.long.dropout(contextual_difference(layer.long, v, b)) * torch.sigmoid(r1) + rv
        difference = self.pool(q, query) - self.pool(r, reference)
        return (self.head(difference) - self.head(-difference)) / 2


def run(config, csv_root, output):
    original = retry.LayerSwap
    def factory(variant, *args, **kwargs):
        return ContextScaleSwap(*args, **kwargs) if variant == "retry_context" else original(variant, *args, **kwargs)
    retry.LayerSwap = factory
    try:
        retry.run(config, csv_root, output)
    finally:
        retry.LayerSwap = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.csv_root, args.output)
