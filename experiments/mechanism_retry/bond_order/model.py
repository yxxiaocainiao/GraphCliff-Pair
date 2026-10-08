"""Optional typed two-hop basis inside LongPoly; original code remains unchanged."""
import itertools

from rdkit import Chem
import torch
from torch import nn

from experiments.mechanism_retry.model import LayerSwap
from graphcliff_pair.vendor.model import LongPoly, normalize_edges, propagate

PAIRS = tuple(itertools.combinations(range(5), 2))


def attach_bond_kind(graph, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != graph.x.size(0):
        raise ValueError("SMILES and graph atom order contract required")
    types = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
    labels = {}
    for bond in mol.GetBonds():
        category = 3 if bond.GetIsAromatic() else types.index(bond.GetBondType()) if bond.GetBondType() in types else 4
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        labels[i, j] = labels[j, i] = category
    edges = list(zip(*graph.edge_index.cpu().tolist()))
    if set(edges) != set(labels) or len(edges) != len(labels):
        raise ValueError("raw molecule and graph directed bonds differ")
    graph.bond_kind = torch.tensor([labels[e] for e in edges], dtype=torch.long, device=graph.x.device)
    return graph


class BondLongPoly(LongPoly):
    def __init__(self, original, mode="skew"):
        if mode not in ("skew", "symmetric"):
            raise ValueError("unknown ordered-path basis")
        super().__init__(original.groups*original.group_channels, original.K, original.groups,
                         original.dropout.p if isinstance(original.dropout, nn.Dropout) else 0)
        self.mode = mode
        self.order_coeffs = nn.Parameter(torch.zeros(original.groups, len(PAIRS)))
        loaded = self.load_state_dict(original.state_dict(), strict=False)
        assert loaded.missing_keys == ["order_coeffs"] and not loaded.unexpected_keys

    def forward(self, x, edge_index, weight, kind):
        if kind.ndim != 1 or kind.numel() != edge_index.size(1) or kind.dtype != torch.long:
            raise ValueError("one integer bond_kind per directed edge required")
        if kind.numel() and (kind.min() < 0 or kind.max() > 4):
            raise ValueError("bond_kind outside fixed five categories")
        n, h = x.shape
        # The original forward does not expose its pre-RMSNorm sum. Reproduce
        # only that recurrence here; reuse all original parameters and propagator.
        grouped = x.view(n, self.groups, self.group_channels)
        result = self.cheb_coeffs[:, 0].view(1, -1, 1)*grouped
        if edge_index.numel() and self.K >= 1:
            prev, current = x, propagate(x, edge_index, weight)
            result += self.cheb_coeffs[:, 1].view(1, -1, 1)*current.view_as(grouped)
            for k in range(2, self.K+1):
                prev, current = current, 2*propagate(current, edge_index, weight)-prev
                result += self.cheb_coeffs[:, k].view(1, -1, 1)*current.view_as(grouped)
        present = [b for b in range(5) if bool((kind == b).any())]
        slices = {b: (edge_index[:, kind == b], weight[kind == b]) for b in present}
        first = {b: propagate(x, *slices[b]) for b in present}
        for j, (b, c) in enumerate(PAIRS):
            if b in slices and c in slices:
                bc = propagate(first[c], *slices[b])
                cb = propagate(first[b], *slices[c])
                basis = (bc-cb if self.mode == "skew" else bc+cb)/2
                result += self.order_coeffs[:, j].view(1, -1, 1)*basis.view_as(grouped)
        result = result*self.group_scale.view(1, -1, 1)+self.group_bias.view(1, -1, 1)
        return self.dropout(self.activation(self.norm(result.reshape(n, h))))


class BondOrderPair(LayerSwap):
    def __init__(self, hidden_size=256, num_layers=3, heads=4, mode="skew"):
        super().__init__("retry_full", hidden_size, num_layers, heads, "sag")
        for layer in self.encoder.layers:
            layer.long = BondLongPoly(layer.long, mode)

    def encode_one(self, graph):
        x = self.atom_encoder(graph.x)
        weight = normalize_edges(len(x), graph.edge_index, x.new_ones(graph.edge_index.size(1)))
        for layer in self.encoder.layers:
            x2, x1, v = self.local(layer, x, graph)
            x = (layer.long(x2, graph.edge_index, weight, graph.bond_kind)*torch.sigmoid(x1) + v) + x
        return x

    def encode(self, graph):
        counts = torch.bincount(graph.batch[graph.edge_index[0]], minlength=graph.num_graphs)
        if (counts == 0).any():
            return torch.cat([self.encode_one(g) for g in graph.to_data_list()])
        return self.encode_one(graph)
