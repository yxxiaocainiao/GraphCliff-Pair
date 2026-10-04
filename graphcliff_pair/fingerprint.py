"""复用既有 Morgan 环境原子归属实现，保留节点顺序检查。"""
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from .vendor.dataset_utils import atom_features,bond_features

def membership(graph):
    """相同SMILES解析和Kekulize保持原子顺序；碰撞位取各环境原子并集。"""
    mol = Chem.MolFromSmiles(graph.smiles)
    Chem.Kekulize(mol, clearAromaticFlags=False)
    assert torch.equal(torch.tensor([atom_features(a) for a in mol.GetAtoms()]), graph.x)
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
        attrs.extend([bond_features(b), bond_features(b)])
    if edges:
        assert torch.equal(torch.tensor(edges).T, graph.edge_index)
        assert torch.equal(torch.tensor(attrs), graph.edge_attr)
    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol, additionalOutput=info)
    mask = torch.zeros(mol.GetNumAtoms(), 1024, dtype=torch.bool)
    for bit, occurrences in info.GetBitInfoMap().items():
        for center, radius in occurrences:
            atom_ids = {center}
            for bond_id in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center):
                b = mol.GetBondWithIdx(bond_id)
                atom_ids.update([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
            mask[list(atom_ids), bit] = True
    assert set(mask.any(0).nonzero().flatten().tolist()) == set(fp.GetOnBits())
    return mask
