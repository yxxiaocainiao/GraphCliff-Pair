"""复用官方分子特征及既有划分；仅新增严格检查与结构近邻配对。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

from .vendor.dataset_utils import smiles_to_graph
from .vendor.protocol import split_train_valid

FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_development(path, split_seed=42):
    # 不创建 test 图，不构造 test loader；只读取必要的 train/valid 列。
    frame = pd.read_csv(path, usecols=["smiles", "y", "split", "cliff_mol"])
    if not frame["split"].isin(["train", "test"]).all():
        raise ValueError("首轮仅支持官方 train/test 文件，不接受其他划分")
    train_rows, valid_rows = split_train_valid(frame, 0.1, split_seed)
    development = frame.loc[np.concatenate([train_rows, valid_rows])].copy()
    if len(train_rows) < 2 or not len(valid_rows):
        raise ValueError("训练/验证支持量不足")
    if not np.isfinite(development.y.to_numpy(dtype=float)).all():
        raise ValueError("开发集活性包含非有限值")
    if not development.cliff_mol.isin([0, 1]).all():
        raise ValueError("cliff_mol 必须为二值")
    canonical, molecules, graphs = {}, {}, {}
    for row in sorted(development.index):
        mol = Chem.MolFromSmiles(frame.at[row, "smiles"])
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"source_row={row} 无法构图，停止而非丢弃")
        canonical[row] = Chem.MolToSmiles(mol, canonical=True)
        molecules[row] = mol
        graph = smiles_to_graph(frame.at[row, "smiles"], frame.at[row, "y"])
        if graph.x.ndim != 2 or graph.x.shape != (mol.GetNumAtoms(), 38) or graph.edge_attr.shape[1] != 13:
            raise ValueError(f"source_row={row} 原子/键特征不匹配")
        graphs[row] = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
    shared = {canonical[int(i)] for i in train_rows} & {canonical[int(i)] for i in valid_rows}
    if shared:
        raise ValueError(f"train/valid 有 {len(shared)} 个 canonical 重复分子；拒绝静默改划分")
    fps = {row: FP.GetFingerprint(mol) for row, mol in molecules.items()}
    train_pairs = top1(train_rows, train_rows, canonical, fps)
    valid_pairs = top1(valid_rows, train_rows, canonical, fps)
    return frame, graphs, train_rows, valid_rows, train_pairs, valid_pairs

def top1(queries, references, canonical, fps):
    # 严格排序保障相似度并列时选择最小原始行号，且选择过程不接受活性。
    references = sorted(int(row) for row in references)
    result = []
    for row in sorted(int(i) for i in queries):
        candidates = [i for i in references if canonical[i] != canonical[row]]
        if not candidates:
            raise ValueError(f"source_row={row} 没有合法参考")
        scores = DataStructs.BulkTanimotoSimilarity(fps[row], [fps[i] for i in candidates])
        k = int(np.argmax(scores))
        result.append({"query": row, "reference": candidates[k], "similarity": float(scores[k])})
    return result

def prepare(path, output, split_seed=42):
    frame, graphs, tr, va, tp, vp = read_development(path, split_seed)
    manifest = dict(dataset=Path(path).stem, input_sha256=digest(path), split_seed=split_seed,
                    train_rows=tr.tolist(), valid_rows=va.tolist(), train_pairs=tp, valid_pairs=vp,
                    train_count=len(tr), valid_count=len(va), graph_count=len(graphs),
                    train_valid_canonical_overlap=0, test_evaluated=False,
                    fp_radius=2, fp_bits=1024, reference_policy="train-only top1; canonical exclusion; tie by source_row")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pairs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["dataset", "train_count", "valid_count", "graph_count", "test_evaluated"]}), flush=True)
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.csv, args.output)

if __name__ == "__main__":
    main()
