"""Thin, sequential adapter around the existing trainer; no test labels parsed."""
import argparse
import json
from pathlib import Path
import pandas as pd
from graphcliff_pair import data, train
from .model import BranchSwap
ROOT = train.ROOT

def safe_frame(path, rows):
    structure = pd.read_csv(path, usecols=["smiles", "split"])
    allowed = sorted(set(rows))
    allowed_set = set(allowed)
    if set(structure.index[structure.split == "train"]) != set(allowed):
        raise ValueError("frozen development rows differ from official training partition")
    labels = pd.read_csv(path, usecols=["y", "cliff_mol"], skiprows=lambda line: line > 0 and line-1 not in allowed_set)
    labels.index = allowed
    return structure.join(labels)

def frozen_read(path, split_seed=42):
    source = ROOT / "artifacts/interaction_seed42_20261004" / Path(path).stem / "pairs.json"
    frozen = json.loads(source.read_text(encoding="utf-8"))
    if data.digest(path) != frozen["input_sha256"]:
        raise ValueError("source CSV changed")
    frame = safe_frame(path, frozen["train_rows"] + frozen["valid_rows"])
    reader = data.pd.read_csv
    def supplied(request, **kwargs):
        if Path(request).resolve() != Path(path).resolve():
            raise ValueError("unexpected CSV request")
        return frame.copy()
    data.pd.read_csv = supplied
    try:
        result = data.read_development(path, split_seed)
    finally:
        data.pd.read_csv = reader
    for key, actual in zip(["train_rows", "valid_rows", "train_pairs", "valid_pairs"], result[2:]):
        value = actual.tolist() if hasattr(actual, "tolist") else actual
        if value != frozen[key]:
            raise ValueError("frozen rows/pairs changed: " + key)
    return result

def run(config, csv_root, output):
    original = train.PairRegressor, train.read_development, train.save_json
    def save(path, value):
        if Path(path).name == "manifest.json":
            value["experiment_source_sha256"] = {str(p.relative_to(ROOT)):data.digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
            value["frozen_pair_sha256"] = {d:data.digest(ROOT / "artifacts/interaction_seed42_20261004" / d / "pairs.json") for d in value["config"]["datasets"]}
            value["test_label_policy"] = "parse labels only on frozen official-train rows"
        original[2](path, value)
    train.PairRegressor, train.read_development, train.save_json = BranchSwap, frozen_read, save
    try:
        train.run(config, csv_root, output)
    finally:
        train.PairRegressor, train.read_development, train.save_json = original

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--csv-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.config, a.csv_root, a.output)
