"""Sequential adapter to the existing trainer; labels only from official train rows."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from graphcliff_pair import data, train
from experiments.long_branch_swap.run import safe_frame
from .model import ResidualFP, LayerSwap, SingleBaseline


def read_safe(path, split_seed=42):
    structure = pd.read_csv(path, usecols=["smiles", "split"])
    frame = safe_frame(path, structure.index[structure.split == "train"].tolist())
    reader = data.pd.read_csv
    def supplied(request, **kwargs):
        if Path(request).resolve() != Path(path).resolve():
            raise ValueError("unexpected CSV request")
        return frame.copy()
    data.pd.read_csv = supplied
    try:
        return data.read_development(path, split_seed)
    finally:
        data.pd.read_csv = reader


def run(config_path, csv_root, output):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    names = [a["name"] for a in cfg["arms"]]
    if len(set(names)) != len(names) or any(a["loss"] != "mse" for a in cfg["arms"]):
        raise ValueError("unique arms and plain MSE required")
    for index, choice in enumerate(cfg["arms"]):
        if choice.get("frozen") and "base" not in names[:index]:
            raise ValueError("frozen correction requires a preceding same-run base")
    original = train.PairRegressor, train.read_development, train.train_arm, train.predict, train.save_json, train.GraphCliffRegressor

    def factory(variant, h, layers, heads, readout):
        return LayerSwap(variant, h, layers, heads, "sag" if variant.endswith("_fp") else readout)

    def predict(model, variant, queries, references):
        if isinstance(model, ResidualFP):
            return model(queries)
        return original[3](model, variant, queries, references)

    def arm(config, choice, seed, frame, graphs, tp, vp, base_state, folder, device, scale):
        if not choice["variant"].startswith("fp_"):
            return original[2](config, choice, seed, frame, graphs, tp, vp, base_state, folder, device, scale)
        train.set_seed(SimpleNamespace(seed=seed))
        model = ResidualFP(config["hidden_size"], config["num_layers"],
                           choice["space"], choice.get("frozen", False), choice.get("shuffle", False))
        checkpoint = Path(folder).parent / "base/best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"] if model.frozen else base_state
        model.load_base(state)
        constructor = train.GraphCliffRegressor
        # The existing direct trainer loads this complete initial state; no partial checkpoint loads.
        train.GraphCliffRegressor = lambda *args, **kwargs: model
        try:
            result = original[2](config, dict(choice, variant="direct", readout="sag"), seed,
                                 frame, graphs, tp, vp, model.state_dict(), folder, device, scale)
        finally:
            train.GraphCliffRegressor = constructor
        result.update(task="single_molecule", space=model.space, frozen=model.frozen,
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      membership_null=model.shuffle,
                      baseline_checkpoint_sha256=data.digest(checkpoint) if model.frozen else None)
        original[4](Path(folder) / "summary.json", result)
        return result

    def save(path, value):
        if Path(path).name == "manifest.json":
            files = sorted(Path(__file__).parent.glob("*.py"))
            value["experiment_source_sha256"] = {str(p.relative_to(train.ROOT)):data.digest(p) for p in files}
            value["test_label_policy"] = "parse official train labels only; test is not evaluated"
            value["fppool_external_sha256"] = data.digest(train.ROOT / "external/fppool/pooling.py")
        original[4](path, value)
    train.PairRegressor, train.read_development, train.train_arm, train.predict, train.save_json = factory, read_safe, arm, predict, save
    train.GraphCliffRegressor = SingleBaseline
    try:
        train.run(config_path, csv_root, output)
    finally:
        train.PairRegressor, train.read_development, train.train_arm, train.predict, train.save_json, train.GraphCliffRegressor = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.csv_root, args.output)
