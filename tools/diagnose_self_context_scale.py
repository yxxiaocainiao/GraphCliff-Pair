"""Observe real centered checkpoints without replacing their forward outputs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_dense_batch

from experiments.mechanism_retry.audit import audit
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.run import read_safe
from graphcliff_pair.data import digest
from graphcliff_pair.train import ROOT, batches, set_seed
from tools.check_self_context_scale import contextual_difference


def describe(values):
    values = np.concatenate(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("no finite observations")
    return dict(count=len(values), finite_count=len(finite), mean=float(finite.mean()),
                quantiles={str(q): float(np.quantile(finite, q)) for q in (0, .01, .1, .5, .9, .99, 1)})


def diagnose(output):
    output = Path(output)
    checked = audit(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = manifest["config"]
    preflight = json.loads((ROOT / "experiments/mechanism_retry/layer_preflight_20261007.json").read_text())
    if cfg["seeds"] != [42] or manifest["config_sha256"] != preflight["core_config_sha256"]:
        raise ValueError("unexpected pilot")
    if str(torch.__version__) != manifest["torch"]:
        raise ValueError("torch environment changed")
    prior = json.loads((ROOT / "experiments/mechanism_retry/layer_results_20261007.json").read_text())
    for relative, source in prior["inherited_dependencies"].items():
        if digest(ROOT / relative) != source["workspace_sha256"]:
            raise ValueError("inherited source changed")
    torch.set_num_threads(2)
    set_seed(SimpleNamespace(seed=42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, replays = [], []
    for source in preflight["datasets"]:
        dataset = source["dataset"]
        if digest(source["input_path"]) != source["input_sha256"]:
            raise ValueError("input changed")
        frame, graphs, tr, va, tp, vp = read_safe(source["input_path"], cfg["split_seed"])
        pairs = json.loads((output / dataset / "pairs.json").read_text())
        if tr.tolist() != pairs["train_rows"] or va.tolist() != pairs["valid_rows"] or tp != pairs["train_pairs"] or vp != pairs["valid_pairs"]:
            raise ValueError("split or pairing changed")
        folder = output / dataset / "seed42/centered"
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model = LayerSwap("retry_centered", cfg["hidden_size"], cfg["num_layers"], cfg["heads"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        observed, handles = {}, []
        for index, (layer, interaction) in enumerate(zip(model.encoder.layers, model.interactions), 1):
            slot = []
            def attention_hook(module, inputs, result, slot=slot):
                slot.append(result[0].detach())
            def interaction_hook(module, inputs, result, slot=slot, layer=layer, index=index):
                if len(slot) != 4:
                    raise ValueError("unexpected centered MHA call order")
                cross_q, cross_r, self_q, self_r = slot
                q, r, qb, rb, mode = inputs
                if mode != "centered":
                    raise ValueError("wrong interaction mode")
                qm, rm = to_dense_batch(q, qb)[1], to_dense_batch(r, rb)[1]
                for side, u, v in (("query", self_q[qm], cross_q[qm]), ("reference", self_r[rm], cross_r[rm])):
                    store = observed.setdefault((index, side), {})
                    def save(name, tensor):
                        store.setdefault(name, []).append(tensor.float().cpu().numpy())
                    long = layer.long
                    def affine(x, bias=True):
                        grouped = x.reshape(-1, long.groups, long.group_channels)
                        scaled = grouped * long.group_scale[None, :, None]
                        if bias:
                            scaled = scaled + long.group_bias[None, :, None]
                        return scaled.flatten(1)
                    eps = torch.finfo(u.dtype).eps if long.norm.eps is None else long.norm.eps
                    delta = v - u
                    self_power = affine(u).square().mean(-1)
                    save("self_affine_rms", self_power.sqrt())
                    save("delta_scaled_rms", affine(delta, False).square().mean(-1).sqrt())
                    save("old_affine_rms", affine(delta).square().mean(-1).sqrt())
                    save("self_eps_fraction", eps / (self_power + eps))
                    responses = {}
                    for t in (.01, .1, 1.):
                        old = LayerSwap.transform(long, t * delta) - LayerSwap.transform(long, torch.zeros_like(delta))
                        new = contextual_difference(long, u, u + t * delta)
                        for name, value in (("old", old), ("proposed", new)):
                            norm = value.square().mean(-1).sqrt()
                            responses[name, t] = norm
                            save(f"{name}_response_rms_t{t}", norm)
                    for name in ("old", "proposed"):
                        small, large = responses[name, .01], responses[name, .1]
                        valid = (small > 0) & (large > 0)
                        elasticity = torch.full_like(small, float("nan"))
                        elasticity[valid] = torch.log10(large[valid] / small[valid])
                        save(f"{name}_radial_elasticity_001_01", elasticity)
                slot.clear()
            handles.append(interaction.attention.register_forward_hook(attention_hook))
            handles.append(interaction.register_forward_hook(interaction_hook))
        actual = []
        try:
            with torch.no_grad():
                for q, r, yq, yr, selected in batches(vp, graphs, frame, cfg["batch_size"], device):
                    actual.extend((model(q, r) + yr).flatten().cpu().tolist())
        finally:
            for handle in handles:
                handle.remove()
        saved = pd.read_csv(folder / "validation_predictions.csv")
        if saved["query"].tolist() != [p["query"] for p in vp]:
            raise ValueError("validation query identity changed")
        np.testing.assert_allclose(actual, saved.prediction, atol=2e-5, rtol=2e-5)
        replays.append(dict(dataset=dataset, checkpoint_sha256=digest(folder / "best.pt"),
                            count=len(actual), max_abs_difference=float(np.max(np.abs(np.array(actual) - saved.prediction.to_numpy())))))
        expected_counts = {"query": sum(graphs[p["query"]].num_nodes for p in vp),
                           "reference": sum(graphs[p["reference"]].num_nodes for p in vp)}
        for (index, side), metrics in sorted(observed.items()):
            stats = {key: describe(values) for key, values in metrics.items()}
            if any(value["count"] != expected_counts[side] for value in stats.values()):
                raise ValueError("node coverage differs")
            fractions = np.concatenate(metrics["self_eps_fraction"])
            records.append(dict(dataset=dataset, layer=index, side=side, node_occurrences=expected_counts[side],
                                self_eps_dominant_count=int((fractions >= .5).sum()),
                                self_eps_at_least_one_percent_count=int((fractions >= .01).sum()), metrics=stats))
        print("DIAGNOSED", dataset, "all validation queries", len(vp), flush=True)
        del model
    return dict(scope="post-hoc operator probe at old centered states; not a trained proposed model", records=records,
                checkpoint_replays=replays, audit=checked, source_commit=manifest["code_commit"],
                prototype_sha256=digest(ROOT / "tools/check_self_context_scale.py"), device=str(device),
                perturbation_amplitudes=[.01, .1, 1.], additional_fits=0, test_evaluated=False,
                reference_occurrences_may_repeat=True, chemical_causality_claim=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    result = diagnose(args.output)
    Path(args.json_output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["records"]:
        stats = row["metrics"]
        print(row["dataset"], row["layer"], row["side"], "eps_dominant", row["self_eps_dominant_count"],
              "elasticity_medians", stats["old_radial_elasticity_001_01"]["quantiles"]["0.5"],
              stats["proposed_radial_elasticity_001_01"]["quantiles"]["0.5"])
