"""Verify real-state captures and distinguish observation from a solved cause."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import torch


def value_hashes(mapping):
    return {key: value["sha256"] for key, value in mapping.items()}


def main():
    artifacts = train.ROOT / "artifacts"
    historical = artifacts / "mechanism_retry_repro_long_20261008/CHEMBL234_Ki/retry_self"
    old = [(historical / f"repeat{i}/trace.jsonl").read_text().splitlines() for i in [1, 2]]
    captures, observations, source_checks = [], [], 0
    for name in "abcdef":
        folder = artifacts / f"mechanism_retry_forward_20261008_{name}"
        summary = json.loads((folder / "summary.json").read_text())
        observed = json.loads((folder / "forward485.json").read_text())
        assert summary["status"] == "completed" and summary["epochs"] == 6 and summary["optimizer_steps"] == 498
        assert not summary["test_evaluated"]
        for path, expected in summary["source_sha256"].items():
            assert train.digest(train.ROOT / path) == expected, path
            source_checks += 1
        trace = folder / "trace.jsonl"
        assert train.digest(trace) == summary["trace_sha256"]
        rows = trace.read_text().splitlines()
        assert len(rows) == 510 and rows[:495] == old[0][:495] == old[1][:495]
        target = json.loads(rows[495])
        assert target["step"] == 485 and target["epoch"] == 6
        assert observed["weight_sha256"] == json.loads(old[0][494])["weight_sha256"]
        assert observed["rng_before"] == target["rng_before"]
        assert train.digest(folder / "step485.pt") == observed["capture_sha256"]
        # Trusted local captures contain PyG objects, never an external checkpoint.
        saved = torch.load(folder / "step485.pt", map_location="cpu", weights_only=False)
        assert train.state_hash(saved["model"]) == observed["weight_sha256"]
        for call, side in enumerate(["query", "reference"]):
            graph = saved[side]
            descriptors = {"x": observed["observed"][f"atom_encoder/{call}"]["inputs"]["0"],
                           "edge_index": observed["observed"][f"encoder.layers.0.short/{call}"]["inputs"]["1"],
                           "edge_attr": observed["observed"][f"encoder.layers.0.short/{call}"]["inputs"]["2"],
                           "batch": observed["observed"]["interactions.0/0"]["inputs"][str(call+2)]}
            for key, descriptor in descriptors.items():
                assert train.state_hash({"value": getattr(graph, key)}) == descriptor["sha256"]
        pred = torch.tensor(observed["prediction_values"], dtype=torch.float32).reshape(-1, 1)
        digest = train.state_hash({"prediction": pred})
        assert digest == target["prediction_sha256"]
        matches = [i+1 for i, trace in enumerate(old) if digest == json.loads(trace[495])["prediction_sha256"]]
        captures.append(dict(name=name, epochs=6, optimizer_steps=498, historical_prediction_matches=matches,
                             observed_calls=len(observed["observed"]), capture_sha256=observed["capture_sha256"],
                             forward_report_sha256=train.digest(folder / "forward485.json"),
                             elapsed_seconds=summary["elapsed_seconds"]))
        observations.append(observed)
    baseline = observations[0]
    differences = []
    for i, observed in enumerate(observations[1:], 1):
        assert observed["observed"].keys() == baseline["observed"].keys()
        for key, first in baseline["observed"].items():
            second = observed["observed"][key]
            if value_hashes(first["outputs"]) != value_hashes(second["outputs"]):
                differences.append(dict(capture=captures[i]["name"], call=key,
                                        input_values_equal=value_hashes(first["inputs"]) == value_hashes(second["inputs"]),
                                        rng_before_equal=first["rng_before"] == second["rng_before"]))
    replays = []
    for name in ["replay1", "replay2", "replay3", "deferred_latest1", "deferred_latest2"]:
        path = artifacts / f"mechanism_retry_forward_20261008_{name}/forward485.json"
        value = json.loads(path.read_text())
        assert value["capture_sha256"] == baseline["capture_sha256"]
        assert value["weight_sha256"] == baseline["weight_sha256"]
        changes = [key for key, first in baseline["observed"].items()
                   if value_hashes(first["outputs"]) != value_hashes(value["observed"][key]["outputs"])]
        replays.append(dict(name=name, report_sha256=train.digest(path), changed_output_calls=changes,
                            captured_prediction_equal=value["prediction"]["sha256"] == baseline["prediction"]["sha256"]))
    path = artifacts / "mechanism_retry_forward_20261008_plain_latest/plain.json"
    plain = json.loads(path.read_text())
    assert plain["capture_sha256"] == baseline["capture_sha256"] and len(plain["hashes"]) == 50
    result = dict(captures=captures, differing_output_calls=differences, replays=replays,
                  plain_replay=dict(repetitions=50, unique_predictions=plain["unique_predictions"],
                                    captured_prediction_equal=all(x == baseline["prediction"]["sha256"] for x in plain["hashes"]),
                                    report_sha256=train.digest(path)),
                  diagnostic_optimizer_steps=sum(x["optimizer_steps"] for x in captures), source_hash_checks=source_checks,
                  module_output_divergence_observed=bool(differences), operator_localized=False, historical_divergence_resolved=False,
                  model_architecture_changed=False, official_test_evaluated=False,
                  limit="captures matching one historical branch cannot identify why the other branch occurred")
    train.save_json(train.ROOT / "experiments/mechanism_retry/layer_forward_results_20261008.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
