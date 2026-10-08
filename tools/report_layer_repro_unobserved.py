"""Audit two original-worker rechecks against both historical trajectories."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
from tools.check_layer_repro_long import compare


def main():
    artifacts = train.ROOT / "artifacts"
    current = artifacts / "mechanism_retry_repro_unobserved_20261008"
    old = artifacts / "mechanism_retry_repro_long_20261008/CHEMBL234_Ki/retry_self"
    manifest_path = old.parents[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runtime = dict(torch=train.torch.__version__, cuda=train.torch.version.cuda,
                   gpu=train.torch.cuda.get_device_name(), pyg=__import__("torch_geometric").__version__,
                   rdkit=__import__("rdkit").__version__)
    assert all(value == manifest[key] for key, value in runtime.items())
    folders = {f"new{i}": current / f"repeat{i}" for i in (1, 2)}
    folders.update({f"old{i}": old / f"repeat{i}" for i in (1, 2)})
    summaries, checks = {}, 0
    for name, folder in folders.items():
        value = json.loads((folder / "summary.json").read_text())
        assert value["status"] == "completed" and value["epochs"] == 20 and value["optimizer_steps"] == 1660
        assert not value["test_evaluated"]
        assert train.digest(folder / "trace.jsonl") == value["trace_sha256"]
        for path, digest in value["source_sha256"].items():
            assert train.digest(train.ROOT / path) == digest, path
            checks += 1
        rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == 1700
        assert [row["step"] for row in rows if row["event"] == "batch"] == list(range(1, 1661))
        for event in ("epoch_start", "epoch_end"):
            assert [row["epoch"] for row in rows if row["event"] == event] == list(range(1, 21))
        assert all(row["steps"] == row["epoch"] * 83 for row in rows if row["event"] == "epoch_end")
        target = rows[495]
        assert target["event"] == "batch" and target["step"] == 485 and target["epoch"] == 6
        summaries[name] = dict(summary_sha256=train.digest(folder / "summary.json"),
                               trace_sha256=value["trace_sha256"], source_files=len(value["source_sha256"]),
                               elapsed_seconds=value["elapsed_seconds"], peak_cuda_mb=value["peak_cuda_mb"],
                               input_sha256=value["input_sha256"], initialization=value["initialization"],
                               target_prediction_sha256=target["prediction_sha256"], target_loss=target["loss"],
                               valid_count=value["valid_count"])
    comparisons = []
    for a, b in (("new1", "new2"), ("old1", "old2"), ("new1", "old1"),
                 ("new1", "old2"), ("new2", "old1"), ("new2", "old2")):
        result = compare(folders[a], folders[b])
        if result["first_difference"]:
            result["first_difference"]["fields"].sort()
        comparisons.append(dict(first=a, second=b, **result))
    result = dict(summaries=summaries, comparisons=comparisons,
                  historical_manifest_sha256=train.digest(manifest_path), audit_runtime=runtime,
                  current_pair_equal=comparisons[0]["first_difference"] is None,
                  historical_pair_equal=comparisons[1]["first_difference"] is None,
                  target_historical_matches={name: [old_name for old_name in ("old1", "old2")
                                              if summaries[name]["target_prediction_sha256"] == summaries[old_name]["target_prediction_sha256"]]
                                             for name in ("new1", "new2")},
                  source_hash_checks=checks, diagnostic_optimizer_steps=3320, current_trace_events=3400,
                  validation_predictions_hashed=20 * sum(summaries[name]["valid_count"] for name in ("new1", "new2")),
                  current_elapsed_seconds=sum(summaries[name]["elapsed_seconds"] for name in ("new1", "new2")),
                  official_test_evaluated=False, performance_metrics_computed=False,
                  extra_module_hooks=False, historical_divergence_resolved=False,
                  limit="two repeats on one task/operator/seed cannot erase the historical divergence or certify all training")
    path = train.ROOT / "experiments/mechanism_retry/layer_repro_unobserved_results_20261008.json"
    train.save_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
