"""Audit the fixed acceptance suite; never compute efficacy metrics."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
from tools.check_layer_repro_long import compare


def main():
    root = train.ROOT / "artifacts/mechanism_retry_repro_long_20261008"
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "completed.json").read_text())
    assert not manifest["code_dirty"] and not manifest["test_evaluated"]
    for path, expected in manifest["source_sha256"].items():
        assert train.digest(train.ROOT / path) == expected, f"source drift: {path}"
    comparisons, summaries, divergence_evidence = [], [], []
    validation_replays, trace_records = 0, 0
    for saved in completed["comparisons"]:
        folders = [root / saved["dataset"] / saved["variant"] / f"repeat{i}" for i in [1, 2]]
        checked = compare(*folders)
        for item in [checked, saved]:
            if item["first_difference"] is not None:
                item["first_difference"]["fields"].sort()
        assert checked == saved
        comparisons.append(checked)
        if checked["first_difference"] is not None:
            line_number = checked["first_difference"]["line"]
            rows = []
            for folder in folders:
                with (folder / "trace.jsonl").open(encoding="utf-8") as stream:
                    previous = None
                    for number, line in enumerate(stream, 1):
                        record = json.loads(line)
                        if number == line_number:
                            rows.append((previous, record))
                            break
                        previous = record
            (previous_a, a), (previous_b, b) = rows
            assert previous_a == previous_b
            assert a["input_sha256"] == b["input_sha256"] and a["rng_before"] == b["rng_before"]
            divergence_evidence.append(dict(dataset=checked["dataset"], variant=checked["variant"],
                                       epoch=a["epoch"], step=a["step"], previous_record_equal=True,
                                       tracked_input_equal=True, rng_before_equal=True,
                                       rng_after_forward_equal=a["rng_after_forward"] == b["rng_after_forward"],
                                       prediction_equal=a["prediction_sha256"] == b["prediction_sha256"],
                                       losses=[a["loss"], b["loss"]],
                                       previous_weight_sha256=previous_a.get("weight_sha256")))
        for folder in folders:
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["source_sha256"] == manifest["source_sha256"]
            assert not summary["test_evaluated"]
            epochs = []
            batch_count = 0
            with (folder / "trace.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    trace_records += 1
                    if record["event"] == "batch":
                        batch_count += 1
                        assert record["step"] == batch_count
                    elif record["event"] == "epoch_end":
                        epochs.append(record["epoch"])
                        validation_replays += summary["valid_count"]
                        assert record["steps"] == batch_count
            assert epochs == list(range(1, 21))
            assert batch_count == summary["optimizer_steps"]
            summaries.append(dict(dataset=summary["dataset"], variant=summary["variant"],
                                  optimizer_steps=summary["optimizer_steps"], epochs=summary["epochs"],
                                  elapsed_seconds=summary["elapsed_seconds"], peak_cuda_mb=summary["peak_cuda_mb"],
                                  summary_sha256=train.digest(folder / "summary.json")))
    if completed["status"] == "passed":
        assert len(comparisons) == 4 and all(row["first_difference"] is None for row in comparisons)
        assert len(summaries) == 8
    else:
        assert completed["status"] == "diverged" and comparisons[-1]["first_difference"] is not None
    result = dict(status=completed["status"], code_commit=manifest["code_commit"],
                  manifest_sha256=train.digest(root / "manifest.json"),
                  source_files_verified=len(manifest["source_sha256"]), comparisons=comparisons, summaries=summaries,
                  divergence_evidence=divergence_evidence,
                  diagnostic_optimizer_steps=sum(row["optimizer_steps"] for row in summaries),
                  validation_predictions_hashed=validation_replays, trace_records=trace_records,
                  fit_elapsed_seconds=sum(row["elapsed_seconds"] for row in summaries),
                  official_test_evaluated=False, performance_metrics_computed=False,
                  planned_runs=8, completed_runs=len(summaries),
                  unexecuted=[dict(dataset=dataset, variant=variant) for dataset in manifest["plan"]["datasets"]
                              for variant in manifest["plan"]["variants"]
                              if not any(row["dataset"] == dataset and row["variant"] == variant for row in comparisons)],
                  limitation="only completed pairs were checked; same machine/software, seed42, native amax, strict mode, first20epochs")
    train.save_json(train.ROOT / "experiments/mechanism_retry/layer_repro_long_results_20261008.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
