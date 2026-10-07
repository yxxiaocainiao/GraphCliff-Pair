"""Recheck bounded divergence and fix evidence without training."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train


def read(name):
    path = train.ROOT / "artifacts" / f"mechanism_retry_repro_20261008_{name}" / "trace.json"
    value = json.loads(path.read_text())
    assert value["status"] == "completed" and not value["test_evaluated"]
    assert len(value["records"]) == value["steps_requested"]
    return value, train.digest(path)


def compare(a, b):
    x, hx = read(a)
    y, hy = read(b)
    assert x["initialization"] == y["initialization"]
    assert x["order_sha256"] == y["order_sha256"]
    assert x["source_sha256"] == y["source_sha256"], "paired runs must use identical source snapshots"
    first = None
    for u, v in zip(x["records"], y["records"]):
        changes = [key for key in u if u[key] != v[key]]
        if changes:
            first = dict(step=u["step"], changed_fields=changes)
            break
    return dict(a=a, b=b, trace_sha256=[hx, hy], steps_compared=len(x["records"]), first_difference=first)


def main():
    pairs = [("strict_a", "strict_b"), ("warn_a", "warn_b"), ("exact_a", "exact_b"),
             ("io_a", "io_b"), ("native_a", "native_b"), ("fix_a", "fix_b")]
    comparisons = [compare(a, b) for a, b in pairs]
    assert comparisons[2]["first_difference"]["step"] == 54
    assert set(comparisons[2]["first_difference"]["changed_fields"]) == {
        "gradient_sha256", "gradient_by_parameter", "gradient_norm", "weight_sha256"}
    assert comparisons[-1]["first_difference"] is None and comparisons[-1]["steps_compared"] == 83
    fixed, _ = read("fix_a")
    for path, expected in fixed["source_sha256"].items():
        assert train.digest(train.ROOT / path) == expected, f"final fix source drift: {path}"
    x, _ = read("io_a")
    y, _ = read("io_b")
    u, v = x["records"][53]["module_outputs"], y["records"][53]["module_outputs"]
    assert all(u[key]["forward"] == v[key]["forward"] for key in u)
    assert all(u[key] == v[key] for key in u if key.startswith("head"))
    assert u["sagpool/0/0"]["backward"] != v["sagpool/0/0"]["backward"]
    path = train.ROOT / "artifacts/mechanism_retry_repro_20261008_max_production.json"
    isolated = json.loads(path.read_text())
    for row in isolated["results"]:
        assert row["tied_channels"] > 0 and row["nonzero_upstream_tied_channels"] > 0
        assert row["installed"]["unique_gradient_hashes"] > 1
        assert row["native"]["unique_gradient_hashes"] == 1
        assert row["installed"]["unique_forward_hashes"] == row["native"]["unique_forward_hashes"] == 1
    inventory = []
    for path in sorted((train.ROOT / "artifacts").glob("mechanism_retry_repro_20261008_*/trace.json")):
        run = json.loads(path.read_text())
        assert run["status"] == "completed"
        inventory.append(dict(path=str(path.relative_to(train.ROOT)), sha256=train.digest(path), steps=len(run["records"])))
    result = dict(comparisons=comparisons, isolated_max=isolated["results"],
                  isolated_report_sha256=train.digest(train.ROOT / "artifacts/mechanism_retry_repro_20261008_max_production.json"),
                  inventory=inventory, diagnostic_optimizer_steps=sum(row["steps"] for row in inventory),
                  model_architecture_changed=False, official_test_evaluated=False,
                  scope="CHEMBL234 centered, seed42, first epoch only; no complete training reproducibility guarantee",
                  superseded="initial isolated reports used default matmul precision, failed to replay real forward, and cannot exclude max pooling")
    output = train.ROOT / "experiments/mechanism_retry/layer_repro_results_20261008.json"
    train.save_json(output, result)
    print(json.dumps(dict(comparisons=comparisons, isolated_max=result["isolated_max"],
                          diagnostic_optimizer_steps=result["diagnostic_optimizer_steps"]), indent=2))


if __name__ == "__main__":
    main()
