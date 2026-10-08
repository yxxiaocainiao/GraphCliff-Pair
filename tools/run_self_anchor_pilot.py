"""Two independent repetitions of one fixed eight-fit mechanism pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
from experiments.mechanism_retry import run as retry
from experiments.mechanism_retry.self_anchor import SelfAnchorSwap
from experiments.mechanism_retry.repro import stable_max_pool

CONFIG = train.ROOT / "experiments/mechanism_retry/self_anchor_screen.json"
PROTOCOL = train.ROOT / "experiments/mechanism_retry/self_anchor_execution_20261008.md"
PREFLIGHT = train.ROOT / "experiments/mechanism_retry/self_anchor_preflight_20261008.json"


def worker(output):
    assert os.environ.get("PYTHONHASHSEED") == "42" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    constructor, pool, save, seed = retry.LayerSwap, shared.global_max_pool, train.save_json, train.set_seed
    def factory(variant, *args, **kwargs):
        if variant in ("retry_anchor", "retry_gated_delta"):
            return SelfAnchorSwap(*args, **kwargs, anchor=variant == "retry_anchor")
        return constructor(variant, *args, **kwargs)
    def strict_seed(args):
        seed(args)
        train.torch.use_deterministic_algorithms(True, warn_only=False)
    def record(path, value):
        if Path(path).name == "manifest.json":
            value.update(pooling_backward="native amax, equal gradient at exact ties",
                         deterministic="strict algorithms enabled; no bitwise full-training guarantee",
                         reproducibility="two independent same-seed process repetitions; descriptive range separation, not significance",
                         entrypoint_sha256=train.digest(__file__), protocol_sha256=train.digest(PROTOCOL),
                         preflight_sha256=train.digest(PREFLIGHT))
        save(path, value)
    retry.LayerSwap, shared.global_max_pool, train.save_json, train.set_seed = factory, stable_max_pool, record, strict_seed
    try:
        retry.run(CONFIG, "D:/GraphCliff-main/benchmark_data", output)
    finally:
        retry.LayerSwap, shared.global_max_pool, train.save_json, train.set_seed = constructor, pool, save, seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
    else:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=train.ROOT), "commit before training"
        preflight = json.loads(PREFLIGHT.read_text())
        assert preflight["optimizer_steps"] == 0 and not preflight["test_evaluated"]
        assert all(train.digest(train.ROOT / p) == digest for p, digest in preflight["source_sha256"].items()), "preflight sources changed"
        expected = {str(p.relative_to(train.ROOT)): train.digest(p) for p in
                    [CONFIG, PROTOCOL, PREFLIGHT, Path(__file__), Path(__file__).with_name("report_self_anchor_pilot.py"),
                     *sorted((train.ROOT / "experiments/mechanism_retry").glob("*.py")),
                     *sorted((train.ROOT / "graphcliff_pair").rglob("*.py"))]}
        train.save_json(output / "suite.json", dict(source_sha256=expected,
                        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=train.ROOT, text=True).strip(),
                        planned_fits=8, official_test_evaluated=False))
        env = dict(os.environ, PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        for repeat in (1, 2):
            assert all(train.digest(train.ROOT / p) == digest for p, digest in expected.items())
            subprocess.run([sys.executable, __file__, "--worker", "--output", str(output / f"repeat{repeat}")],
                           cwd=train.ROOT, env=env, check=True)
        assert all(train.digest(train.ROOT / p) == digest for p, digest in expected.items())
        train.save_json(output / "completed.json", dict(completed_fits=8, test_evaluated=False))
