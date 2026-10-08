"""Frozen full/self confirmation; reuse the historical trainer unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
from experiments.mechanism_retry import run as retry
from experiments.mechanism_retry.model import LayerSwap, SingleBaseline
from experiments.mechanism_retry.repro import stable_max_pool

CONFIG = train.ROOT / "experiments/mechanism_retry/full_self_screen.json"
PROTOCOL = train.ROOT / "experiments/mechanism_retry/full_self_execution_20261008.md"
PREFLIGHT = train.ROOT / "experiments/mechanism_retry/full_self_preflight_20261008.json"


def sources():
    paths = [CONFIG, PROTOCOL, Path(__file__), Path(__file__).with_name("report_full_self_confirmation.py"),
             train.ROOT / "docs/sources.json", train.ROOT / "external/fppool/pooling.py",
             *sorted((train.ROOT / "experiments/mechanism_retry").glob("*.py")),
             *sorted((train.ROOT / "experiments/long_branch_swap").glob("*.py")),
             *sorted((train.ROOT / "graphcliff_pair").rglob("*.py"))]
    return {str(p.relative_to(train.ROOT)): train.digest(p) for p in paths}


def preflight():
    torch = train.torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    cfg = json.loads(CONFIG.read_text())
    history = json.loads((train.ROOT / "experiments/mechanism_retry/layer_preflight_20261007.json").read_text())
    old_pool = shared.global_max_pool
    shared.global_max_pool = stable_max_pool
    records, inputs = [], []
    try:
        for source in history["datasets"]:
            assert train.digest(source["input_path"]) == source["input_sha256"]
            frame, graphs, tr, va, tp, vp = retry.read_safe(source["input_path"], cfg["split_seed"])
            assert len(tp) == source["train_count"] and len(vp) == source["valid_count"]
            inputs.append(dict(**source, pair_sha256=train.hashlib.sha256(json.dumps([tr.tolist(), va.tolist(), tp, vp], sort_keys=True).encode()).hexdigest()))
            for seed in cfg["seeds"]:
                train.set_seed(SimpleNamespace(seed=seed))
                base = SingleBaseline(38, 13, hidden_size=256, num_layers=3, dropout=0)
                state = {k: v.clone() for k, v in base.state_dict().items()}
                del base
                first = None
                for arm in cfg["arms"]:
                    train.set_seed(SimpleNamespace(seed=seed))
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    model = LayerSwap(arm["variant"])
                    selected = model.load_shared(state)
                    initial = model.state_dict()
                    assert all(torch.equal(initial[k], v) for k, v in selected.items())
                    if first is not None:
                        common = set(first) & set(initial)
                        assert all(torch.equal(first[k], initial[k]) for k in common)
                    else:
                        first = {k: v.clone() for k, v in initial.items()}
                    full_hash = train.state_hash(initial)
                    head_hash = train.state_hash(model.head.state_dict())
                    params = train.parameters(model)
                    assert params == {"full": 6021198, "self": 6810654}[arm["name"]]
                    model.cuda().train()
                    train.set_seed(SimpleNamespace(seed=seed))
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    q, r, yq, yr, _ = next(train.batches(tp, graphs, frame, 32, torch.device("cuda")))
                    prediction = model(q, r)
                    loss = torch.mean((prediction-(yq-yr))**2)
                    loss.backward()
                    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    records.append(dict(dataset=source["dataset"], seed=seed, arm=arm["name"], parameters=params,
                                        full_sha256=full_hash, head_sha256=head_hash, loss=float(loss.detach())))
                    print("PREFLIGHT_OK", source["dataset"], seed, arm["name"], flush=True)
                    del model, prediction, loss, q, r, yq, yr
                    torch.cuda.empty_cache()
    finally:
        shared.global_max_pool = old_pool
    train.save_json(PREFLIGHT, dict(source_sha256=sources(), inputs=inputs, records=records,
                                   optimizer_steps=0, test_evaluated=False))


def worker(output):
    assert os.environ.get("PYTHONHASHSEED") == "42" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    pool, save, seed = shared.global_max_pool, train.save_json, train.set_seed
    def strict_seed(args):
        seed(args)
        train.torch.use_deterministic_algorithms(True, warn_only=False)
    def record(path, value):
        if Path(path).name == "manifest.json":
            value.update(pooling_backward="native amax equal-tie gradients", deterministic="strict available algorithms; no GPU bitwise guarantee",
                         reproducibility="three training seeds on one reused split; descriptive, not significance",
                         entrypoint_sha256=train.digest(__file__), protocol_sha256=train.digest(PROTOCOL), preflight_sha256=train.digest(PREFLIGHT))
        save(path, value)
    shared.global_max_pool, train.save_json, train.set_seed = stable_max_pool, record, strict_seed
    try:
        retry.run(CONFIG, "D:/GraphCliff-main/benchmark_data", output)
    finally:
        shared.global_max_pool, train.save_json, train.set_seed = pool, save, seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.worker:
        worker(args.output / "runs")
    else:
        assert args.output is not None
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=train.ROOT), "commit before training"
        check = json.loads(PREFLIGHT.read_text())
        assert check["optimizer_steps"] == 0 and not check["test_evaluated"]
        assert sources() == check["source_sha256"], "preflight sources changed"
        assert all(train.digest(x["input_path"]) == x["input_sha256"] for x in check["inputs"])
        args.output.mkdir(parents=True, exist_ok=False)
        train.save_json(args.output / "suite.json", dict(source_sha256=sources(), preflight_sha256=train.digest(PREFLIGHT),
                        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=train.ROOT, text=True).strip(), planned_fits=12, test_evaluated=False))
        subprocess.run([sys.executable, __file__, "--worker", "--output", str(args.output)], cwd=train.ROOT,
                       env=dict(os.environ, PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8"), check=True)
        assert sources() == check["source_sha256"]
        train.save_json(args.output / "completed.json", dict(completed_fits=12, test_evaluated=False))
