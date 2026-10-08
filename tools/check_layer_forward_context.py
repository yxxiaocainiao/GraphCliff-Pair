"""Bounded, hook-free replay under three execution histories; no optimizer."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from graphcliff_pair import train
from tools import trace_layer_forward as base

CAPTURE = train.ROOT / "artifacts/mechanism_retry_forward_20261008_a/step485.pt"
CONTEXTS = ("fresh", "allocator_churn", "warm_backward")


def restore(saved):
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    train.np.random.set_state(saved["numpy_rng"])


def worker(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    assert os.environ["PYTHONHASHSEED"] == "42"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    source = dict(base.previous.sources())
    for file in (Path(__file__), Path(base.__file__)):
        source[str(file.relative_to(train.ROOT))] = train.digest(file)
    saved = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    assert train.digest(CAPTURE) == "f47c54b26281e471e14680a869c595d850ba6aa4b5fe8184b04be67aea6ede28"
    torch.set_num_threads(2)
    train.set_seed(SimpleNamespace(seed=42))
    torch.use_deterministic_algorithms(True, warn_only=False)
    retained = []
    if args.context == "allocator_churn":
        blocks = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in (1, 257, 4097, 65537, 1048577, 8388609)]
        retained = blocks[1::2]
        del blocks
    base.shared.global_max_pool = base.stable_max_pool
    model = base.previous.LayerSwap("retry_self").cuda().train()
    model.load_state_dict(saved["model"])
    q, r = saved["query"].cuda(), saved["reference"].cuda()
    inputs = {f"{side}_{key}": getattr(graph, key) for side, graph in (("q", q), ("r", r))
              for key in ("x", "edge_index", "edge_attr", "batch")}
    identity = train.state_hash(inputs)
    weights = train.state_hash(saved["model"])
    warmups = 2 if args.context == "warm_backward" else 0
    for _ in range(warmups):
        model.zero_grad(set_to_none=True)
        result = model(q, r)
        result.square().mean().backward()
        del result
    model.zero_grad(set_to_none=True)
    assert train.state_hash(model.state_dict()) == weights, "warmup mutated model state"
    assert train.state_hash(inputs) == identity
    assert all(m.training and not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    hashes, rng_after, values = [], [], {}
    expected = json.loads((CAPTURE.parent / "forward485.json").read_text())["rng_before"]
    for _ in range(20):
        restore(saved)
        assert base.previous.full_rng() == expected
        result = model(q, r)
        digest = train.state_hash({"prediction": result})
        hashes.append(digest)
        rng_after.append(base.previous.full_rng())
        values[digest] = result.detach().cpu().flatten().tolist()
        del result
    assert train.state_hash(model.state_dict()) == weights and train.state_hash(inputs) == identity
    assert all(train.digest(train.ROOT / path) == digest for path, digest in source.items())
    train.save_json(output / "result.json", dict(context=args.context, capture_sha256=train.digest(CAPTURE),
                    source_sha256=source, weight_sha256=weights, input_sha256=identity, rng_before=expected,
                    hashes=hashes, values=values, rng_after=rng_after, unique_predictions=len(values),
                    warmup_forward_backward=warmups, measured_forwards=20, optimizer_steps=0,
                    address_mod4096={k: v.data_ptr() % 4096 for k, v in inputs.items()},
                    first_parameter_mod4096=next(model.parameters()).data_ptr() % 4096,
                    allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                    retained_bytes=sum(x.numel() for x in retained), torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    matmul_precision=torch.get_float32_matmul_precision(),
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32, deterministic=torch.are_deterministic_algorithms_enabled(),
                    module_hooks=0, model_state_unchanged=True, test_evaluated=False))
    print(args.context, "unique predictions:", len(values), flush=True)


def report(output):
    historical = train.ROOT / "artifacts/mechanism_retry_repro_long_20261008/CHEMBL234_Ki/retry_self"
    old = [json.loads((historical / f"repeat{i}/trace.jsonl").read_text().splitlines()[495]) for i in (1, 2)]
    records, checks, prediction_hashes = [], 0, set()
    for context in CONTEXTS:
        for repeat in (1, 2):
            path = output / context / f"repeat{repeat}/result.json"
            value = json.loads(path.read_text())
            assert value["context"] == context and len(value["hashes"]) == len(value["rng_after"]) == 20
            assert value["unique_predictions"] == len(set(value["hashes"])) == len(value["values"])
            assert value["capture_sha256"] == train.digest(CAPTURE)
            assert value["model_state_unchanged"] and value["deterministic"] and value["matmul_precision"] == "high"
            prediction_hashes.update(value["hashes"])
            assert value["optimizer_steps"] == value["module_hooks"] == 0 and not value["test_evaluated"]
            assert value["weight_sha256"] == "cc9c9bb3a426abe3c7e3fde8a52bfefe1e9a22c0162845d7f025f1edbec6c5a5"
            assert value["rng_before"] == old[0]["rng_before"] == old[1]["rng_before"]
            for path_name, digest in value["source_sha256"].items():
                assert train.digest(train.ROOT / path_name) == digest
                checks += 1
            records.append(dict(context=context, repeat=repeat, result_sha256=train.digest(path),
                                unique_predictions=value["unique_predictions"],
                                historical_matches=sorted({i+1 for digest in value["hashes"] for i, row in enumerate(old)
                                                           if digest == row["prediction_sha256"]}),
                                rng_after_matches=all(x == old[1]["rng_after_forward"] for x in value["rng_after"]),
                                measured_forwards=20, warmup_forward_backward=value["warmup_forward_backward"],
                                input_sha256=value["input_sha256"], address_mod4096=value["address_mod4096"],
                                first_parameter_mod4096=value["first_parameter_mod4096"],
                                allocated_bytes=value["allocated_bytes"], retained_bytes=value["retained_bytes"]))
    assert len({x["input_sha256"] for x in records}) == 1
    result = dict(records=records, measured_forwards=120, warmup_forward_backward=4, optimizer_steps=0,
                  unique_predictions_across_contexts=len(prediction_hashes),
                  source_hash_checks=checks, historical_first_branch_reproduced=any(1 in x["historical_matches"] for x in records),
                  official_test_evaluated=False, limit="three sampled contexts do not reconstruct the full historical training execution")
    train.save_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", choices=CONTEXTS)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(json.dumps(report(Path(args.output)), indent=2))
    elif args.context:
        worker(args)
    else:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        for context in CONTEXTS:
            for repeat in (1, 2):
                subprocess.run([sys.executable, __file__, "--context", context, "--output",
                                str(output / context / f"repeat{repeat}")], cwd=train.ROOT, env=env, check=True)
        print(json.dumps(report(output), indent=2))
