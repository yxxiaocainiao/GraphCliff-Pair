"""Fixed twenty-epoch acceptance; stream traces, compare independent processes."""
import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
import torch
from experiments.mechanism_retry.model import LayerSwap, SingleBaseline
from experiments.mechanism_retry.repro import stable_max_pool
from experiments.mechanism_retry.run import read_safe
from tools.trace_layer_repro import rng

PLAN = train.ROOT / "experiments/mechanism_retry/repro_long_screen.json"


def hash_value(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def full_rng():
    return dict(**rng(), python=hash_value(random.getstate()), numpy=hash_value(train.np.random.get_state()))


def sources():
    files = sorted((train.ROOT / "graphcliff_pair").rglob("*.py"))
    files += sorted((train.ROOT / "experiments/mechanism_retry").glob("*.py"))
    files += [Path(__file__), train.ROOT / "tools/trace_layer_repro.py", PLAN,
              train.ROOT / "experiments/mechanism_retry/context_scale_screen.json"]
    return {str(p.relative_to(train.ROOT)): train.digest(p) for p in files}


def optimizer_hash(optimizer):
    state = optimizer.state_dict()
    tensors = {f"{index}/{key}": value for index, entry in state["state"].items()
               for key, value in entry.items() if isinstance(value, torch.Tensor)}
    return hash_value((train.state_hash(tensors), state["param_groups"]))


def worker(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(PLAN.read_text())
    cfg = json.loads((train.ROOT / plan["training_config"]).read_text())
    captured_sources = sources()
    summary = dict(dataset=args.dataset, variant=args.variant, epochs=plan["epochs"],
                   source_sha256=captured_sources, status="running", test_evaluated=False)
    started = time.perf_counter()
    try:
        assert os.environ.get("PYTHONHASHSEED") == "42"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.set_num_threads(2)
        device = torch.device("cuda")
        shared.global_max_pool = stable_max_pool
        path = Path(args.csv_root) / f"{args.dataset}.csv"
        frame, graphs, _, _, pairs, valid = read_safe(path, cfg["split_seed"])
        summary.update(input_sha256=train.digest(path), train_count=len(pairs), valid_count=len(valid))
        train.set_seed(SimpleNamespace(seed=42))
        base = SingleBaseline(38, 13, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=0)
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        del base
        train.set_seed(SimpleNamespace(seed=42))
        model = LayerSwap(args.variant, cfg["hidden_size"], cfg["num_layers"], cfg["heads"], "sag")
        model.load_shared(base_state)
        summary["initialization"] = train.state_hash(model.state_dict())
        assert summary["initialization"] == "30d154fb519f3750e5c24dfcd8efea743b75da78573c7e4eeb4d1c96b113a098"
        model.to(device)
        train.set_seed(SimpleNamespace(seed=42))
        torch.use_deterministic_algorithms(True, warn_only=False)
        loss_fn = train.DeltaLoss("mse", alpha_max=cfg["alpha_max"], warmup_epochs=cfg["warmup_epochs"], cap=cfg["weight_cap"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]))
        scheduler = train.WarmupCosineScheduler(optimizer, cfg["warmup_epochs"], cfg["schedule_epochs"], cfg["lr"], cfg["min_lr"])
        generator = torch.Generator().manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        step = 0
        with (output / "trace.jsonl").open("x", encoding="utf-8") as stream:
            def emit(record):
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            for epoch in range(1, plan["epochs"] + 1):
                model.train()
                order = torch.randperm(len(pairs), generator=generator).tolist()
                emit(dict(event="epoch_start", epoch=epoch, order_sha256=hash_value(order),
                          generator_sha256=train.state_hash({"state": generator.get_state()}), rng=full_rng()))
                for q, r, yq, yr, selected in train.batches(pairs, graphs, frame, cfg["batch_size"], device, order):
                    step += 1
                    record = dict(event="batch", epoch=epoch, step=step, pairs_sha256=hash_value(selected), rng_before=full_rng())
                    inputs = {f"{side}_{key}": getattr(graph, key) for side, graph in [("q", q), ("r", r)]
                              for key in ["x", "edge_index", "edge_attr", "batch"]}
                    record["input_sha256"] = train.state_hash(dict(**inputs, yq=yq, yr=yr))
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(q, r)
                    record.update(prediction_sha256=train.state_hash({"prediction": pred}), rng_after_forward=full_rng())
                    loss, _ = loss_fn(pred, yq-yr, epoch)
                    record["loss"] = float(loss.detach())
                    loss.backward()
                    gradients = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
                    record["gradient_sha256"] = train.state_hash(gradients)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                    record["gradient_norm"] = float(norm)
                    optimizer.step()
                    record.update(weight_sha256=train.state_hash(model.state_dict()), rng_after_step=full_rng())
                    emit(record)
                before = full_rng()
                records = train.evaluate(model, dict(variant=args.variant), valid, graphs, frame, cfg["batch_size"], device)
                after = full_rng()
                assert before == after, "validation unexpectedly consumed RNG"
                assert len(records) == len(valid)
                assert all(train.np.isfinite(x["prediction"]) for x in records)
                lr_next = scheduler.step(epoch)
                emit(dict(event="epoch_end", epoch=epoch, steps=step, validation_sha256=hash_value(records),
                          weight_sha256=train.state_hash(model.state_dict()), optimizer_sha256=optimizer_hash(optimizer),
                          rng=after, lr_next=lr_next))
                stream.flush()
                print(f"{args.dataset}/{args.variant} epoch={epoch}/{plan['epochs']} steps={step}", flush=True)
        assert step == plan["epochs"] * ((len(pairs)+cfg["batch_size"]-1)//cfg["batch_size"])
        assert sources() == captured_sources, "source changed during acceptance"
        summary.update(status="completed", optimizer_steps=step, trace_sha256=train.digest(output / "trace.jsonl"),
                       elapsed_seconds=time.perf_counter()-started, peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20)
    except Exception:
        summary.update(status="error", error=traceback.format_exc())
        raise
    finally:
        train.save_json(output / "summary.json", summary)


def compare(first, second):
    a = json.loads((first / "summary.json").read_text())
    b = json.loads((second / "summary.json").read_text())
    for key in ["dataset", "variant", "epochs", "source_sha256", "input_sha256", "train_count", "valid_count", "initialization", "status"]:
        assert a[key] == b[key], f"repeat metadata differs: {key}"
    assert a["status"] == "completed"
    for folder, summary in [(first, a), (second, b)]:
        assert train.digest(folder / "trace.jsonl") == summary["trace_sha256"]
    count, difference = 0, None
    with (first / "trace.jsonl").open(encoding="utf-8") as fa, (second / "trace.jsonl").open(encoding="utf-8") as fb:
        for count, (left, right) in enumerate(zip_longest(fa, fb), 1):
            if left != right:
                u, v = json.loads(left) if left else {}, json.loads(right) if right else {}
                difference = dict(line=count, epoch=u.get("epoch"), step=u.get("step"), event=u.get("event"),
                                  fields=[k for k in set(u) | set(v) if u.get(k) != v.get(k)])
                break
    if difference is None:
        assert count == a["optimizer_steps"] + 2*a["epochs"], "trace coverage is incomplete"
    return dict(dataset=a["dataset"], variant=a["variant"], optimizer_steps_per_repeat=a["optimizer_steps"],
                records_compared=count, first_difference=difference, trace_sha256=[a["trace_sha256"], b["trace_sha256"]])


def suite(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(PLAN.read_text())
    snapshot = sources()
    manifest = dict(plan=plan, source_sha256=snapshot,
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=train.ROOT, text=True).strip(),
                    code_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=train.ROOT, text=True).strip()),
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    pyg=__import__("torch_geometric").__version__, rdkit=__import__("rdkit").__version__, test_evaluated=False)
    train.save_json(output / "manifest.json", manifest)
    env = dict(os.environ, PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    comparisons = []
    for dataset in plan["datasets"]:
        for variant in plan["variants"]:
            folders = [output / dataset / variant / f"repeat{i}" for i in [1, 2]]
            for folder in folders:
                assert sources() == snapshot
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--dataset", dataset,
                                "--variant", variant, "--csv-root", args.csv_root, "--output", str(folder)],
                               env=env, cwd=train.ROOT, check=True)
            result = compare(*folders)
            comparisons.append(result)
            train.save_json(output / "comparisons.json", comparisons)
            print(f"PAIR_DONE {dataset}/{variant}: {result['first_difference']}", flush=True)
            if result["first_difference"] is not None:
                train.save_json(output / "completed.json", dict(status="diverged", comparisons=comparisons))
                return
    train.save_json(output / "completed.json", dict(status="passed", comparisons=comparisons, test_evaluated=False))
    print("ALL_DONE reproducibility acceptance passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=["CHEMBL234_Ki", "CHEMBL244_Ki"])
    parser.add_argument("--variant", choices=["retry_self", "retry_centered"])
    parser.add_argument("--csv-root", default="D:/GraphCliff-main/benchmark_data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.worker and (args.dataset is None or args.variant is None):
        parser.error("worker requires dataset and variant")
    worker(args) if args.worker else suite(args)
