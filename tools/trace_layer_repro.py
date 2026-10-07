"""Bounded real-batch training trace; no validation selection or test access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import torch
from experiments.mechanism_retry.model import LayerSwap, SingleBaseline
from experiments.mechanism_retry.run import read_safe


def tensor_hash(value):
    return train.state_hash({"tensor": value})


def rng():
    return dict(cpu=tensor_hash(torch.get_rng_state()),
                cuda=[tensor_hash(x) for x in torch.cuda.get_rng_state_all()])


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((train.ROOT / "experiments/mechanism_retry/context_scale_screen.json").read_text())
    report = dict(strict=args.strict, steps_requested=args.steps, records=[], test_evaluated=False,
                  native_max=args.native_max,
                  device=args.device, torch=torch.__version__,
                  pythonhashseed_at_start=os.environ.get("PYTHONHASHSEED"),
                  cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  source_sha256={str(p.relative_to(train.ROOT)): train.digest(p) for p in
                                 [Path(__file__), train.ROOT / "experiments/mechanism_retry/model.py",
                                  train.ROOT / "graphcliff_pair/train.py",
                                  train.ROOT / "experiments/mechanism_retry/repro.py",
                                  train.ROOT / "tools/check_layer_max_backward.py"]})
    stage = "setup"
    try:
        torch.set_num_threads(2)
        if args.native_max:
            import graphcliff_pair.model as shared
            from experiments.mechanism_retry.repro import stable_max_pool as native_max
            shared.global_max_pool = native_max
        frame, graphs, _, _, pairs, _ = read_safe(Path(args.csv_root) / "CHEMBL234_Ki.csv", cfg["split_seed"])
        train.set_seed(SimpleNamespace(seed=42))
        base = SingleBaseline(38, 13, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=0)
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        del base
        train.set_seed(SimpleNamespace(seed=42))
        model = LayerSwap("retry_centered", cfg["hidden_size"], cfg["num_layers"], cfg["heads"], "sag")
        model.load_shared(base_state)
        report["initialization"] = train.state_hash(model.state_dict())
        assert report["initialization"] == "30d154fb519f3750e5c24dfcd8efea743b75da78573c7e4eeb4d1c96b113a098"
        device = torch.device(args.device)
        model.to(device).train()
        train.set_seed(SimpleNamespace(seed=42))
        torch.use_deterministic_algorithms(True, warn_only=not args.strict)
        # Reuse the exact training loss path, including its elementwise reduction.
        loss_fn = train.DeltaLoss("mse", alpha_max=cfg["alpha_max"], warmup_epochs=cfg["warmup_epochs"], cap=cfg["weight_cap"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]))
        order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(42)).tolist()
        report["order_sha256"] = hashlib.sha256(json.dumps(order).encode()).hexdigest()
        for step, (q, r, yq, yr, selected) in enumerate(train.batches(pairs, graphs, frame, cfg["batch_size"], device, order), 1):
            if step > args.steps:
                break
            record = dict(step=step, pairs=selected, rng_before=rng(),
                          input_sha256=train.state_hash({**{"q_"+k: getattr(q, k) for k in ("x", "edge_index", "edge_attr", "batch")},
                                                       **{"r_"+k: getattr(r, k) for k in ("x", "edge_index", "edge_attr", "batch")},
                                                       "yq": yq, "yr": yr}))
            report["records"].append(record)
            optimizer.zero_grad(set_to_none=True)
            if step == args.capture_step:
                torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                query=q.clone().cpu(), reference=r.clone().cpu(), target=(yq-yr).cpu(),
                                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()),
                           output / "before_step.pt")
            stage = f"step {step} forward"
            handles = []
            if step == args.hook_step:
                from check_layer_max_backward import observe_modules
                record["module_outputs"], handles = observe_modules(model)
            pred = model(q, r)
            record.update(prediction_sha256=tensor_hash(pred), rng_after_forward=rng())
            loss, _ = loss_fn(pred, yq-yr, 1)
            record["loss"] = float(loss.detach())
            stage = f"step {step} backward"
            loss.backward()
            for handle in handles:
                handle.remove()
            gradients = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
            assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
            record["gradient_sha256"] = train.state_hash(gradients)
            record["gradient_by_parameter"] = {k: tensor_hash(v) for k, v in gradients.items()}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            record["gradient_norm"] = float(norm)
            stage = f"step {step} optimizer"
            optimizer.step()
            record.update(weight_sha256=train.state_hash(model.state_dict()), rng_after_step=rng())
            train.save_json(output / "trace.json", report)
            print(f"step={step} loss={record['loss']:.9f}", flush=True)
        assert len(report["records"]) == args.steps
        report["status"] = "completed"
    except Exception:
        report.update(status="error", failed_stage=stage, error=traceback.format_exc())
        raise
    finally:
        train.save_json(output / "trace.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", default="D:/GraphCliff-main/benchmark_data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--capture-step", type=int, default=0)
    parser.add_argument("--hook-step", type=int, default=0)
    parser.add_argument("--native-max", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 83:
        parser.error("bounded diagnosis requires 1..83 steps (at most the first epoch)")
    run(args)
