"""独立训练入口：原编码器/调度器与库内优化器，新增配对和记录胶水。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from .data import digest, read_development
from .fingerprint import membership
from .loss import DeltaLoss
from .model import PairRegressor, parameters
from .vendor.model import GraphCliffRegressor
from .vendor.training import set_seed, WarmupCosineScheduler

ROOT = Path(__file__).resolve().parents[1]

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def predict(model, variant, queries, references):
    if variant == "direct":
        return model(queries.x, queries.edge_index, queries.edge_attr, queries.batch)
    return model(queries, references)

def batches(pairs, graphs, frame, size, device, order=None):
    indices = list(range(len(pairs))) if order is None else order
    for start in range(0, len(indices), size):
        selected = [pairs[int(i)] for i in indices[start:start+size]]
        qrows = [p["query"] for p in selected]
        rrows = [p["reference"] for p in selected]
        query = Batch.from_data_list([graphs[i] for i in qrows]).to(device)
        reference = Batch.from_data_list([graphs[i] for i in rrows]).to(device)
        yq = torch.tensor(frame.loc[qrows, "y"].to_numpy(), dtype=torch.float32, device=device).reshape(-1, 1)
        yr = torch.tensor(frame.loc[rrows, "y"].to_numpy(), dtype=torch.float32, device=device).reshape(-1, 1)
        yield query, reference, yq, yr, selected

@torch.no_grad()
def evaluate(model, arm, pairs, graphs, frame, size, device):
    model.eval()
    records = []
    for q, r, yq, yr, selected in batches(pairs, graphs, frame, size, device):
        output = predict(model, arm["variant"], q, r)
        restored = output if arm["variant"] == "direct" else output + yr
        for pair, truth, reference_y, pred in zip(selected, yq.flatten().tolist(), yr.flatten().tolist(), restored.flatten().tolist()):
            records.append(dict(**pair, y=truth, reference_y=reference_y, prediction=pred,
                                cliff_mol=int(frame.at[pair["query"], "cliff_mol"])))
    return records

def metrics(records):
    frame = pd.DataFrame(records)
    error = frame.prediction.to_numpy() - frame.y.to_numpy()
    if not np.isfinite(error).all() or not len(error):
        raise ValueError("预测非有限或为空")
    cliff = frame.cliff_mol.to_numpy().astype(bool)
    def rmse(mask):
        return float(np.sqrt(np.mean(error[mask]**2))) if np.any(mask) else None
    return dict(overall_rmse=rmse(np.ones(len(error), dtype=bool)), cliff_rmse=rmse(cliff),
                noncliff_rmse=rmse(~cliff), mae=float(np.mean(np.abs(error))),
                prediction_std=float(frame.prediction.std(ddof=0)), count=len(error), cliff_count=int(cliff.sum()))

def train_arm(config, arm, seed, frame, graphs, train_pairs, valid_pairs, base_state, output, device, scale):
    set_seed(SimpleNamespace(seed=seed))
    if arm["variant"] == "direct":
        if arm["readout"] != "sag" or arm["loss"] != "mse":
            raise ValueError("direct 首轮必须使用原读出和 MSE")
        model = GraphCliffRegressor(38, 13, hidden_size=config["hidden_size"], num_layers=config["num_layers"], dropout=0)
        model.load_state_dict(base_state)
        shared = {k: v for k, v in base_state.items() if k.startswith(("atom_encoder.", "encoder.", "sagpool."))}
    else:
        model = PairRegressor(arm["variant"], config["hidden_size"], config["num_layers"], config["heads"], arm["readout"])
        shared = model.load_shared(base_state)
    actual_state = model.state_dict()
    for key, value in shared.items():
        if not torch.equal(actual_state[key], value):
            raise RuntimeError("共享初始化不一致")
    # 同结构组的 head 初始化也一致：每个 arm 重置 seed，再构建相同模块顺序。
    encoder_state = {k: v for k, v in shared.items() if k.startswith(("atom_encoder.", "encoder."))}
    initialization = dict(encoder_sha256=state_hash(encoder_state), shared_sha256=state_hash(shared),
                          full_sha256=state_hash(actual_state))
    if arm["variant"] != "direct":
        initialization["head_sha256"] = state_hash(model.head.state_dict())
    model = model.to(device)
    # 额外模块构造会消耗随机数；训练前重置，使编码器 dropout 起点一致。
    set_seed(SimpleNamespace(seed=seed))
    loss_fn = DeltaLoss(arm["loss"], scale=scale, alpha_max=config["alpha_max"],
                        warmup_epochs=config["warmup_epochs"], cap=config["weight_cap"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], betas=tuple(config["betas"]))
    scheduler = WarmupCosineScheduler(optimizer, config["warmup_epochs"], config["schedule_epochs"], config["lr"], config["min_lr"])
    generator = torch.Generator().manual_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "initialization.json", initialization)
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    graph_forwards, best_epoch = 0, None
    for epoch in range(1, config["epochs"]+1):
        model.train()
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        total, n, weights_min, weights_max = 0., 0, float("inf"), 0.
        for q, r, yq, yr, _ in batches(train_pairs, graphs, frame, config["batch_size"], device, order):
            optimizer.zero_grad(set_to_none=True)
            target = yq if arm["variant"] == "direct" else yq-yr
            pred = predict(model, arm["variant"], q, r)
            loss, weights = loss_fn(pred, target, epoch)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise RuntimeError("梯度非有限")
            optimizer.step()
            total += float(loss.detach()) * len(target)
            n += len(target)
            weights_min = min(weights_min, float(weights.min()))
            weights_max = max(weights_max, float(weights.max()))
            graph_forwards += len(target)*(1 if arm["variant"] == "direct" else 2)
        records = evaluate(model, arm, valid_pairs, graphs, frame, config["batch_size"], device)
        result = metrics(records)
        mse = result["overall_rmse"]**2
        # checkpoint 取真正的最低验证 Overall MSE，不采用 cliff 或加权验证目标。
        if mse < best:
            best, stale, best_epoch = mse, 0, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, output / "best.pt")
        else:
            stale += 1
        lr = scheduler.step(epoch)
        history.append(dict(epoch=epoch, train_loss=total/n, valid_mse=mse, lr_next=lr,
                            weight_min=weights_min, weight_max=weights_max, best_epoch=best_epoch))
        save_json(output / "history.json", history)
        print(f"{output.parent.name}/{arm['name']} epoch={epoch} valid_rmse={result['overall_rmse']:.4f} best={best_epoch}", flush=True)
        if stale >= config["patience"]:
            break
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model_state_dict"])
    restored_records = evaluate(model, arm, valid_pairs, graphs, frame, config["batch_size"], device)
    result = metrics(restored_records)
    if abs(result["overall_rmse"]**2-best) > 1e-5:
        raise RuntimeError("最佳模型重载未恢复验证预测")
    pd.DataFrame(restored_records).to_csv(output / "validation_predictions.csv", index=False)
    summary = dict(**result, arm=arm["name"], seed=seed, best_epoch=best_epoch, epochs_run=epoch,
                   parameters=parameters(model), train_graph_forwards=graph_forwards, elapsed_seconds=time.perf_counter()-started,
                   peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None,
                   train_delta_scale=scale, initialization=initialization, test_evaluated=False)
    save_json(output / "summary.json", summary)
    return summary

def run(config_path, csv_root, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["schedule_epochs"] <= config["warmup_epochs"]:
        raise ValueError("训练预算非法")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_files = sorted((ROOT / "graphcliff_pair").rglob("*.py"))
    manifest = dict(config=config, config_sha256=digest(config_path), device=str(device),
                    torch=torch.__version__, pyg=__import__("torch_geometric").__version__,
                    rdkit=__import__("rdkit").__version__,
                    source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_files},
                    source_manifest_sha256=digest(ROOT / "docs/sources.json"),
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    code_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                    test_evaluated=False, deterministic="upstream best-effort warn_only; not bitwise GPU guarantee")
    save_json(output / "manifest.json", manifest)
    summaries = []
    for dataset in config["datasets"]:
        path = Path(csv_root) / f"{dataset}.csv"
        frame, graphs, tr, va, train_all, valid_all = read_development(path, config["split_seed"])
        tp = train_all[:config.get("limit_train_queries", len(train_all))]
        vp = valid_all[:config.get("limit_valid_queries", len(valid_all))]
        if not tp or not vp:
            raise ValueError("smoke 查询数量不能为零")
        all_deltas = np.array([frame.at[p["query"], "y"]-frame.at[p["reference"], "y"] for p in train_all])
        scale = max(float(np.median(np.abs(all_deltas))), 1e-6)
        if any(arm["readout"] == "fppool" for arm in config["arms"]):
            needed = {p[k] for p in tp+vp for k in ["query", "reference"]}
            for i in needed:
                g = graphs[i]
                g.atom_fp = membership(SimpleNamespace(smiles=frame.at[i,"smiles"],x=g.x,edge_index=g.edge_index,edge_attr=g.edge_attr))
        dataset_out = output / dataset
        dataset_out.mkdir()
        save_json(dataset_out / "pairs.json", dict(input_sha256=digest(path), train_rows=tr.tolist(), valid_rows=va.tolist(),
                                                  train_pairs=tp, valid_pairs=vp, full_train_pair_count=len(train_all), scale=scale))
        # 参考标签基线没有模型，也没有查询标签参与预测。
        nearest = [dict(**p, y=float(frame.at[p["query"],"y"]), reference_y=float(frame.at[p["reference"],"y"]),
                        prediction=float(frame.at[p["reference"],"y"]), cliff_mol=int(frame.at[p["query"],"cliff_mol"])) for p in vp]
        save_json(dataset_out / "nearest_reference.json", metrics(nearest))
        for seed in config["seeds"]:
            set_seed(SimpleNamespace(seed=seed))
            base = GraphCliffRegressor(38,13,hidden_size=config["hidden_size"],num_layers=config["num_layers"],dropout=0)
            base_state = {k:v.clone() for k,v in base.state_dict().items()}
            del base
            for arm in config["arms"]:
                folder = dataset_out / f"seed{seed}" / arm["name"]
                summary = train_arm(config, arm, seed, frame, graphs, tp, vp, base_state, folder, device, scale)
                summary["dataset"] = dataset
                summaries.append(summary)
                save_json(output / "summary.json", summaries)
                print(f"COMPLETED {dataset} seed={seed} arm={arm['name']} ({len(summaries)} runs)", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    save_json(output / "completed.json", dict(runs=len(summaries), test_evaluated=False, status=config["status"]))
    print("ALL_DONE", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.csv_root, args.output)

if __name__ == "__main__":
    main()
