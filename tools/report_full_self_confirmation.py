"""Audit and replay all 12 fits before applying the registered seed gate."""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
from experiments.mechanism_retry.audit import audit
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.run import read_safe
from experiments.mechanism_retry.repro import stable_max_pool
from tools.run_full_self_confirmation import CONFIG, PROTOCOL, PREFLIGHT, sources


def report(output):
    torch, np = train.torch, train.np
    torch.set_num_threads(2)
    suite = json.loads((output / "suite.json").read_text())
    done = json.loads((output / "completed.json").read_text())
    assert done["completed_fits"] == suite["planned_fits"] == 12 and not done["test_evaluated"]
    assert suite["source_sha256"] == sources()
    assert suite["preflight_sha256"] == train.digest(PREFLIGHT)
    pre = json.loads(PREFLIGHT.read_text())
    cfg = json.loads(CONFIG.read_text())
    root = output / "runs"
    checked = audit(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["config"] == cfg and not manifest["code_dirty"] and not manifest["test_evaluated"]
    assert manifest["config_sha256"] == train.digest(CONFIG)
    assert manifest["protocol_sha256"] == train.digest(PROTOCOL)
    assert manifest["entrypoint_sha256"] == train.digest(train.ROOT / "tools/run_full_self_confirmation.py")
    assert manifest["code_commit"] == suite["code_commit"]
    assert manifest["preflight_sha256"] == train.digest(PREFLIGHT)
    rows, decisions, nearest = [], [], []
    old_pool = shared.global_max_pool
    shared.global_max_pool = stable_max_pool
    try:
        for source in pre["inputs"]:
            dataset = source["dataset"]
            assert train.digest(source["input_path"]) == source["input_sha256"]
            frame, graphs, tr, va, tp, vp = read_safe(source["input_path"], cfg["split_seed"])
            pairs = json.loads((root / dataset / "pairs.json").read_text())
            assert pairs["input_sha256"] == source["input_sha256"]
            assert pairs["train_rows"] == tr.tolist() and pairs["valid_rows"] == va.tolist()
            assert pairs["train_pairs"] == tp and pairs["valid_pairs"] == vp
            assert train.hashlib.sha256(json.dumps([tr.tolist(), va.tolist(), tp, vp], sort_keys=True).encode()).hexdigest() == source["pair_sha256"]
            assert set(tr).isdisjoint(va) and frame.loc[list(tr)+list(va), "split"].eq("train").all()
            assert all(p["reference"] in set(tr) and p["query"] != p["reference"] for p in tp+vp)
            nearest.append(dict(dataset=dataset, **json.loads((root / dataset / "nearest_reference.json").read_text())))
            identities = []
            for seed in cfg["seeds"]:
                heads = []
                for arm in cfg["arms"]:
                    folder = root / dataset / f"seed{seed}" / arm["name"]
                    summary = json.loads((folder / "summary.json").read_text())
                    history = json.loads((folder / "history.json").read_text())
                    assert 1 <= len(history) == summary["epochs_run"] <= cfg["epochs"]
                    assert summary["best_epoch"] == min(history, key=lambda x: x["valid_mse"])["epoch"]
                    assert [h["epoch"] for h in history] == list(range(1, len(history)+1))
                    assert len(history) == cfg["epochs"] or len(history)-summary["best_epoch"] == cfg["patience"]
                    assert all(h["weight_min"] == h["weight_max"] == 1 for h in history)
                    reference = next(x for x in pre["records"] if x["dataset"] == dataset and x["seed"] == seed and x["arm"] == arm["name"])
                    assert summary["parameters"] == reference["parameters"]
                    assert summary["initialization"]["full_sha256"] == reference["full_sha256"]
                    heads.append(summary["initialization"]["head_sha256"])
                    assert heads[-1] == reference["head_sha256"]
                    train.set_seed(SimpleNamespace(seed=seed))
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    model = LayerSwap(arm["variant"]).cuda()
                    checkpoint = torch.load(folder / "best.pt", map_location="cuda", weights_only=True)
                    assert checkpoint["epoch"] == summary["best_epoch"]
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    replay = train.pd.DataFrame(train.evaluate(model, arm, vp, graphs, frame, cfg["batch_size"], torch.device("cuda")))
                    saved = train.pd.read_csv(folder / "validation_predictions.csv", float_precision="round_trip")
                    keys = ["query", "reference", "y", "reference_y", "cliff_mol"]
                    np.testing.assert_array_equal(replay[keys], saved[keys])
                    assert saved["query"].tolist() == [p["query"] for p in vp]
                    assert saved["query"].is_unique and len(saved) == source["valid_count"]
                    np.testing.assert_allclose(replay.prediction, saved.prediction, atol=1e-5, rtol=1e-5)
                    for metric, value in train.metrics(replay.to_dict("records")).items():
                        if value is not None:
                            np.testing.assert_allclose(value, summary[metric], atol=1e-7, rtol=1e-7)
                    identities.append(saved[keys].to_dict("records"))
                    rows.append(dict(dataset=dataset, seed=seed, arm=arm["name"], overall_rmse=summary["overall_rmse"],
                                     cliff_rmse=summary["cliff_rmse"], noncliff_rmse=summary["noncliff_rmse"],
                                     epochs_run=len(history), best_epoch=summary["best_epoch"], parameters=summary["parameters"],
                                     optimizer_steps=len(history)*math.ceil(len(tp)/cfg["batch_size"]),
                                     elapsed_seconds=summary["elapsed_seconds"], peak_cuda_mb=summary["peak_cuda_mb"],
                                     checkpoint_sha256=train.digest(folder / "best.pt"),
                                     replay_max_abs_error=float(np.max(np.abs(replay.prediction-saved.prediction)))))
                    print("REPLAY_OK", dataset, seed, arm["name"], flush=True)
                    del model, checkpoint
                assert len(set(heads)) == 1
            assert all(x == identities[0] for x in identities)
    finally:
        shared.global_max_pool = old_pool
    aggregates = []
    for dataset in cfg["datasets"]:
        groups = {arm: [x for x in rows if x["dataset"] == dataset and x["arm"] == arm] for arm in ("full", "self")}
        mean_pass = {}
        for arm, items in groups.items():
            assert len(items) == 3
            aggregates.append(dict(dataset=dataset, arm=arm, **{metric: dict(mean=float(np.mean([x[metric] for x in items])),
                                 sample_std=float(np.std([x[metric] for x in items], ddof=1))) for metric in ("overall_rmse", "cliff_rmse")}))
        for metric in ("overall_rmse", "cliff_rmse"):
            mean_pass[metric] = float(np.mean([x[metric] for x in groups["self"]])) < float(np.mean([x[metric] for x in groups["full"]]))
        paired = []
        for full, self in zip(groups["full"], groups["self"]):
            assert full["seed"] == self["seed"]
            gains = {metric: full[metric]-self[metric] for metric in mean_pass}
            paired.append(dict(seed=full["seed"], rmse_gains=gains, both_improved=all(v > 0 for v in gains.values())))
        count = sum(x["both_improved"] for x in paired)
        decisions.append(dict(dataset=dataset, means_pass=mean_pass, paired=paired, both_improved_seeds=count,
                              passed=all(mean_pass.values()) and count >= 2))
    result = dict(status="Go for capacity/operator audit only" if all(x["passed"] for x in decisions) else "No-Go; stop this architecture round",
                  rows=rows, aggregates=aggregates, decisions=decisions, nearest_reference=nearest, audit=checked,
                  source_commit=suite["code_commit"], suite_sha256=train.digest(output / "suite.json"),
                  protocol_sha256=train.digest(PROTOCOL), preflight_sha256=train.digest(PREFLIGHT), reporter_sha256=train.digest(__file__),
                  formal_fits=12, train_epochs=sum(x["epochs_run"] for x in rows), optimizer_steps=sum(x["optimizer_steps"] for x in rows),
                  elapsed_seconds=sum(x["elapsed_seconds"] for x in rows), checkpoints_replayed=12,
                  validation_predictions_replayed=6*(292+247), test_evaluated=False, old_no_go_changed=False, significance_claim=False)
    destination = train.ROOT / "experiments/mechanism_retry/full_self_results_20261008"
    train.save_json(destination.with_suffix(".json"), result)
    lines = ["# M60 full/self多seed确认", "", result["status"], "",
             "固定2任务×3seed×2臂，全部新训练；按验证Overall选模，100epoch上限/patience15。仅开发集，不是独立test，也不宣称显著性。", "",
             "|任务|seed|臂|Overall RMSE|Cliff RMSE|best/实际epoch|", "|---|---:|---|---:|---:|---:|"]
    lines += [f"|{x['dataset']}|{x['seed']}|{x['arm']}|{x['overall_rmse']:.6f}|{x['cliff_rmse']:.6f}|{x['best_epoch']}/{x['epochs_run']}|" for x in rows]
    lines += ["", "均值±样本标准差(ddof=1)，同一固定划分的3个训练seed：", "", "|任务|臂|Overall|Cliff|", "|---|---|---:|---:|"]
    lines += [f"|{x['dataset']}|{x['arm']}|{x['overall_rmse']['mean']:.6f} ± {x['overall_rmse']['sample_std']:.6f}|{x['cliff_rmse']['mean']:.6f} ± {x['cliff_rmse']['sample_std']:.6f}|" for x in aggregates]
    lines += ["", "门槛：每任务双指标均值均优于full，且至少2/3 seed双指标同时改善；两任务均通过才继续容量/算子混杂核验。", ""]
    lines += [f"- {x['dataset']}：均值比较{x['means_pass']}，双指标同时改善{x['both_improved_seeds']}/3 seed，任务通过={x['passed']}。" for x in decisions]
    lines += ["", f"核验：12个checkpoint重放，最大绝对差{max(x['replay_max_abs_error'] for x in rows):.3g}；实际{result['train_epochs']}epoch/{result['optimizer_steps']} optimizer步骤，fit合计{result['elapsed_seconds']/60:.2f}分钟。", "",
              "边界：full6021198、self6810654参数，未匹配容量；两臂均使用训练参考活性恢复查询预测，不是单分子官方GraphCliff。三个seed不能证明统计显著性、化学机制或创新；通过筛查也不能直接归因去除LongPoly。GPU采用strict可用算子/native amax，但历史同seed分叉仍未解决，不承诺全程逐位。旧候选No-Go不改，不自动添加模块或扩训。逐seed差值、nearest-reference及耗时/显存/初始化/来源核验见同名JSON。", ""]
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(result["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output)
