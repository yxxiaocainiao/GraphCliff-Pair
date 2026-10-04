"""Read already audited outputs only; never import training or open source/test CSVs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["artifacts/interaction_seed42_20261004",
        "artifacts/interaction_seed43_44_20261004",
        "artifacts/ablation_seed42_20261004"]
TASKS = ["CHEMBL234_Ki", "CHEMBL244_Ki"]
SEEDS = [42, 43, 44]
ARMS = ["direct", "global", "pair_mlp", "cross"]
CONTRASTS = [("global", "direct"), ("cross", "global"), ("cross", "pair_mlp")]
AUDIT = ROOT / "docs/validation_through_seed42_audit.json"


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def locations():
    result = {}
    for run in RUNS:
        folder = ROOT / run
        manifest = read(folder / "manifest.json")
        completion = read(folder / "completed.json")
        config = manifest["config"]
        keys = [(d, s, arm["name"]) for d in config["datasets"]
                for s in config["seeds"] for arm in config["arms"]]
        assert completion["runs"] == len(keys) and completion["test_evaluated"] is False
        assert len(read(folder / "summary.json")) == len(keys)
        for key in keys:
            assert key not in result, f"duplicate model: {key}"
            result[key] = folder / key[0] / f"seed{key[1]}" / key[2]
    return result


def prepare(output):
    # Freeze before loading any per-query validation errors.
    assert not output.exists(), "Analysis plan already exists; no re-binning allowed"
    output.parent.mkdir(parents=True, exist_ok=True)
    lookup = locations()
    assert len(lookup) == 42
    evidence = {tuple(row[k] for k in ("dataset", "seed", "arm")): row
                for row in read(AUDIT)["evidence"]}
    assert set(lookup) == set(evidence)
    tasks = {}
    for task in TASKS:
        path = ROOT / RUNS[0] / task / "pairs.json"
        pairs = read(path)
        values = np.asarray([p["similarity"] for p in pairs["train_pairs"]])
        inner = sorted(set(float(x) for x in np.quantile(values, [.25, .5, .75])
                           if 0 < x < 1))
        tasks[task] = dict(edges=[0.] + inner + [1.],
                           bin_rule="[lower,upper), final bin includes 1; tied edges merged",
                           training_pair_count=len(values), scale=pairs["scale"],
                           pairs_path=path.relative_to(ROOT).as_posix(), pairs_sha256=sha(path))
    write(output, dict(status="frozen_before_stratified_errors", runs=RUNS, tasks=tasks,
                       audit_sha256=sha(AUDIT), model_count=42,
                       candidate_rule=dict(contrasts=CONTRASTS, task_order=TASKS,
                           seeds=SEEDS, compare="Q1 minus Q4 paired squared-error penalty",
                           minimum_queries_per_extreme_bin=20,
                           same_direction_all_three_seeds=True,
                           same_direction_after_dropping_largest_absolute_penalty_in_each_bin=True,
                           minimum_mean_gap_fraction_of_direct_overall_mse=.05,
                           interpretation="Pragmatic exploratory screening, not power or significance")))


def metrics(frame):
    residual = frame.prediction.to_numpy() - frame.y.to_numpy()
    cliff = frame.cliff_mol.to_numpy() == 1
    def rmse(mask):
        return float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else None
    return dict(overall_rmse=float(np.sqrt(np.mean(residual ** 2))),
                cliff_rmse=rmse(cliff), noncliff_rmse=rmse(~cliff),
                mae=float(np.mean(np.abs(residual))), count=len(frame), cliff_count=int(cliff.sum()))


def load_checked(plan):
    assert sha(AUDIT) == plan["audit_sha256"], "Historical audit changed"
    historical = read(AUDIT)
    records = {tuple(row[k] for k in ("dataset", "seed", "arm")): row
               for row in historical["records"]}
    evidence = {tuple(row[k] for k in ("dataset", "seed", "arm")): row
                for row in historical["evidence"]}
    lookup = locations()
    assert set(lookup) == set(records) == set(evidence) and len(lookup) == 42
    frames, sources, canonical = {}, {}, {}
    columns = ["query", "reference", "similarity", "y", "reference_y", "cliff_mol"]
    for key, folder in lookup.items():
        hashes = {}
        for name, field in [("validation_predictions.csv", "predictions_sha256"),
                            ("history.json", "history_sha256"), ("best.pt", "checkpoint_sha256")]:
            hashes[name] = sha(folder / name)
            assert hashes[name] == evidence[key][field], f"Changed audited file: {folder/name}"
        summary = read(folder / "summary.json")
        assert (summary["arm"], summary["seed"], summary["test_evaluated"]) == (key[2], key[1], False)
        frame = pd.read_csv(folder / "validation_predictions.csv", float_precision="round_trip")
        assert frame[columns + ["prediction"]].notna().all().all()
        assert np.isfinite(frame[columns + ["prediction"]].to_numpy()).all()
        assert not frame["query"].duplicated().any()
        assert frame.cliff_mol.isin([0, 1]).all()
        assert frame.similarity.between(0, 1).all()
        actual = metrics(frame)
        for name, value in actual.items():
            for saved in [summary, records[key]]:
                if value is None:
                    assert saved[name] is None
                else:
                    assert np.isclose(value, saved[name], rtol=1e-9, atol=1e-9), (key, name)
        task = key[0]
        if task not in canonical:
            canonical[task] = frame[columns].copy()
        else:
            pd.testing.assert_frame_equal(frame[columns], canonical[task], check_exact=True)
        pair_path = folder.parents[1] / "pairs.json"
        assert sha(pair_path) == plan["tasks"][task]["pairs_sha256"], "Pairing changed between stages"
        pairing = read(pair_path)
        assert frame["query"].tolist() == pairing["valid_rows"]
        assert set(frame.reference).issubset(set(pairing["train_rows"]))
        assert not set(pairing["train_rows"]) & set(pairing["valid_rows"])
        pair_table = pd.DataFrame(pairing["valid_pairs"])
        pd.testing.assert_frame_equal(frame[["query", "reference", "similarity"]], pair_table,
                                      check_exact=True)
        history = read(folder / "history.json")
        assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
        best = min(history, key=lambda row: row["valid_mse"])["epoch"]
        assert best == summary["best_epoch"] == records[key]["best_epoch"]
        assert len(history) == summary["epochs_run"] == records[key]["epochs_run"]
        frames[key] = frame
        sources["/".join(map(str, key))] = dict(folder=folder.relative_to(ROOT).as_posix(),
                                             **hashes, summary_sha256=sha(folder / "summary.json"))
    return frames, dict(runs=42, status="passed", sources=sources,
                        checks=["audited prediction/history/checkpoint bytes unchanged",
                                "all metrics independently recomputed",
                                "identical query/reference/labels/similarity across all arms and seeds",
                                "train-only reference row membership and recorded validation pairs",
                                "best epoch equals minimum saved validation MSE"],
                        limitations=["No raw source CSV opened; source label provenance inherited from historical audit",
                                     "Checkpoint hashed only; no inference/replay or training performed",
                                     "No test labels or interrupted-stage outputs read"])


def groups(frame, plan, task):
    edges = plan["tasks"][task]["edges"]
    return np.searchsorted(edges[1:-1], frame.similarity.to_numpy(), side="right")


def dropped_mean(values):
    if len(values) < 2:
        return None
    return float(np.mean(np.delete(values, int(np.argmax(np.abs(values))))))


def reference_results(frames, plan):
    layers, candidates, contrasts = [], [], []
    for task in TASKS:
        bins = groups(frames[(task, 42, "direct")], plan, task)
        nbins = len(plan["tasks"][task]["edges"]) - 1
        for seed in SEEDS:
            for arm in ARMS + ["nearest_reference"]:
                frame = frames[(task, seed, "direct" if arm == "nearest_reference" else arm)].copy()
                if arm == "nearest_reference":
                    frame["prediction"] = frame.reference_y
                for b in range(nbins):
                    part = frame.loc[bins == b]
                    layers.append(dict(dataset=task, seed=seed, arm=arm, bin=b,
                                       **metrics(part)) if len(part) else
                                  dict(dataset=task, seed=seed, arm=arm, bin=b, count=0, cliff_count=0))
        for treatment, reference in CONTRASTS:
            seed_rows = []
            for seed in SEEDS:
                treated = frames[(task, seed, treatment)]
                baseline = frames[(task, seed, reference)]
                penalty = (treated.prediction - treated.y).to_numpy() ** 2 - (
                    baseline.prediction - baseline.y).to_numpy() ** 2
                means, dropped = [], []
                for b in range(nbins):
                    vals = penalty[bins == b]
                    means.append(float(vals.mean()) if len(vals) else None)
                    dropped.append(dropped_mean(vals))
                    contrasts.append(dict(dataset=task, seed=seed, treatment=treatment,
                                          reference=reference, bin=b, count=len(vals),
                                          mean_squared_error_penalty=means[-1],
                                          drop_max_absolute_penalty=dropped[-1]))
                seed_rows.append(dict(seed=seed, gap=means[0]-means[-1]
                                      if means[0] is not None and means[-1] is not None else None,
                                      dropped_gap=dropped[0]-dropped[-1]
                                      if dropped[0] is not None and dropped[-1] is not None else None))
            low_count, high_count = int((bins == 0).sum()), int((bins == nbins-1).sum())
            gaps = [r["gap"] for r in seed_rows]
            dropped_gaps = [r["dropped_gap"] for r in seed_rows]
            base_mse = float(np.mean([np.mean((frames[(task, s, "direct")].prediction -
                                              frames[(task, s, "direct")].y) ** 2) for s in SEEDS]))
            valid = all(x is not None for x in gaps + dropped_gaps)
            consistent = valid and (all(x > 0 for x in gaps + dropped_gaps) or
                                    all(x < 0 for x in gaps + dropped_gaps))
            mean_gap = float(np.mean(gaps)) if valid else None
            material = valid and abs(mean_gap) >= .05 * base_mse
            enough = min(low_count, high_count) >= 20
            candidates.append(dict(dataset=task, treatment=treatment, reference=reference,
                                   low_count=low_count, high_count=high_count, seeds=seed_rows,
                                   mean_gap=mean_gap, direct_mse=base_mse,
                                   gap_fraction_of_direct_mse=abs(mean_gap)/base_mse if valid else None,
                                   direction_consistent_after_drop=bool(consistent),
                                   material=bool(material), enough_queries=enough,
                                   passes=bool(consistent and material and enough)))
    # Priority fixed before looking at stratified errors: contrast, then task.
    ordered = sorted(candidates, key=lambda r: (CONTRASTS.index((r["treatment"], r["reference"])),
                                               TASKS.index(r["dataset"])))
    selected = next((row for row in ordered if row["passes"]), None)
    return dict(stage="D1_reference_reliability", layers=layers, contrasts=contrasts,
                candidate_checks=ordered, selected_candidate=selected,
                decision="continue_D2" if selected else "pause_no_useful_stable_pattern",
                inference="Exploratory association only; no new mechanism cause has been demonstrated")


def plot_reference(result, plan, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"global": "#0072B2", "pair_mlp": "#009E73", "cross": "#D55E00"}
    for col, task in enumerate(TASKS):
        pairing = read(ROOT / plan["tasks"][task]["pairs_path"])
        edges = plan["tasks"][task]["edges"]
        n = len(edges)-1
        pos = np.arange(n)
        labels = [f"[{edges[i]:.3f}, {edges[i+1]:.3f}{']' if i==n-1 else ')'}" for i in pos]
        train = np.searchsorted(edges[1:-1], [p["similarity"] for p in pairing["train_pairs"]], side="right")
        valid = np.searchsorted(edges[1:-1], [p["similarity"] for p in pairing["valid_pairs"]], side="right")
        counts = np.bincount(valid, minlength=n)
        axes[0, col].bar(pos-.18, np.bincount(train, minlength=n)/len(train), .36,
                         label="Training pairs", color="#999999")
        axes[0, col].bar(pos+.18, counts/len(valid), .36, label="Validation queries", color="#56B4E9")
        for i, count in enumerate(counts):
            axes[0, col].text(i+.18, count/len(valid)+.012, f"n={count}", ha="center", fontsize=9)
        axes[0, col].set(title=task, ylabel="Fraction", xticks=pos, xticklabels=labels, ylim=(0, .36))
        axes[0, col].legend(fontsize=9, loc="upper center", ncol=2)
        direct = {(r["seed"], r["bin"]): r["overall_rmse"]**2 for r in result["layers"]
                  if r["dataset"] == task and r["arm"] == "direct" and r["count"]}
        for arm in colors:
            values = np.array([[next(r["overall_rmse"]**2 for r in result["layers"]
                                      if r["dataset"]==task and r["arm"]==arm and r["seed"]==s
                                      and r["bin"]==b)-direct[s,b] for b in pos] for s in SEEDS])
            axes[1, col].errorbar(pos, values.mean(axis=0), yerr=values.std(axis=0, ddof=1),
                                  marker="o", capsize=3, label=arm, color=colors[arm])
            for values_seed in values:
                axes[1, col].plot(pos, values_seed, color=colors[arm], alpha=.2, linewidth=.8)
        axes[1, col].axhline(0, color="black", linewidth=.8)
        axes[1, col].set(ylabel="MSE(model) - MSE(direct)", xlabel="Training-fixed similarity bins",
                         xticks=pos, xticklabels=labels)
        axes[1, col].legend(fontsize=9)
    fig.suptitle("Reference similarity diagnostics: 3 training seeds, 1 fixed split\n"
                 "Error bars: between-seed SD; positive penalty is worse", fontsize=13)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output / f"reference_similarity.{ext}", dpi=220)
    plt.close(fig)


def markdown_reference(result, plan, path):
    lines = ["# D1：参考相似度与性能退化", "",
             "仅使用已审计输出，不读取源CSV或test标签；D0的42组文件身份、指标和样本对齐检查通过。",
             "分层由训练配对相似度四分位数预先固定，边界重复时合并。当前候选筛选是探索性的，不是因果或显著性检验。", "",
             "![参考相似度诊断](reference_similarity.png)", "",
             "图中误差条为同一固定划分上三个训练seed之间的SD，不是置信区间；逐seed细线也全部保留。", "",
             "## 固定的筛选规则", "",
             "比较最低与最高相似度层的配对平方误差差。候选须两层均至少20分子，三seed方向一致，",
             "各层剔除最大绝对误差差样本后方向仍一致，并且平均层间差达到direct总体MSE的5%。",
             "20样本和5%是本轮预先固定的实用筛选门槛，不是论文参数、功效保证或显著性界限。", ""]
    for task in TASKS:
        edges = plan["tasks"][task]["edges"]
        lines += [f"## {task}", "", f"边界：`{edges}`。", "",
                  "|层|分子数|cliff分子数|direct RMSE|global RMSE|MLP RMSE|cross RMSE|参考标签 RMSE|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for b in range(len(edges)-1):
            rows = [r for r in result["layers"] if r["dataset"]==task and r["bin"]==b]
            first = rows[0]
            values = [float(np.mean([r["overall_rmse"] for r in rows if r["arm"]==a]))
                      for a in ARMS + ["nearest_reference"]]
            lines.append(f"|Q{b+1}|{first['count']}|{first['cliff_count']}|" +
                         "|".join(f"{x:.4f}" for x in values)+"|")
        lines += ["", "上表模型RMSE为逐seed RMSE的均值。参考标签对照无训练随机性，仅一个划分，",
                  "不把相同参考预测称为三次独立重复。", ""]
    lines += ["## 各候选的完整检查", "",
              "gap = Q1的MSE处理−对照差 − Q4的同一差；正值代表低相似度层更不利。", "",
              "|任务|处理−对照|seed42 gap|seed43 gap|seed44 gap|占direct MSE|删除极端样本仍一致|通过|",
              "|---|---|---:|---:|---:|---:|---|---|"]
    for r in result["candidate_checks"]:
        values = "|".join(f"{x['gap']:+.4f}" if x["gap"] is not None else "NA" for x in r["seeds"])
        ratio = f"{100*r['gap_fraction_of_direct_mse']:.1f}%" if r["gap_fraction_of_direct_mse"] is not None else "NA"
        lines.append(f"|{r['dataset']}|{r['treatment']}−{r['reference']}|{values}|{ratio}|"
                     f"{'是' if r['direction_consistent_after_drop'] else '否'}|{'是' if r['passes'] else '否'}|")
    lines += ["", f"当前决策：`{result['decision']}`。", "",
              "候选优先级在分析前固定为global−direct、cross−global、cross−MLP，各对照内按234、244任务顺序。",
              "仅继续首个通过候选的D2检查，不挑选最有利的阈值；所有候选与反例均保留。",
              "如果没有候选通过，按用户授权立即暂停，D2/D3和后续训练均不启动。",
              "当前结果仅限这两个开发任务和当前实现；每层的cliff数是分子标记，不是查询—参考对标签。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "reference"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/diagnostics")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "analysis_plan.json"
    if args.phase == "prepare":
        prepare(plan_path)
        print("Frozen training-derived bins and candidate screening before stratified validation errors")
        return
    assert not (output / "reference_results.json").exists(), "Existing analysis must not be silently replaced"
    plan = read(plan_path)
    frames, checks = load_checked(plan)
    write(output / "input_check.json", checks)
    result = reference_results(frames, plan)
    result["plan_sha256"] = sha(plan_path)
    result["input_check_sha256"] = sha(output / "input_check.json")
    write(output / "reference_results.json", result)
    plot_reference(result, plan, output)
    markdown_reference(result, plan, output / "reference_results.md")
    print(json.dumps(dict(checked_models=checks["runs"], decision=result["decision"],
                          selected_candidate=result["selected_candidate"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
