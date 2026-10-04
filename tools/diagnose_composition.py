"""Composition check for the single preselected D1 candidate, using saved outputs."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnose_reference import (read, write, document_sha, load_checked, groups,
                                      dropped_mean, SEEDS)


def prepare(output):
    target = output / "composition_plan.json"
    assert not target.exists(), "Do not replace the composition plan"
    previous = read(output / "reference_results.json")
    assert previous["decision"] == "continue_D2"
    candidate = previous["selected_candidate"]
    plan = read(output / "analysis_plan.json")
    write(target, dict(candidate={k: candidate[k] for k in ["dataset", "treatment", "reference"]},
                       expected_gap_direction=int(np.sign(candidate["mean_gap"])),
                       delta_boundary=plan["tasks"][candidate["dataset"]]["scale"],
                       delta_boundary_source="full training-pair median abs delta, already recorded",
                       partitions=["molecule_cliff", "delta_magnitude", "joint"],
                       minimum_queries_per_extreme_bin=20,
                       continuation_rule="At least one stable subgroup in each of molecule_cliff and delta_magnitude; joint descriptive only",
                       stable_rule="same D1 gap direction in all 3 seeds and after dropping max abs penalty per cell",
                       previous_results_lf_text_sha256=document_sha(output / "reference_results.json"),
                       previous_plan_lf_text_sha256=document_sha(output / "analysis_plan.json")))


def calculate(frames, analysis, plan):
    candidate = plan["candidate"]
    task, treatment, reference = (candidate[k] for k in ["dataset", "treatment", "reference"])
    base = frames[(task, 42, reference)]
    bins = groups(base, analysis, task)
    last = len(analysis["tasks"][task]["edges"])-2
    delta = np.abs((base.y - base.reference_y).to_numpy())
    cliff = base.cliff_mol.to_numpy() == 1
    large = delta > plan["delta_boundary"]
    partitions = {
        "molecule_cliff": [("noncliff", ~cliff), ("cliff", cliff)],
        "delta_magnitude": [("small_delta", ~large), ("large_delta", large)],
        "joint": [(f"{c}_{d}", (cliff if c == "cliff" else ~cliff) &
                   (large if d == "large" else ~large)) for c in ["noncliff", "cliff"]
                  for d in ["small", "large"]],
    }
    rows, checks = [], []
    for partition, subsets in partitions.items():
        for label, mask in subsets:
            counts = [int((mask & (bins == b)).sum()) for b in [0, last]]
            seed_gaps = []
            for seed in SEEDS:
                a, b = frames[(task, seed, treatment)], frames[(task, seed, reference)]
                penalty = (a.prediction-a.y).to_numpy()**2 - (b.prediction-b.y).to_numpy()**2
                cells = []
                for name, group in [("Q1", 0), ("Q4", last)]:
                    selected = mask & (bins == group)
                    values = penalty[selected]
                    mean = float(values.mean()) if len(values) else None
                    trimmed = dropped_mean(values)
                    cells.append((mean, trimmed))
                    rows.append(dict(partition=partition, subgroup=label, similarity_bin=name,
                                     seed=seed, count=len(values), mean_penalty=mean,
                                     dropped_penalty=trimmed,
                                     mean_abs_delta=float(delta[selected].mean()) if len(values) else None))
                gap = cells[0][0]-cells[1][0] if all(x[0] is not None for x in cells) else None
                trimmed_gap = cells[0][1]-cells[1][1] if all(x[1] is not None for x in cells) else None
                seed_gaps.append(dict(seed=seed, gap=gap, dropped_gap=trimmed_gap))
            values = [r[field] for r in seed_gaps for field in ["gap", "dropped_gap"]]
            same = all(v is not None and np.sign(v)==plan["expected_gap_direction"] for v in values)
            enough = min(counts) >= plan["minimum_queries_per_extreme_bin"]
            checks.append(dict(partition=partition, subgroup=label, low_count=counts[0],
                               high_count=counts[1], seeds=seed_gaps,
                               same_direction_all_seeds_and_drop=bool(same), enough_queries=enough,
                               stable=bool(same and enough)))
    supported = {p: [r["subgroup"] for r in checks if r["partition"]==p and r["stable"]]
                 for p in ["molecule_cliff", "delta_magnitude"]}
    proceed = all(supported.values())
    return dict(stage="D2_composition", candidate=candidate, delta_boundary=plan["delta_boundary"],
                rows=rows, checks=checks, stable_subgroups=supported,
                decision="continue_D3" if proceed else "pause_composition_evidence_insufficient",
                limitation="Within-subgroup association rules out neither causal confounding nor training/inference failure; small joint cells are descriptive")


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    names = {"molecule_cliff": "Molecule cliff mark", "delta_magnitude": "Absolute target delta",
             "joint": "Joint cells (descriptive)"}
    for ax, partition in zip(axes, names):
        checks = [r for r in result["checks"] if r["partition"]==partition]
        for i, check in enumerate(checks):
            for group, offset, color in [("Q1", -.12, "#56B4E9"), ("Q4", .12, "#0072B2")]:
                values = [r["mean_penalty"] for r in result["rows"] if r["partition"]==partition
                          and r["subgroup"]==check["subgroup"] and r["similarity_bin"]==group]
                if all(v is not None for v in values):
                    ax.errorbar(np.mean(values), i+offset, xerr=np.std(values, ddof=1), fmt="o",
                                capsize=3, color=color, label=group if i==0 else None)
        labels = [f"{r['subgroup'].replace('_', ' ')}\nQ1 n={r['low_count']}; Q4 n={r['high_count']}" for r in checks]
        ax.set(yticks=range(len(checks)), yticklabels=labels, title=names[partition],
               xlabel="MSE(global) - MSE(direct)")
        ax.axvline(0, color="black", linewidth=.8)
        ax.tick_params(axis="y", labelsize=9)
        ax.margins(y=.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"{result['candidate']['dataset']}: fixed reference-pattern composition\n"
                 f"Small delta <= training median {result['delta_boundary']:.4f}; bars are 3-seed SD",
                 fontsize=13)
    for ext in ["png", "svg", "pdf"]:
        target = output / f"delta_cliff_composition.{ext}"
        fig.savefig(target, dpi=220)
        if ext == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())
                              + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def markdown(result, output):
    lines = ["# D2：活性差值与分子cliff构成", "",
             f"仅检验D1按固定优先级选出的`{result['candidate']}`；没有改分层或另选候选。", "",
             f"差值幅度界限为已记录训练配对的中位绝对差`{result['delta_boundary']:.8f}`；",
             "small为不大于该值，large为大于该值。没有根据验证误差选界限。",
             "cliff是分子标记，不代表当前参考对构成cliff。", "",
             "![差值与cliff构成](delta_cliff_composition.png)", "",
             "误差条为同一划分上3个训练seed的SD；横坐标是处理−对照MSE，正值表示global更差。", "",
             "## 全部子组检查", "",
             "gap仍定义为Q1处理−对照差减Q4同一差。只有两端各至少20分子、3seed及最大误差差删除后",
             "都保留D1方向的子组，才作为下一步的信息依据。joint只描述，不额外搜索子组。", "",
             "|分组|子组|Q1数|Q4数|seed42 gap|seed43 gap|seed44 gap|删除后方向一致|够样本|稳定|",
             "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for r in result["checks"]:
        values = "|".join(f"{s['gap']:+.4f}" if s["gap"] is not None else "NA" for s in r["seeds"])
        lines.append(f"|{r['partition']}|{r['subgroup']}|{r['low_count']}|{r['high_count']}|{values}|"
                     f"{'是' if r['same_direction_all_seeds_and_drop'] else '否'}|"
                     f"{'是' if r['enough_queries'] else '否'}|{'是' if r['stable'] else '否'}|")
    lines += ["", f"稳定子组：`{result['stable_subgroups']}`。", "",
              f"阶段决策：`{result['decision']}`。", "",
              "继续D3要求cliff划分和差值大小划分各至少一个稳定子组；任何不满足都按用户授权暂停。",
              "这种一致性是筛选信息价值，不是因果、显著性或泛化证明。小样本子组保留数值但不用于通过门槛。",
              "即使一个子组稳定，也不能推出该子组所有分子都退化，或仅凭本结果决定训练配对阈值。", ""]
    (output / "composition_results.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "composition"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/diagnostics")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "prepare":
        prepare(output)
        print("D2 partitions and continuation rule saved before composition errors")
        return
    assert not (output / "composition_results.json").exists(), "Do not silently replace results"
    plan = read(output / "composition_plan.json")
    assert document_sha(output / "reference_results.json")==plan["previous_results_lf_text_sha256"]
    assert document_sha(output / "analysis_plan.json")==plan["previous_plan_lf_text_sha256"]
    analysis = read(output / "analysis_plan.json")
    frames, _ = load_checked(analysis)
    result = calculate(frames, analysis, plan)
    result["plan_lf_text_sha256"] = document_sha(output / "composition_plan.json")
    write(output / "composition_results.json", result)
    plot(result, output)
    markdown(result, output)
    print({"decision": result["decision"], "stable_subgroups": result["stable_subgroups"]})


if __name__ == "__main__":
    main()
