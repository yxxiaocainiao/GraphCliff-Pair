# SPDX-License-Identifier: GPL-3.0-only
import argparse,json
from pathlib import Path
import numpy as np
from graphcliff_pair import train

def report(output):
    output=Path(output);a=json.loads((output/"audit.json").read_text());m=json.loads((output/"manifest.json").read_text());c=m["config"]
    if a["runs"]!=18 or c["seeds"]!=[42,43,44] or "limit_train_queries" in c:raise ValueError("gate requires full fixed pilot")
    decisions=[];lines=["# 原GraphCliff＋ACA-MSE：三任务三种子验证", "", "当前仅为训练Loss迁移探索，不是原创方法或二区录用证据。只评估验证集，无测试标签解析或测试推断。", "", "| 数据集 | MSE Cliff均值±标准差 | ACA Cliff均值±标准差 | Cliff相对变化 | Overall相对变化 | 改善种子数 |", "|---|---:|---:|---:|---:|---:|"]
    for d in c["datasets"]:
        base=[next(r for r in a["rows"] if r["dataset"]==d and r["seed"]==s and r["arm"]=="mse") for s in c["seeds"]]
        aca=[next(r for r in a["rows"] if r["dataset"]==d and r["seed"]==s and r["arm"]=="aca") for s in c["seeds"]]
        b=np.array([r["cliff_rmse"] for r in base]);v=np.array([r["cliff_rmse"] for r in aca]);change=float(v.mean()/b.mean()-1)
        overall=float(np.mean([r["overall_rmse"] for r in aca])/np.mean([r["overall_rmse"] for r in base])-1);wins=int((v<b).sum())
        decisions.append(dict(dataset=d,cliff_relative_change=change,overall_relative_change=overall,improving_seeds=wins,qualifies=change<=-.03 and wins>=2))
        lines.append(f"| {d} | {b.mean():.6f}±{b.std(ddof=1):.6f} | {v.mean():.6f}±{v.std(ddof=1):.6f} | {change*100:+.2f}% | {overall*100:+.2f}% | {wins}/3 |")
    go=sum(r["qualifies"] for r in decisions)>=2 and all(r["overall_relative_change"]<=.01 and r["cliff_relative_change"]<=.03 for r in decisions)
    decision=dict(go=go,decisions=decisions,test_evaluated=False)
    train.save_json(output/"decision.json",decision)
    lines += ["", "判定："+("Go，只支持讨论下一步，不自动扩展30任务。" if go else "No-Go，停止当前固定ACA-MSE迁移；不追加参数搜索、FPPool、任务或30任务训练。"), "", "预定门槛：至少两任务Cliff均值改善≥3%，各有至少2/3种子改善；任一任务Overall均值不得恶化>1%，Cliff均值不得恶化>3%。这些是探索门槛，不是期刊要求。", "", "三个任务的官方训练标签经训练行校准，与−log10(nM)一致；不是标准化标签。第三任务按至少20个验证cliff分子的覆盖条件，再选最小训练规模；仍需谨慎解读有限样本。", "", f"18个最佳检查点已在训练设备重放，最大预测绝对差{max(r['max_replay_error'] for r in a['rows']):.8g}。两组模型参数量及每seed初始权重完全一致。", "", "复用官方v3 GPLv3 ACALoss，alpha0/0.1，cliff上下界1，p2，squared=True；squared同时改变回归与embedding距离形式。这不是原论文MAE设置的逐项复现，也未证明迁移系数最优。结构gate关闭；只在训练batch标签上构建三元组。", "", "## 每次运行", "", "| 数据集 | Seed | 组 | Overall RMSE | Cliff RMSE | Noncliff RMSE | 轮数/最佳 | 秒 | 候选/高价值三元组 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in a["rows"]:lines.append(f"| {r['dataset']} | {r['seed']} | {r['arm']} | {r['overall_rmse']:.6f} | {r['cliff_rmse']:.6f} | {r['noncliff_rmse']:.6f} | {r['epochs']}/{r['best_epoch']} | {r['elapsed_seconds']:.1f} | {r['candidate_triplets']}/{r['high_value_triplets']} |")
    lines += ["", f"训练源码提交：`{m['code_commit']}`；启动时工作树有改动：`{m['code_dirty']}`。原始CSV、模型和逐分子预测保留在ignored artifacts目录；只公开聚合审计。旧实验/队列不重启。", ""]
    (output/"results.md").write_text("\n".join(lines),encoding="utf-8",newline="\n");print("GO" if go else "NO_GO",flush=True);return go

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);report(p.parse_args().output)
