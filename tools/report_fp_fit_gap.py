"""Read-only checkpoint diagnostic: residual fit on train versus validation."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments.mechanism_retry.audit import audit
from experiments.mechanism_retry.model import ResidualFP, SingleBaseline
from experiments.mechanism_retry.run import read_safe
from graphcliff_pair.data import digest
from graphcliff_pair.fingerprint import membership
from graphcliff_pair.train import ROOT, batches, set_seed


def report(output, destination, raw):
    checked = audit(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = manifest["config"]
    preflight = json.loads((ROOT / "experiments/mechanism_retry/fp_preflight_20261007.json").read_text())
    assert cfg["seeds"] == [42] and cfg["batch_size"] == 32
    assert str(torch.__version__) == manifest["torch"]
    config_path = ROOT / "experiments/mechanism_retry/fp_screen.json"
    assert json.loads(config_path.read_text()) == cfg
    # Historical configuration was CRLF; this checkout is LF. Do not rewrite it.
    historical_config = config_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert hashlib.sha256(historical_config).hexdigest() == manifest["config_sha256"] == preflight["config_sha256"]
    source_path = ROOT / "docs/sources.json"
    historical_sources = source_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert hashlib.sha256(historical_sources).hexdigest() == manifest["source_manifest_sha256"]
    dependencies = {}
    for relative in ("experiments/long_branch_swap/model.py", "experiments/long_branch_swap/run.py"):
        committed = subprocess.check_output(["git", "show", manifest["code_commit"]+":"+relative], cwd=ROOT)
        assert committed.replace(b"\r\n", b"\n") == (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        dependencies[relative] = digest(ROOT / relative)
    torch.set_num_threads(2)
    set_seed(SimpleNamespace(seed=42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw.mkdir(parents=True, exist_ok=True)
    records, replay, checkpoints, inputs = [], [], [], []
    for source in preflight["datasets"]:
        dataset = source["dataset"]
        assert digest(source["input_path"]) == source["input_sha256"]
        inputs.append(dict(dataset=dataset, path=source["input_path"], sha256=source["input_sha256"]))
        frame, graphs, tr, va, tp, vp = read_safe(source["input_path"], cfg["split_seed"])
        pairs = json.loads((output / dataset / "pairs.json").read_text())
        assert tr.tolist() == pairs["train_rows"] and va.tolist() == pairs["valid_rows"]
        assert tp == pairs["train_pairs"] and vp == pairs["valid_pairs"]
        assert set(tr).isdisjoint(va) and frame.loc[list(tr) + list(va), "split"].eq("train").all()
        assert len(tp) == source["train_count"] and len(vp) == source["valid_count"]
        for row, g in graphs.items():
            g.atom_fp = membership(SimpleNamespace(smiles=frame.at[row, "smiles"], x=g.x,
                                                  edge_index=g.edge_index, edge_attr=g.edge_attr))
        tables = {}
        for arm in (a for a in cfg["arms"] if a["name"] in ("base", "output_frozen", "output_frozen_null")):
            name = arm["name"]
            folder = output / dataset / "seed42" / name
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            summary = json.loads((folder / "summary.json").read_text())
            assert checkpoint["epoch"] == summary["best_epoch"]
            model = (SingleBaseline(38, 13, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=0)
                     if name == "base" else ResidualFP(cfg["hidden_size"], cfg["num_layers"], arm["space"], arm["frozen"], arm.get("shuffle", False)))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(device).eval()
            checkpoints.append(dict(dataset=dataset, arm=name, epoch=checkpoint["epoch"], sha256=digest(folder / "best.pt")))
            # Validation first: stop on replay failure before interpreting train fit.
            for partition, selected_pairs in (("valid", vp), ("train", tp)):
                predictions = []
                with torch.no_grad():
                    for q, r, yq, yr, selected in batches(selected_pairs, graphs, frame, cfg["batch_size"], device):
                        pred = model(q) if isinstance(model, ResidualFP) else model(q.x, q.edge_index, q.edge_attr, q.batch)
                        predictions.extend(dict(query=p["query"], reference=p["reference"], y=y,
                                                prediction=v, cliff_mol=int(frame.at[p["query"], "cliff_mol"]))
                                           for p, y, v in zip(selected, yq.flatten().tolist(), pred.flatten().tolist()))
                table = pd.DataFrame(predictions)
                assert table["query"].is_unique and len(table) == len(selected_pairs)
                assert np.isfinite(table[["y", "prediction"]].to_numpy()).all()
                if partition == "valid":
                    saved = pd.read_csv(folder / "validation_predictions.csv", float_precision="round_trip")
                    np.testing.assert_array_equal(table[["query", "reference", "y", "cliff_mol"]], saved[["query", "reference", "y", "cliff_mol"]])
                    np.testing.assert_allclose(table.prediction, saved.prediction, atol=2e-5, rtol=2e-5)
                    replay.append(dict(dataset=dataset, arm=name, count=len(table), max_abs_difference=float(np.max(np.abs(table.prediction - saved.prediction)))))
                table.to_csv(raw / f"{dataset}_{name}_{partition}.csv", index=False)
                tables[name, partition] = table
                print(dataset, name, partition, len(table), flush=True)
            del model
        for partition in ("train", "valid"):
            base = tables["base", partition]
            for name in ("output_frozen", "output_frozen_null"):
                table = tables[name, partition]
                np.testing.assert_array_equal(table[["query", "reference", "y", "cliff_mol"]], base[["query", "reference", "y", "cliff_mol"]])
                e = base.y.to_numpy() - base.prediction.to_numpy()
                c = table.prediction.to_numpy() - base.prediction.to_numpy()
                cliff = base.cliff_mol.to_numpy().astype(bool)
                for subset, mask in (("overall", np.ones(len(e), dtype=bool)), ("cliff", cliff), ("noncliff", ~cliff)):
                    assert mask.any()
                    base_mse, candidate_mse = float(np.mean(e[mask]**2)), float(np.mean((e[mask]-c[mask])**2))
                    alignment, energy = float(2*np.mean(e[mask]*c[mask])), float(np.mean(c[mask]**2))
                    np.testing.assert_allclose(base_mse-candidate_mse, alignment-energy, atol=1e-12, rtol=1e-12)
                    records.append(dict(dataset=dataset, arm=name, partition=partition, subset=subset, count=int(mask.sum()),
                                        base_rmse=base_mse**.5, candidate_rmse=candidate_mse**.5,
                                        mse_gain=base_mse-candidate_mse, twice_error_correction_product=alignment, correction_energy=energy))
    result = dict(date="2026-10-08", purpose="existing checkpoint train-versus-validation fit diagnostic",
                  device=str(device), audit=checked, source_commit=manifest["code_commit"],
                  tool_sha256=digest(Path(__file__)), inputs=inputs, checkpoints=checkpoints,
                  config_provenance=dict(workspace_sha256=digest(config_path), historical_crlf_sha256=manifest["config_sha256"], semantic_identity=True),
                  source_manifest_provenance=dict(workspace_sha256=digest(source_path), historical_crlf_sha256=manifest["source_manifest_sha256"]),
                  inherited_dependencies=dependencies,
                  replay=replay, rows=records, training_runs=0, optimizer_steps=0, test_evaluated=False,
                  raw_predictions=str(raw.resolve()), old_no_go_unchanged=True)
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# M58：既有FPPool checkpoint训练/验证残差诊断", "",
             "只做eval推断，不训练、不选新参数；主候选及历史No-Go不变。两个任务、seed42、6个既有best checkpoint；样本/拆分/配对/来源和冻结骨干经旧audit核验，验证预测须先通过原容差重放。", "",
             "令e=y-f₀，c=f-f₀，则MSE收益=E[e²]−E[(e−c)²]=2E[ec]−E[c²]。正值表示相对baseline改善。标签按原训练器float32转换；全部分解逐项核验恒等式。", "",
             "|任务|分支|分区|子集|n|Base RMSE|分支 RMSE|MSE收益|2E[ec]|E[c²]|",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append("|{dataset}|{arm}|{partition}|{subset}|{count}|{base_rmse:.6f}|{candidate_rmse:.6f}|{mse_gain:+.8f}|{twice_error_correction_product:+.8f}|{correction_energy:.8f}|".format(**row))
    lines += ["", "观察（主候选output_frozen，相对同任务baseline）：", ""]
    for dataset in cfg["datasets"]:
        for subset in ("overall", "cliff"):
            selected = {r["partition"]: r for r in records if r["dataset"] == dataset and r["arm"] == "output_frozen" and r["subset"] == subset}
            lines.append(f"- {dataset} {subset}：训练MSE收益{selected['train']['mse_gain']:+.8f}，验证{selected['valid']['mse_gain']:+.8f}。")
    lines += ["", "判读：训练收益不能替代验证收益；2E[ec]为负时，修正与baseline误差的平均乘积方向不利，且正的修正能量进一步增加MSE。分解只描述误差项，不识别原因，也不是添加尺度/门控的新授权。", "",
              "核验记录：初次执行因checkout LF与历史CRLF的SHA差异停止；配置内容一致，按历史CRLF重新编码后配置及来源清单SHA均完全匹配，未改历史文件。标签身份比较采用CSV round_trip解析，避免默认解析的4.4e-16舍入差；样本ID和标签仍要求精确相等。", "",
              "限制：训练和验证来自同一官方train内部划分，官方test未评估。checkpoint按历史验证Overall选取；这是所选checkpoint的拟合诊断，不是独立泛化评估或过拟合因果证明。Cliff子集沿用源数据cliff_mol标签，未重定义。null为固定分子内行反转，保留列计数，只是弱对照。仅一个seed，不推出FPPool普遍有效/无效，不据此救回失败候选。", "",
              f"验证重放：6个checkpoint，最大绝对差{max(r['max_abs_difference'] for r in replay):.3g}；0训练/optimizer步骤。原始预测仅留本地 `{raw.resolve()}`；机器可读来源/分解见同名JSON。", ""]
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    report(args.output, args.destination, args.raw)
