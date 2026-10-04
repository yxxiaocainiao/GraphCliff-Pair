"""独立重算已完成队列；核对分区、参考、初始化、选模和预算。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from graphcliff_pair.data import FP, digest, read_development
from graphcliff_pair.train import metrics
from tools.report_diagnostics import nearest_reference, check_metrics, reconstruct_weights, check_weight_history

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def pair_metrics(predictions, source):
    # 明确的辅助诊断，不冒充官方 OR-similarity/MMP 对。
    indexed = predictions.set_index("query")
    rows = sorted(indexed.index.tolist())
    fps = [FP.GetFingerprint(Chem.MolFromSmiles(source.at[row, "smiles"])) for row in rows]
    truth, prediction = [], []
    for i in range(len(rows)):
        scores = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        for j, score in enumerate(scores, i+1):
            if score >= 0.8:
                truth.append(float(indexed.at[rows[j], "y"]-indexed.at[rows[i], "y"]))
                prediction.append(float(indexed.at[rows[j], "prediction"]-indexed.at[rows[i], "prediction"]))
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    nonzero = np.abs(truth)>1e-8
    rho = None
    if len(truth)>=3 and np.ptp(truth)>1e-12 and np.ptp(prediction)>1e-12:
        rho = float(pd.Series(truth).rank().corr(pd.Series(prediction).rank()))
    return dict(morgan_valid_pair_count=len(truth), morgan_valid_cliff_pair_count=int((np.abs(truth)>=1).sum()),
                morgan_signed_delta_mae=float(np.mean(np.abs(truth-prediction))) if len(truth) else None,
                morgan_sign_accuracy=float(np.mean(np.sign(truth[nonzero])==np.sign(prediction[nonzero]))) if nonzero.any() else None,
                morgan_delta_spearman=rho)

def check_initializations(initializations,variants):
    if set(initializations)!=set(variants) or any(v not in ['direct','global_diff','cross_attention','pair_mlp'] for v in variants.values()):
        raise AssertionError('初始化与预定模型类型不一致')
    groups={(d,s) for d,s,a in initializations}
    for dataset,seed in groups:
        selected={arm:value for (d,s,arm),value in initializations.items() if (d,s)==(dataset,seed)}
        if len({v['encoder_sha256'] for v in selected.values()})!=1:
            raise AssertionError('编码器初始化不同')
        # pair_mlp输入为6H，不能加载2H的原reg_head；按variant而非arm别名识别。
        heads={value['head_sha256'] for (d,s,arm),value in initializations.items()
               if (d,s)==(dataset,seed) and variants[(d,s,arm)] in ['global_diff','cross_attention']}
        if len(heads)>1:
            raise AssertionError('兼容 head 初始化不同')
        for loss_group in [('global','global_dynamic'),('cross','cross_dynamic'),
                           ('global_fp','global_fp_static','global_fp_dynamic'),
                           ('cross_fp','cross_fp_static','cross_fp_dynamic')]:
            hashes={selected[name]['full_sha256'] for name in loss_group if name in selected}
            if len(hashes)>1:
                raise AssertionError(f'Loss 对照模型完整初值不同: {loss_group}')

def audit(folders, csv_root):
    records, evidence, initializations = [], [], {}
    seen, cached, variants = set(), {}, {}
    baselines, baseline_checked, weight_cache = {}, set(), {}
    training_hashes = None
    fixed_config = None
    for folder in map(Path, folders):
        manifest = read_json(folder/"manifest.json")
        complete = read_json(folder/"completed.json")
        config = manifest["config"]
        arm_variants={arm['name']:arm['variant'] for arm in config['arms']}
        core_config={k:v for k,v in config.items() if k not in ["status","datasets","seeds","arms"]}
        if fixed_config is None:
            fixed_config=core_config
        elif fixed_config!=core_config:
            raise AssertionError("队列的固定训练预算/超参数不一致")
        expected = {(d,s,a["name"]) for d in config["datasets"] for s in config["seeds"] for a in config["arms"]}
        summaries = read_json(folder/"summary.json")
        if len(summaries)!=len(expected) or complete["runs"]!=len(expected) or complete["test_evaluated"]:
            raise AssertionError("队列不完整或出现 test 评估")
        # 独立predict CLI不被train/evaluate导入，其后续I/O修正不改变训练身份。
        core = {k:v for k,v in manifest["source_sha256"].items()
                if k.replace('\\','/')!='graphcliff_pair/predict.py'}
        if training_hashes is None:
            training_hashes=core
        else:
            # 新增纯报告工具不应影响训练；已有训练模块必须身份一致。
            for key in training_hashes:
                if key not in core or core[key]!=training_hashes[key]:
                    raise AssertionError(f"训练源码身份不同: {key}")
        actual=set()
        for item in summaries:
            key=(item["dataset"],item["seed"],item["arm"])
            if key not in expected or key in seen:
                raise AssertionError(f"重复或未预定结果: {key}")
            seen.add(key);actual.add(key)
            dataset, seed, arm = key
            if dataset not in cached:
                source=Path(csv_root)/f"{dataset}.csv"
                frame, _, tr, va, tp, vp=read_development(source, config["split_seed"])
                cached[dataset]=(frame,tr,va,tp,vp,digest(source))
            frame,tr,va,tp,vp,data_hash=cached[dataset]
            scale=max(float(np.median([abs(frame.at[p['query'],'y']-frame.at[p['reference'],'y']) for p in tp])),1e-6)
            pairing=read_json(folder/dataset/"pairs.json")
            if pairing["input_sha256"]!=data_hash or pairing["train_rows"]!=tr.tolist() or pairing["valid_rows"]!=va.tolist():
                raise AssertionError("数据身份或原划分不符")
            tp=tp[:config.get("limit_train_queries",len(tp))]
            vp=vp[:config.get("limit_valid_queries",len(vp))]
            if pairing["train_pairs"]!=tp or pairing["valid_pairs"]!=vp:
                raise AssertionError("不是完整固定 Top-1 配对流")
            if not np.isclose(pairing['scale'],scale,rtol=1e-12,atol=1e-12) or not np.isclose(item['train_delta_scale'],scale,rtol=1e-12,atol=1e-12):
                raise AssertionError('Loss scale不是完整训练配对的中位绝对差')
            if (folder,dataset) not in baseline_checked:
                baseline=nearest_reference(frame,vp)
                check_metrics(baseline,read_json(folder/dataset/'nearest_reference.json'),'Top-1参考标签')
                if dataset in baselines:
                    check_metrics(baseline,baselines[dataset],'跨队列Top-1参考标签')
                baselines[dataset]=dict(dataset=dataset,**baseline)
                baseline_checked.add((folder,dataset))
            model_dir=folder/dataset/f"seed{seed}"/arm
            predictions=pd.read_csv(model_dir/"validation_predictions.csv")
            if not predictions["query"].is_unique or predictions["query"].tolist()!=[p["query"] for p in vp]:
                raise AssertionError("查询身份/顺序不符")
            if predictions["reference"].tolist()!=[p["reference"] for p in vp]:
                raise AssertionError("参考选择不符")
            if not np.allclose(predictions.y,frame.loc[predictions["query"],"y"]) or not np.allclose(predictions.reference_y,frame.loc[predictions["reference"],"y"]):
                raise AssertionError("标签或参考活性错位")
            if not np.array_equal(predictions.cliff_mol,frame.loc[predictions["query"],"cliff_mol"]):
                raise AssertionError("cliff 子集错位")
            result=metrics(predictions.to_dict("records"))
            for name in ["overall_rmse","cliff_rmse","noncliff_rmse","mae"]:
                if result[name] is None:
                    if item[name] is not None:
                        raise AssertionError("缺少支持量却报告指标")
                elif abs(result[name]-item[name])>1e-6:
                    raise AssertionError(f"指标重算不符: {name}")
            history=read_json(model_dir/"history.json")
            arm_config=next(a for a in config['arms'] if a['name']==arm)
            weight_key=(dataset,seed,arm_config['loss'])
            if weight_key not in weight_cache:
                weight_cache[weight_key]=reconstruct_weights(frame,tp,config,arm_config['loss'],seed,scale,config['epochs'])
            weight_distributions=weight_cache[weight_key][:len(history)]
            check_weight_history(weight_distributions,history)
            best=min(history,key=lambda h:h["valid_mse"])
            if item["best_epoch"]!=best["epoch"] or abs(result["overall_rmse"]**2-best["valid_mse"])>1e-5:
                raise AssertionError("未使用最低验证 Overall MSE")
            if not (model_dir/"best.pt").is_file() or item["test_evaluated"] or result["prediction_std"]<=1e-8:
                raise AssertionError("checkpoint、test 边界或预测变异检查失败")
            initialization=read_json(model_dir/"initialization.json")
            if initialization!=item["initialization"]:
                raise AssertionError("初始化记录不一致")
            initializations[key]=initialization
            variants[key]=arm_variants[arm]
            steps=((len(tp)+config["batch_size"]-1)//config["batch_size"])*len(history)
            record={k:v for k,v in item.items() if k!="initialization"}
            record.update(result,optimizer_steps=steps,**pair_metrics(predictions,frame))
            record['weight_diagnostics']={'mode':arm_config['loss'],'epoch_distributions':weight_distributions,
                                          'history_extrema_checked':True,'reconstructed_on':'cpu_float32',
                                          'definition':'normalized training batch weights, each training query once per epoch'}
            records.append(record)
            evidence.append(dict(dataset=dataset,seed=seed,arm=arm,predictions_sha256=digest(model_dir/"validation_predictions.csv"),
                                 checkpoint_sha256=digest(model_dir/"best.pt"),history_sha256=digest(model_dir/"history.json")))
        if actual!=expected:
            raise AssertionError("结果没有覆盖预定矩阵")
    check_initializations(initializations,variants)
    return dict(runs=len(records),records=records,evidence=evidence,nearest_reference=[baselines[d] for d in sorted(baselines)],
                verification="source/data/row/pair/label/initialization/selection/metrics checked",test_evaluated=False,
                pair_definition="validation-internal Morgan radius2/1024 Tanimoto>=0.8; delta=y_b-y_a; secondary diagnosis, not official cliff mask")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",nargs="+",required=True)
    parser.add_argument("--csv-root",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=audit(args.runs,args.csv_root)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(f"AUDIT_OK {result['runs']} complete runs; test_evaluated=False")

if __name__=="__main__":
    main()
