"""检查真实运行交付，不把低预算结果用作性能排名。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

def verify(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    complete=json.loads((folder/'completed.json').read_text())
    summaries=json.loads((folder/'summary.json').read_text())
    config=manifest['config']
    expected=len(config['datasets'])*len(config['seeds'])*len(config['arms'])
    if len(summaries)!=expected or complete['runs']!=expected:
        raise AssertionError('实际完成数量不符')
    heads,encoders,initial={},{},{}
    for item in summaries:
        key=(item['dataset'],item['seed'])
        encoders.setdefault(key,set()).add(item['initialization']['encoder_sha256'])
        if item['arm'] not in ['direct','pair_mlp']:
            heads.setdefault(key,set()).add(item['initialization']['head_sha256'])
        initial[(key,item['arm'])]=item['initialization']['full_sha256']
        path=folder/item['dataset']/f"seed{item['seed']}"/item['arm']
        pred=pd.read_csv(path/'validation_predictions.csv')
        history=json.loads((path/'history.json').read_text())
        observed=float(np.mean((pred.prediction-pred.y)**2))
        if abs(observed-item['overall_rmse']**2)>1e-5:
            raise AssertionError('报告和预测重算不符')
        if abs(observed-min(row['valid_mse'] for row in history))>1e-5:
            raise AssertionError('不是最低验证 MSE checkpoint')
        if not (path/'best.pt').exists() or not np.isfinite(pred.prediction).all() or item['test_evaluated']:
            raise AssertionError('checkpoint、有限预测或测试边界检查失败')
        if item['prediction_std']<=1e-8:
            raise AssertionError('预测为常量')
    if any(len(s)!=1 for s in encoders.values()) or any(len(s)!=1 for s in heads.values()):
        raise AssertionError('共享编码器或兼容 head 初值不一致')
    for key in encoders:
        if len({initial[(key,a)] for a in ['cross_fp','cross_fp_static','cross_fp_dynamic']})!=1:
            raise AssertionError('Loss 消融模型初值不一致')
    return dict(runs=expected, all_nonconstant_finite=True, predictions_recalculated=True,
                best_checkpoint_verified=True, shared_encoder_verified=True, shared_head_verified=True,
                loss_ablation_initialization_verified=True,test_evaluated=False,
                status='infrastructure verified; no efficacy claim')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    result=verify(args.folder)
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
