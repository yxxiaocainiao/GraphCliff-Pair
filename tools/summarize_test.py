"""最终测试完成后的完整矩阵核对与汇总；不进行模型选择或扩展判断。"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from freeze_validation import validate_freeze
from summarize_results import analyze
from graphcliff_pair.data import digest
from graphcliff_pair.train import metrics
from graphcliff_pair.predict import training_rows_only
from tools.report_diagnostics import check_metrics, nearest_markdown

def summarize(records,freeze_hash):
    if not records or any(r.get('test_evaluated') is not True for r in records):
        raise ValueError('仅接受完整最终test记录')
    # 复用固定矩阵和条件比较；测试结果不能触发开发阶段的扩展门槛。
    result=analyze({'records':records,'test_evaluated':False},'full')
    result.pop('interaction_expansion_gate')
    result.update(phase='final_test',test_evaluated=True,model_selection_on_test=False,freeze_sha256=freeze_hash)
    return result

def run(freeze_path,output,csv_root,prefix):
    frozen=validate_freeze(freeze_path)
    output=Path(output)
    complete=json.loads((output/'completed.json').read_text(encoding='utf-8'))
    recorded=json.loads((output/'test_metrics.json').read_text(encoding='utf-8'))
    freeze_hash=digest(freeze_path)
    if complete!={'runs':78,'test_evaluated':True,'model_selection_on_test':False} or recorded.get('freeze_sha256')!=freeze_hash or recorded.get('test_evaluated') is not True or recorded.get('model_selection_on_test') is not False:
        raise ValueError('测试完成标记或冻结身份不一致')
    records=recorded['records']
    # 先检查固定矩阵完整性，再逐个重算，避免重复/缺失结果进入报告。
    result=summarize(records,freeze_hash)
    models={(m['dataset'],m['seed'],m['arm']):m for m in frozen['models']}
    frames={}
    for task,sha in frozen['data_sha256'].items():
        source=Path(csv_root)/f'{task}.csv'
        if digest(source)!=sha: raise ValueError('测试数据身份改变')
        frame=pd.read_csv(source,usecols=['smiles','split','y','cliff_mol'])
        frames[task]=frame.loc[frame['split']=='test'].reset_index(drop=True)
    evidence=[]
    banks,reference_rows,baselines={},{},{}
    for record in records:
        task,seed,arm=record['dataset'],record['seed'],record['arm']
        entry=models[(task,seed,arm)]
        path=output/f'{task}_seed{seed}_{arm}_predictions.csv'
        if record['checkpoint_sha256']!=entry['checkpoint_sha256'] or digest(path)!=record['predictions_sha256']:
            raise ValueError('测试权重/预测身份不一致')
        pred=pd.read_csv(path);truth=frames[task]
        if len(pred)!=len(truth) or pred.query_index.tolist()!=list(range(len(truth))) or pred.smiles.tolist()!=truth.smiles.tolist():
            raise ValueError('测试查询行号或结构顺序不一致')
        pairing=json.loads((ROOT/entry['run_dir']).parents[1].joinpath('pairs.json').read_text(encoding='utf-8'))
        train_rows=pairing['train_rows']
        if task not in banks:
            banks[task]=training_rows_only(Path(csv_root)/f'{task}.csv',train_rows)
        elif banks[task].index.tolist()!=train_rows:
            raise ValueError('同任务训练参考池不同')
        refs=pred.reference_row.tolist()
        if any(r not in banks[task].index for r in refs):
            raise ValueError('测试参考超出冻结训练池')
        if task in reference_rows and reference_rows[task]!=refs:
            raise ValueError('同任务测试参考选择不一致')
        reference_rows[task]=refs
        pred['y']=truth.y;pred['cliff_mol']=truth.cliff_mol
        recalculated=metrics(pred.to_dict('records'))
        for name,value in recalculated.items():
            if (value is None and record[name] is not None) or (value is not None and (record[name] is None or not np.isclose(value,record[name],rtol=1e-9,atol=1e-9))):
                raise ValueError(f'测试指标不一致: {task}/{seed}/{arm}/{name}')
        if task not in baselines:
            nearest=pred.copy()
            nearest['prediction']=banks[task].loc[refs,'y'].to_numpy()
            baselines[task]=dict(dataset=task,**metrics(nearest.to_dict('records')))
        evidence.append({'dataset':task,'seed':seed,'arm':arm,'predictions_sha256':digest(path),'checkpoint_sha256':entry['checkpoint_sha256']})
    result['evidence']=evidence
    stored_baselines=recorded.get('nearest_reference',[])
    if len(stored_baselines)!=len(baselines) or {b['dataset'] for b in stored_baselines}!=set(baselines):
        raise ValueError('最终test缺少每任务唯一的Top-1参考标签基线')
    for row in stored_baselines:
        expected={k:v for k,v in baselines[row['dataset']].items() if k!='dataset'}
        try: check_metrics(expected,row,'最终test Top-1')
        except AssertionError as error: raise ValueError(str(error)) from error
    result['nearest_reference']=[baselines[d] for d in sorted(baselines)]
    prefix=Path(prefix)
    if prefix.with_suffix('.json').exists() or prefix.with_suffix('.md').exists(): raise FileExistsError('报告已存在，拒绝覆盖')
    lines=['# 最终测试：固定全部模型','', '完成78个固定checkpoint的评估及预测/标签/指标核对；未按test选择模型。','',
           '误差沿用输入y的负对数活性尺度。Cliff/Non-cliff按原cliff_mol分子标记分组。','',
           '|任务|组|Overall RMSE（均值±SD）|Cliff RMSE（均值±SD）|Non-cliff RMSE（均值±SD）|MAE（均值±SD）|','|---|---|---:|---:|---:|---:|']
    def value(v): return 'NA' if v['mean'] is None else f"{v['mean']:.4f} ± {v['sd']:.4f}"
    for row in result['summary']:
        lines.append(f"|{row['dataset']}|{row['arm']}|{value(row['overall_rmse'])}|{value(row['cliff_rmse'])}|{value(row['noncliff_rmse'])}|{value(row['mae'])}|")
    lines+=nearest_markdown(result['nearest_reference'])
    lines+=['','## 全部固定条件比较','',
            '变化为treatment − reference，负值为误差下降。方向数只计训练种子，不是显著性结论。','',
            '|任务|reference → treatment|ΔOverall RMSE|ΔCliff RMSE|Cliff改善seed|','|---|---|---:|---:|---:|']
    for row in result['comparisons']:
        overall,cliff=row['overall_rmse'],row['cliff_rmse']
        od='NA' if overall is None else f"{overall['paired_delta_mean']:+.4f}"
        cd='NA' if cliff is None else f"{cliff['paired_delta_mean']:+.4f}"
        wins='NA' if cliff is None else f"{cliff['improved_seeds']}/3"
        lines.append(f"|{row['dataset']}|{row['reference']} → {row['treatment']}|{od}|{cd}|{wins}|")
    lines+=['','全部条件比较和逐seed差保留在同名JSON；测试结果不触发调参、最优组选择或开发扩展门槛。',
            '仅两项已有开发任务及同一划分的三个训练seed；训练参考来自train。静态权重对照仅覆盖FPPool条件，容量对照仅覆盖MSE条件。不能将分子对当独立重复，不能宣称统计显著或推广到全部MoleculeACE。']
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('TEST_SUMMARY_OK 78 fixed models; predictions and metrics independently checked')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    parser.add_argument('--test-output',required=True)
    parser.add_argument('--csv-root',required=True)
    parser.add_argument('--output-prefix',required=True)
    args=parser.parse_args()
    run(args.freeze,args.test_output,args.csv_root,args.output_prefix)
