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
    for record in records:
        task,seed,arm=record['dataset'],record['seed'],record['arm']
        entry=models[(task,seed,arm)]
        path=output/f'{task}_seed{seed}_{arm}_predictions.csv'
        if record['checkpoint_sha256']!=entry['checkpoint_sha256'] or digest(path)!=record['predictions_sha256']:
            raise ValueError('测试权重/预测身份不一致')
        pred=pd.read_csv(path);truth=frames[task]
        if len(pred)!=len(truth) or pred.query_index.tolist()!=list(range(len(truth))) or pred.smiles.tolist()!=truth.smiles.tolist():
            raise ValueError('测试查询行号或结构顺序不一致')
        pred['y']=truth.y;pred['cliff_mol']=truth.cliff_mol
        recalculated=metrics(pred.to_dict('records'))
        for name,value in recalculated.items():
            if (value is None and record[name] is not None) or (value is not None and (record[name] is None or not np.isclose(value,record[name],rtol=1e-9,atol=1e-9))):
                raise ValueError(f'测试指标不一致: {task}/{seed}/{arm}/{name}')
        evidence.append({'dataset':task,'seed':seed,'arm':arm,'predictions_sha256':digest(path),'checkpoint_sha256':entry['checkpoint_sha256']})
    result['evidence']=evidence
    prefix=Path(prefix)
    if prefix.with_suffix('.json').exists() or prefix.with_suffix('.md').exists(): raise FileExistsError('报告已存在，拒绝覆盖')
    lines=['# 最终测试：固定全部模型','', '完成78个固定checkpoint的评估及预测/标签/指标核对；未按test选择模型。','',
           '|任务|组|Overall RMSE（均值±SD）|Cliff RMSE（均值±SD）|','|---|---|---:|---:|']
    def value(v): return 'NA' if v['mean'] is None else f"{v['mean']:.4f} ± {v['sd']:.4f}"
    for row in result['summary']:
        lines.append(f"|{row['dataset']}|{row['arm']}|{value(row['overall_rmse'])}|{value(row['cliff_rmse'])}|")
    lines+=['','全部条件比较和逐seed差保留在同名JSON；测试结果不触发调参、最优组选择或开发扩展门槛。',
            '仅两项已有开发任务及三个seed；训练参考来自train。不能将分子对当独立重复，不能宣称统计显著或推广到全部MoleculeACE。']
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
