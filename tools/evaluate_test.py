"""冻结验证后的单次最终评估；先完成全部无查询标签预测，再读取test标签计算指标。"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT))
from freeze_validation import validate_freeze
from graphcliff_pair.data import digest
from graphcliff_pair.predict import run as predict_smiles
from graphcliff_pair.train import metrics
import numpy as np
import pandas as pd
from rdkit import Chem

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def evaluate(freeze_path,csv_root,output,resume_predictions=False):
    frozen=validate_freeze(freeze_path)
    output=Path(output)
    if (output/'completed.json').exists() or (output/'test_metrics.json').exists():
        raise FileExistsError('test已评估，禁止重复运行或按test调整配置')
    if output.exists() and not resume_predictions: raise FileExistsError('输出已存在；仅可显式恢复未完成的预测阶段')
    output.mkdir(parents=True,exist_ok=True)
    state_path=output/'prediction_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'freeze_sha256':digest(freeze_path),'predictions':{},'labels_loaded':False}
    if state['freeze_sha256']!=digest(freeze_path) or state['labels_loaded']:
        raise ValueError('冻结身份改变或已进入标签评估阶段，禁止恢复')
    save(state_path,state)
    rows={}
    for dataset,sha in frozen['data_sha256'].items():
        source=Path(csv_root)/f'{dataset}.csv'
        if digest(source)!=sha: raise ValueError('源数据哈希改变')
        # 此阶段不读取y/cliff_mol，仅核对结构、分区和canonical重复。
        structures=pd.read_csv(source,usecols=['smiles','split'])
        train_valid=set();test=set()
        test_rows=structures.index[structures['split']=='test'].tolist()
        if not test_rows: raise ValueError('test分区为空')
        for i,row in structures.iterrows():
            mol=Chem.MolFromSmiles(row.smiles)
            if mol is None or not mol.GetNumAtoms(): raise ValueError(f'test前结构检查失败: {dataset} row={i}')
            canonical=Chem.MolToSmiles(mol,canonical=True)
            (test if row['split']=='test' else train_valid).add(canonical)
        overlap=train_valid & test
        if overlap: raise ValueError(f'{dataset} test与开发集有{len(overlap)}个canonical重复；停止、不静默改划分')
        rows[dataset]=test_rows
        query_path=output/f'{dataset}_query_smiles.csv'
        queries=structures.loc[test_rows,['smiles']].reset_index(drop=True)
        if query_path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(query_path),queries)
        else: queries.to_csv(query_path,index=False)
    for entry in frozen['models']:
        key=f"{entry['dataset']}_seed{entry['seed']}_{entry['arm']}"
        prediction_path=output/f'{key}_predictions.csv'
        if key in state['predictions']:
            if not prediction_path.is_file() or digest(prediction_path)!=state['predictions'][key]: raise ValueError('预测缓存身份不匹配')
            continue
        predict_smiles(ROOT/entry['run_dir'],Path(csv_root)/f"{entry['dataset']}.csv",output/f"{entry['dataset']}_query_smiles.csv",prediction_path)
        state['predictions'][key]=digest(prediction_path)
        save(state_path,state)
    if len(state['predictions'])!=78: raise ValueError('预测未覆盖完整固定矩阵')
    # 完成所有模型预测后才开始标签评价，任何中断都不能自动重新选模型。
    state['labels_loaded']=True
    save(state_path,state)
    labels={d:pd.read_csv(Path(csv_root)/f'{d}.csv',usecols=['y','cliff_mol']).loc[ids].reset_index(drop=True) for d,ids in rows.items()}
    records=[]
    for entry in frozen['models']:
        dataset,seed,arm=entry['dataset'],entry['seed'],entry['arm']
        key=f'{dataset}_seed{seed}_{arm}'
        predictions=pd.read_csv(output/f'{key}_predictions.csv')
        truth=labels[dataset]
        if len(predictions)!=len(truth) or predictions.query_index.tolist()!=list(range(len(truth))): raise ValueError('测试预测行号/数量不匹配')
        if not np.isfinite(truth.y).all() or not truth.cliff_mol.isin([0,1]).all(): raise ValueError('测试标签非法')
        predictions['y']=truth.y
        predictions['cliff_mol']=truth.cliff_mol
        result=metrics(predictions.to_dict('records'))
        records.append(dict(dataset=dataset,seed=seed,arm=arm,**result,test_evaluated=True,
                            checkpoint_sha256=entry['checkpoint_sha256'],predictions_sha256=state['predictions'][key]))
    save(output/'test_metrics.json',{'records':records,'freeze_sha256':digest(freeze_path),'test_evaluated':True,'model_selection_on_test':False})
    save(output/'completed.json',{'runs':78,'test_evaluated':True,'model_selection_on_test':False})
    print('TEST_COMPLETE 78 fixed checkpoint evaluations; no test-based selection')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    parser.add_argument('--csv-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume-predictions',action='store_true')
    args=parser.parse_args()
    evaluate(args.freeze,args.csv_root,args.output,args.resume_predictions)
