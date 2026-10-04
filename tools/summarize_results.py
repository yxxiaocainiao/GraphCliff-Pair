"""固定矩阵的三种子汇总与条件消融，不挑选最佳种子或组合。"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_runs import audit
from tools.report_diagnostics import nearest_markdown

TASKS=['CHEMBL234_Ki','CHEMBL244_Ki']
SEEDS=[42,43,44]
INTERACTION=['direct','global','pair_mlp','cross']
FACTORIAL={
    (0,0,0):'global', (1,0,0):'cross',
    (0,1,0):'global_fp', (1,1,0):'cross_fp',
    (0,0,1):'global_dynamic', (1,0,1):'cross_dynamic',
    (0,1,1):'global_fp_dynamic', (1,1,1):'cross_fp_dynamic'
}
FULL=INTERACTION+[name for name in FACTORIAL.values() if name not in INTERACTION]+['global_fp_static','cross_fp_static','pair_mlp_fp']
METRICS=['overall_rmse','cliff_rmse','noncliff_rmse','mae']

def compare(frame,task,before,after,label):
    index=frame.set_index(['dataset','seed','arm'])
    result={'dataset':task,'comparison':label,'reference':before,'treatment':after,'n_seeds':3}
    for metric in METRICS:
        a=np.array([index.at[(task,s,before),metric] for s in SEEDS],dtype=float)
        b=np.array([index.at[(task,s,after),metric] for s in SEEDS],dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            result[metric]=None
            continue
        delta=b-a
        result[metric]={'reference_mean':float(a.mean()),'treatment_mean':float(b.mean()),
                        'paired_delta_mean':float(delta.mean()),'paired_delta_sd':float(delta.std(ddof=1)),
                        'relative_change_percent':float((b.mean()-a.mean())/a.mean()*100) if a.mean()>0 else None,
                        'improved_seeds':int((delta<0).sum()),'seed_deltas':dict(zip(map(str,SEEDS),map(float,delta)))}
    return result

def interaction_gate(comparisons):
    tasks=[]
    for task in TASKS:
        controls=[r for r in comparisons if r['dataset']==task and r['comparison'] in ['attention_vs_global','attention_vs_mlp']]
        checks=[]
        for result in controls:
            cliff=result['cliff_rmse'];overall=result['overall_rmse']
            valid=cliff is not None and overall is not None and cliff['relative_change_percent'] is not None and overall['relative_change_percent'] is not None
            passes=bool(valid and cliff['relative_change_percent']<=-2 and overall['relative_change_percent']<=1 and cliff['improved_seeds']>=2)
            checks.append({'control':result['reference'],'pass':passes})
        tasks.append({'dataset':task,'controls':checks,'pass':len(checks)==2 and all(c['pass'] for c in checks)})
    return {'go':all(t['pass'] for t in tasks),'tasks':tasks,'meaning':'resource expansion gate, not significance or publication standard'}

def analyze(audited,phase):
    arms=INTERACTION if phase=='interaction' else FULL
    expected=set(itertools.product(TASKS,SEEDS,arms))
    records=audited['records']
    actual=[(r['dataset'],r['seed'],r['arm']) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected or audited['test_evaluated']:
        raise ValueError(f"矩阵不完整、重复或越界：预期 {len(expected)} 次")
    frame=pd.DataFrame(records)
    groups=[]
    for task,arm in itertools.product(TASKS,arms):
        selected=frame[(frame.dataset==task)&(frame.arm==arm)]
        summary={'dataset':task,'arm':arm,'n_seeds':3}
        for metric in METRICS+['parameters','epochs_run','optimizer_steps','train_graph_forwards','elapsed_seconds','peak_cuda_mb','morgan_valid_pair_count','morgan_valid_cliff_pair_count','morgan_signed_delta_mae','morgan_sign_accuracy','morgan_delta_spearman']:
            values=pd.to_numeric(selected[metric],errors='coerce').dropna() if metric in selected else pd.Series(dtype=float)
            summary[metric]={'mean':float(values.mean()) if len(values) else None,
                             'sd':float(values.std(ddof=1)) if len(values)>1 else None,'n':len(values)}
        groups.append(summary)
    comparisons=[]
    for task in TASKS:
        comparisons.extend([
            compare(frame,task,'global','cross','attention_vs_global'),
            compare(frame,task,'pair_mlp','cross','attention_vs_mlp'),
            compare(frame,task,'direct','cross','attention_vs_direct')])
        if phase=='full':
            for factor,label in enumerate(['attention','fppool','dynamic']):
                for bits in FACTORIAL:
                    if bits[factor]==0:
                        changed=list(bits);changed[factor]=1
                        comparisons.append(compare(frame,task,FACTORIAL[bits],FACTORIAL[tuple(changed)],f'{label}_conditional_{bits}'))
            comparisons.extend([
                compare(frame,task,'global_fp_static','global_fp_dynamic','dynamic_vs_static_global_fp'),
                compare(frame,task,'cross_fp_static','cross_fp_dynamic','dynamic_vs_static_cross_fp'),
                compare(frame,task,'pair_mlp_fp','cross_fp','attention_vs_mlp_fp'),
                compare(frame,task,'direct','cross_fp_dynamic','full_recipe_vs_direct')])
    baselines=audited.get('nearest_reference',[])
    if baselines and (len(baselines)!=len(TASKS) or {r['dataset'] for r in baselines}!=set(TASKS)):
        raise ValueError('最近邻标签对照必须每任务唯一')
    weights=[dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],best_epoch=r['best_epoch'],**r['weight_diagnostics'])
             for r in records if 'weight_diagnostics' in r]
    return {'phase':phase,'runs':len(records),'summary':groups,'comparisons':comparisons,
            'nearest_reference':baselines,'weight_diagnostics':weights,
            'interaction_expansion_gate':interaction_gate(comparisons),'test_evaluated':False,
            'limitations':'two previously explored development tasks; 3 seeds per task; pairs share molecules; no significance claim'}

def write_markdown(path,result):
    lines=['# '+('三种子交互验证' if result['phase']=='interaction' else '三项机制完整消融'),'',
           f"实际完成并审计 {result['runs']} 次训练；每任务每组3种子。test尚未评估。",'',
           'RMSE与MAE沿用输入y的负对数活性尺度；Cliff/Non-cliff按原cliff_mol分子标记分组，不是分子对差值指标。', '',
           '|任务|组|Overall RMSE（均值±SD）|Cliff RMSE（均值±SD）|Non-cliff RMSE（均值±SD）|MAE（均值±SD）|',
           '|---|---|---:|---:|---:|---:|']
    def value(item):
        return 'NA' if item['mean'] is None else f"{item['mean']:.4f} ± {item['sd']:.4f}" if item['sd'] is not None else f"{item['mean']:.4f}"
    for r in result['summary']:
        lines.append(f"|{r['dataset']}|{r['arm']}|{value(r['overall_rmse'])}|{value(r['cliff_rmse'])}|{value(r['noncliff_rmse'])}|{value(r['mae'])}|")
    lines+=nearest_markdown(result.get('nearest_reference',[]))
    lines+=['','## 验证内部相似分子对：辅助诊断','',
            '仅验证分区内部Morgan radius2/1024 Tanimoto≥0.8的无序对，方向为原行号b>a的y_b−y_a。差值MAE沿用y尺度，符号准确率和Spearman无量纲；真实差值为0的对不计符号准确率，支持不足记NA。该配对不等于官方cliff判定或严格MMP，共享端点的对不是独立重复。','',
            '|任务|组|相似对数|差值≥1的对数|Signed delta MAE|Sign accuracy|Delta Spearman|',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        values=[value(r[n]) for n in ['morgan_valid_pair_count','morgan_valid_cliff_pair_count','morgan_signed_delta_mae','morgan_sign_accuracy','morgan_delta_spearman']]
        lines.append(f"|{r['dataset']}|{r['arm']}|"+'|'.join(values)+'|')
    lines+=['','## 条件比较','', '负变化表示误差下降；胜出种子只计本任务三个seed，不将分子对当独立重复。', '',
            '|任务|比较|Overall变化%|Cliff变化%|Cliff胜出seed|','|---|---|---:|---:|---:|']
    for c in result['comparisons']:
        o=c['overall_rmse'];f=c['cliff_rmse']
        op='NA' if o is None or o['relative_change_percent'] is None else f"{o['relative_change_percent']:.2f}"
        fp='NA' if f is None or f['relative_change_percent'] is None else f"{f['relative_change_percent']:.2f}"
        wins='NA' if f is None else f"{f['improved_seeds']}/3"
        lines.append(f"|{c['dataset']}|{c['reference']} → {c['treatment']}|{op}|{fp}|{wins}|")
    lines+=['','## 实际计算预算','',
            '各组分别报告三种子的均值±SD。训练步数由完整训练查询数、batch大小和实际history轮数核算；图前向计训练查询及参考：direct每查询1张，pair每查询2张，不包括验证或测试图。完整拟合耗时包含各轮验证、checkpoint操作及最后重载评估，不能解释为纯训练GPU时间。早停导致实际轮数与工作量不同；峰值显存字段peak_cuda_mb实际单位为MiB（2^20字节）。','',
            '|任务|组|参数数|实际轮数|训练步数|训练图前向次数|完整拟合秒数|峰值显存MiB|',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        budgets=[value(r[name]) for name in ['parameters','epochs_run','optimizer_steps','train_graph_forwards','elapsed_seconds','peak_cuda_mb']]
        lines.append(f"|{r['dataset']}|{r['arm']}|"+'|'.join(budgets)+'|')
    weights=[w for w in result.get('weight_diagnostics',[]) if w['mode']!='mse']
    if weights:
        lines+=['','## 训练权重分布','',
                '重建已发生的训练批次：使用原train pairs、独立shuffle seed、batch size和冻结Loss公式；逐epoch最小/最大值已与history核对（允许CPU/CUDA float32舍入差）。下表为既定最佳checkpoint所在epoch，每个训练查询计一次。完整各轮分位数在JSON；MSE的所有权重恒为1。权重无量纲，不涉及验证或测试标签。','',
                '|任务|组|seed|最佳epoch|min|median|q95|max|均值|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in weights:
            w=row['epoch_distributions'][row['best_epoch']-1]
            values=[f"{w[n]:.4f}" for n in ['min','median','q95','max','mean']]
            lines.append(f"|{row['dataset']}|{row['arm']}|{row['seed']}|{row['best_epoch']}|"+'|'.join(values)+'|')
    gate=result['interaction_expansion_gate']['go']
    lines+=['','## 预定门槛与限制','',f"交互阶段额外任务扩展门槛：{'Go' if gate else 'No-Go'}。这不代表统计显著性或发表条件。",
            '仅两项既有开发任务，不能推广到全部MoleculeACE。均值、方差和逐seed差值全部保留，不挑最优seed或组合。完整组合对基线的总差值不能归因给单个模块。',
            'FPPool比较衡量原SAG读出替换为Morgan-only FPPool的整体作用，不能单凭该对照把收益解释为指纹先验本身。动态Loss对普通MSE比较；静态权重对照仅覆盖FPPool条件，容量对照仅覆盖MSE条件。三个seed使用同一数据划分，主要反映训练随机性。']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--csv-root',required=True)
    parser.add_argument('--phase',choices=['interaction','full'],required=True)
    parser.add_argument('--output-prefix',required=True)
    args=parser.parse_args()
    audited=audit(args.runs,args.csv_root)
    result=analyze(audited,args.phase)
    prefix=Path(args.output_prefix)
    if prefix.with_suffix('.json').exists() or prefix.with_suffix('.md').exists():
        raise FileExistsError('报告已存在，拒绝覆盖')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    write_markdown(prefix.with_suffix('.md'),result)
    print(f"SUMMARY_OK {result['runs']} audited runs; gate={result['interaction_expansion_gate']['go']}")

if __name__=='__main__':
    main()
