"""完整验证结束后冻结报告、权重及推理来源，作为最终test评估的前置条件。"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT))
from audit_runs import audit
from summarize_results import analyze,TASKS,SEEDS,FULL
from graphcliff_pair.data import digest
from replay_validation import check_training_sources,replay,validate_replay,CORE_SOURCES

REQUIRED_SOURCES={'graphcliff_pair/model.py','graphcliff_pair/predict.py','graphcliff_pair/data.py',
                  'graphcliff_pair/fppool.py','graphcliff_pair/fingerprint.py','graphcliff_pair/external.py',
                  'graphcliff_pair/vendor/model.py','graphcliff_pair/vendor/dataset_utils.py',
                  'docs/sources.json','tools/evaluate_test.py','tools/replay_validation.py'} | CORE_SOURCES

def relative(path,root=ROOT):
    resolved=Path(path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('冻结模型/报告路径必须在项目目录内')
    return resolved.relative_to(root.resolve()).as_posix()

def freeze(folders,csv_root,report_json,output,replay_json=None,device=None):
    output=Path(output)
    if output.exists(): raise FileExistsError('冻结文件已存在，拒绝覆盖')
    for folder in map(Path,folders):
        check_training_sources(json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    audited=audit(folders,csv_root)
    expected=analyze(audited,'full')
    recorded=json.loads(Path(report_json).read_text(encoding='utf-8'))
    if recorded!=expected:
        raise ValueError('完整验证报告与独立重算不一致')
    models=[]
    for folder in map(Path,folders):
        manifest=json.loads((folder/'manifest.json').read_text())
        config=manifest['config']
        if config['hidden_size']!=256 or config['num_layers']!=3 or config['epochs']!=100 or config.get('limit_train_queries') or config.get('limit_valid_queries'):
            raise ValueError('必须完成正式全量查询预算，不接受smoke代替')
        for dataset,seed,arm in itertools.product(config['datasets'],config['seeds'],config['arms']):
            model_dir=folder/dataset/f'seed{seed}'/arm['name']
            models.append(dict(dataset=dataset,seed=seed,arm=arm['name'],run_dir=relative(model_dir),
                               checkpoint_sha256=digest(model_dir/'best.pt'),
                               manifest_sha256=digest(folder/'manifest.json'),pairs_sha256=digest(folder/dataset/'pairs.json')))
    if replay_json is None:
        replay_json=output.with_suffix('.replay.json')
        replay(folders,csv_root,replay_json,device)
    evidence=validate_replay(replay_json,models)
    data_hashes={d:digest(Path(csv_root)/f'{d}.csv') for d in TASKS}
    if evidence.get('data_sha256')!=data_hashes:
        raise ValueError('重放数据身份与待冻结数据不一致')
    sources=[p for p in (ROOT/'graphcliff_pair').rglob('*.py')]+[ROOT/'docs/sources.json',ROOT/'tools/evaluate_test.py',ROOT/'tools/replay_validation.py']
    result=dict(status='validation_complete_frozen',runs=78,test_evaluated=False,
                report=relative(report_json),report_sha256=digest(report_json),models=models,
                validation_replay=dict(path=relative(replay_json),sha256=digest(replay_json)),
                source_sha256={relative(p):digest(p) for p in sources},
                data_sha256=data_hashes)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('FREEZE_OK 78 validated checkpoints; test not evaluated')

def validate_freeze(path,root=ROOT):
    path=Path(path)
    if not path.is_file(): raise ValueError('缺少完整验证冻结文件，禁止test评估')
    frozen=json.loads(path.read_text(encoding='utf-8'))
    expected=set(itertools.product(TASKS,SEEDS,FULL))
    actual=[(m['dataset'],m['seed'],m['arm']) for m in frozen.get('models',[])]
    if frozen.get('status')!='validation_complete_frozen' or frozen.get('test_evaluated') is not False or frozen.get('runs')!=78 or len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('冻结文件没有覆盖完整验证矩阵')
    if not REQUIRED_SOURCES.issubset(frozen.get('source_sha256',{})) or set(frozen.get('data_sha256',{}))!=set(TASKS):
        raise ValueError('冻结来源或数据身份不完整')
    def protected(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError('冻结路径越出项目范围')
        return p
    if digest(protected(frozen['report']))!=frozen['report_sha256']:
        raise ValueError('冻结验证报告被修改')
    report=json.loads(protected(frozen['report']).read_text(encoding='utf-8'))
    if report.get('phase')!='full' or report.get('runs')!=78 or report.get('test_evaluated') is not False:
        raise ValueError('验证报告不符合完整消融范围')
    for name,expected_hash in frozen['source_sha256'].items():
        if digest(protected(name))!=expected_hash: raise ValueError(f'推理来源改变: {name}')
    for m in frozen['models']:
        model_dir=protected(m['run_dir'])
        if digest(model_dir/'best.pt')!=m['checkpoint_sha256'] or digest(model_dir.parents[2]/'manifest.json')!=m['manifest_sha256'] or digest(model_dir.parents[1]/'pairs.json')!=m['pairs_sha256']:
            raise ValueError('冻结权重或运行身份被修改')
    proof=frozen.get('validation_replay',{})
    if not proof.get('path') or not proof.get('sha256'):
        raise ValueError('缺少全部冻结权重的validation重放证明')
    replay_path=protected(proof['path'])
    if digest(replay_path)!=proof['sha256']:
        raise ValueError('验证重放证明被修改')
    evidence=validate_replay(replay_path,frozen['models'],root)
    if evidence.get('data_sha256')!=frozen['data_sha256']:
        raise ValueError('冻结与重放数据身份不一致')
    return frozen

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--csv-root',required=True)
    parser.add_argument('--report-json',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--replay-json',help='可复用已完成的全78重放证据；省略则在冻结前执行重放')
    parser.add_argument('--device',choices=['cpu','cuda'],help='自动重放使用的设备')
    args=parser.parse_args()
    freeze(args.runs,args.csv_root,args.report_json,args.output,args.replay_json,args.device)
