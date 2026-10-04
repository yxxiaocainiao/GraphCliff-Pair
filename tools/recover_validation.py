"""新目录恢复失败阶段：原样复制完整模型，未完成模型复用原训练入口从头运行。"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))

import numpy as np
import pandas as pd
import torch

from graphcliff_pair import train
from graphcliff_pair.data import digest
from replay_validation import check_training_sources, check_checkpoint_metadata
from tools.report_diagnostics import check_metrics

MODEL_FILES = ['best.pt', 'summary.json', 'history.json', 'validation_predictions.csv', 'initialization.json']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def completed_models(source, config):
    """仅完整summary中的模型可复用；有best.pt的中断模型不能冒充完成。"""
    items = read_json(source/'summary.json')
    expected = {(d,s,a['name']) for d in config['datasets'] for s in config['seeds'] for a in config['arms']}
    completed = {}
    for item in items:
        key = item['dataset'], item['seed'], item['arm']
        if key not in expected or key in completed:
            raise ValueError('原阶段包含重复或非预定模型')
        directory = source/key[0]/f'seed{key[1]}'/key[2]
        summary = read_json(directory/'summary.json')
        if dict(summary,dataset=key[0]) != item or summary['test_evaluated']:
            raise ValueError('原阶段完成记录与模型summary不符')
        check_checkpoint_metadata(torch.load(directory/'best.pt',map_location='cpu',weights_only=True),
                                  summary,read_json(directory/'history.json'))
        predictions = pd.read_csv(directory/'validation_predictions.csv')
        check_metrics(train.metrics(predictions.to_dict('records')),summary,'已完成模型')
        if read_json(directory/'initialization.json') != summary['initialization']:
            raise ValueError('已完成模型初始化身份不符')
        completed[key] = dict(summary=summary,files={name:digest(directory/name) for name in MODEL_FILES})
    if not completed or len(completed) == len(expected):
        raise ValueError('恢复源必须是有完整模型的未完成阶段')
    return completed, expected


def recover(source,config_path,csv_root,output):
    source,output = Path(source).resolve(),Path(output).resolve()
    if (not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT)
            or source.is_relative_to(output) or output.is_relative_to(source)):
        raise ValueError('源与新输出须在项目内且互不包含')
    if output.exists() or output.with_suffix('.recovery.json').exists():
        raise FileExistsError('恢复输出已存在，拒绝覆盖或静默重用')
    if (source/'completed.json').exists():
        raise ValueError('原阶段已经完成，无需恢复')
    manifest = read_json(source/'manifest.json')
    config = read_json(config_path)
    if config != manifest['config'] or digest(config_path) != manifest['config_sha256']:
        raise ValueError('恢复必须使用原冻结配置，不允许调整参数或删除组')
    check_training_sources(manifest)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != torch.device(manifest['device']).type:
        raise ValueError('恢复训练必须保持原设备类型')
    completed,expected = completed_models(source,config)
    interrupted=[]
    for key in sorted(expected-set(completed)):
        history_path=source/key[0]/f'seed{key[1]}'/key[2]/'history.json'
        interrupted.append(dict(dataset=key[0],seed=key[1],arm=key[2],
                                completed_epochs_before_failure=len(read_json(history_path)) if history_path.exists() else 0,
                                partial_history_sha256=digest(history_path) if history_path.exists() else None))
    evidence = dict(source=str(source.relative_to(ROOT)).replace('\\','/'),
                    source_manifest_sha256=digest(source/'manifest.json'),
                    source_code_commit=manifest['code_commit'],
                    recovery_tool_sha256=digest(Path(__file__)),test_evaluated=False,
                    interrupted_attempts=interrupted,
                    completed_copied=[dict(dataset=k[0],seed=k[1],arm=k[2],files=v['files']) for k,v in completed.items()],
                    restarted_from_initialization=[dict(dataset=k[0],seed=k[1],arm=k[2]) for k in sorted(expected-set(completed))],
                    policy='complete records copied byte-for-byte; incomplete attempts retained at source, not selected or resumed from best.pt')
    output.parent.mkdir(parents=True,exist_ok=True)
    train.save_json(output.with_suffix('.recovery.json'),evidence)
    original_train_arm = train.train_arm

    def copy_or_train(config,arm,seed,frame,graphs,tp,vp,base_state,destination,device,scale):
        key = destination.parents[1].name,seed,arm['name']
        if key not in completed:
            return original_train_arm(config,arm,seed,frame,graphs,tp,vp,base_state,destination,device,scale)
        original = source/key[0]/f'seed{seed}'/arm['name']
        pairing = read_json(source/key[0]/'pairs.json')
        if (pairing['train_pairs'] != tp or pairing['valid_pairs'] != vp
                or pairing['input_sha256'] != digest(Path(csv_root)/f'{key[0]}.csv')
                or not np.isclose(pairing['scale'],scale,rtol=1e-12,atol=1e-12)):
            raise ValueError('恢复配对、数据或Loss scale不同')
        saved = pd.read_csv(original/'validation_predictions.csv')
        if (saved['query'].tolist() != [p['query'] for p in vp]
                or saved['reference'].tolist() != [p['reference'] for p in vp]
                or not np.allclose(saved.y,frame.loc[saved['query'],'y'])
                or not np.allclose(saved.reference_y,frame.loc[saved['reference'],'y'])
                or not np.array_equal(saved.cliff_mol,frame.loc[saved['query'],'cliff_mol'])):
            raise ValueError('恢复验证记录行号或标签不符')
        encoder = {k:v for k,v in base_state.items() if k.startswith(('atom_encoder.','encoder.'))}
        if train.state_hash(encoder) != completed[key]['summary']['initialization']['encoder_sha256']:
            raise ValueError('恢复编码器初始化不同')
        destination.mkdir(parents=True,exist_ok=False)
        for name,sha in completed[key]['files'].items():
            if digest(original/name) != sha:
                raise ValueError('恢复期间原完成结果被修改')
            shutil.copyfile(original/name,destination/name)
            if digest(destination/name) != sha:
                raise ValueError('复制完成结果字节不一致')
        print(f'REUSED_COMPLETE {key[0]} seed={seed} arm={arm["name"]}',flush=True)
        return dict(completed[key]['summary'])

    try:
        train.train_arm = copy_or_train
        train.run(config_path,csv_root,output)
    finally:
        train.train_arm = original_train_arm
    train.save_json(output/'recovery.json',evidence)
    print(f'RECOVERY_COMPLETE copied={len(completed)} newly_trained={len(expected)-len(completed)}; test not evaluated',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--csv-root',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    recover(args.source,args.config,args.csv_root,args.output)
