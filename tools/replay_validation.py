"""重放已完成阶段的 best.pt；证明保存权重、选模 epoch 与验证预测一致。"""
import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from graphcliff_pair.data import digest, read_development
from graphcliff_pair.fingerprint import membership
from graphcliff_pair.model import PairRegressor, parameters
from graphcliff_pair.train import evaluate, metrics
from graphcliff_pair.vendor.model import GraphCliffRegressor
from graphcliff_pair.vendor.training import set_seed

# 同设备类型的固定数值容差，不提供运行时放宽开关。
# GPU训练结果不能用CPU重放作严格对应证明（matmul backend差异可超过此容差）。
TOLERANCES = dict(prediction_atol=1e-5, prediction_rtol=1e-6, metric_atol=1e-5)
CORE_SOURCES = {
    'graphcliff_pair/__init__.py', 'graphcliff_pair/data.py',
    'graphcliff_pair/external.py', 'graphcliff_pair/fingerprint.py',
    'graphcliff_pair/fppool.py', 'graphcliff_pair/loss.py',
    'graphcliff_pair/model.py', 'graphcliff_pair/train.py',
    'graphcliff_pair/vendor/__init__.py', 'graphcliff_pair/vendor/model.py',
    'graphcliff_pair/vendor/dataset_utils.py', 'graphcliff_pair/vendor/protocol.py',
    'graphcliff_pair/vendor/training.py',
}
RUN_FILES = {'checkpoint': 'best.pt', 'predictions': 'validation_predictions.csv',
             'history': 'history.json', 'summary': 'summary.json'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protected(name, root=ROOT):
    """将Windows manifest路径转为跨平台项目相对路径，并拒绝越界。"""
    name = str(name).replace('\\', '/')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('重放路径越出项目范围')
    return path


def relative(path, root=ROOT):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('重放输入与输出必须位于项目目录内')
    return path.relative_to(root.resolve()).as_posix()


def check_training_sources(manifest, root=ROOT):
    normalized = {}
    for name, expected in manifest.get('source_sha256', {}).items():
        name = name.replace('\\', '/')
        if name == 'graphcliff_pair/predict.py':
            continue  # 独立CLI从不被train/evaluate导入，已有审计明确允许修正。
        if name in normalized:
            raise ValueError('manifest源码路径归一化后重复')
        normalized[name] = expected
    current = {relative(p, root) for p in (root/'graphcliff_pair').rglob('*.py')
               if relative(p, root) != 'graphcliff_pair/predict.py'}
    if not CORE_SOURCES.issubset(normalized) or set(normalized) != current:
        raise ValueError('训练核心源码清单缺失或改变')
    for name, expected in normalized.items():
        # 对字节计算哈希，不能用换行归一化掩盖与实际训练manifest的不一致。
        if digest(protected(name, root)) != expected:
            raise ValueError(f'当前训练核心源码与manifest不符: {name}')
    return normalized


def run_identity(run_dir, root=ROOT):
    return dict(run_dir=relative(run_dir, root),
                manifest_sha256=digest(run_dir.parents[2]/'manifest.json'),
                pairs_sha256=digest(run_dir.parents[1]/'pairs.json'),
                **{name+'_sha256': digest(run_dir/file) for name, file in RUN_FILES.items()})


def check_checkpoint_metadata(checkpoint, summary, history):
    if set(checkpoint) != {'epoch', 'model_state_dict'} or not history:
        raise ValueError('checkpoint结构或训练history不完整')
    best = min(history, key=lambda h: h['valid_mse'])
    if checkpoint['epoch'] != best['epoch'] or checkpoint['epoch'] != summary['best_epoch']:
        raise ValueError('checkpoint epoch与最低验证Overall MSE不一致')
    if len(history) != summary['epochs_run'] or [h['epoch'] for h in history] != list(range(1, len(history)+1)):
        raise ValueError('训练epoch记录不连续或与summary不符')
    if not np.isfinite(best['valid_mse']) or abs(summary['overall_rmse']**2-best['valid_mse']) > TOLERANCES['metric_atol']:
        raise ValueError('最佳epoch指标与summary不一致')
    return best


def compare_predictions(replayed, recorded, summary, best):
    """逐行身份和标签一致，逐预测与整体指标均需通过固定容差。"""
    replayed = pd.DataFrame(replayed)
    if len(replayed) != len(recorded) or not len(replayed):
        raise ValueError('重放验证查询数不一致')
    for column in ['query', 'reference', 'cliff_mol']:
        if not np.array_equal(replayed[column], recorded[column]):
            raise ValueError(f'重放身份错位: {column}')
    for column in ['similarity', 'y', 'reference_y']:
        if not np.allclose(replayed[column], recorded[column], atol=1e-7, rtol=1e-7):
            raise ValueError(f'重放结构相似度或标签不一致: {column}')
    actual, expected = replayed.prediction.to_numpy(), recorded.prediction.to_numpy()
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('重放或记录的预测非有限')
    if not np.allclose(actual, expected, atol=TOLERANCES['prediction_atol'], rtol=TOLERANCES['prediction_rtol']):
        raise ValueError(f'checkpoint未恢复原验证预测: max_abs={np.max(np.abs(actual-expected)):.9g}')
    result = metrics(replayed.to_dict('records'))
    differences = []
    for name, value in result.items():
        target = summary[name]
        if value is None or target is None:
            if value != target:
                raise ValueError(f'重放指标支持量不一致: {name}')
        elif not np.isfinite(target) or abs(value-target) > TOLERANCES['metric_atol']:
            raise ValueError(f'重放指标与summary不一致: {name}')
        else:
            differences.append(abs(value-target))
    if abs(result['overall_rmse']**2-best['valid_mse']) > TOLERANCES['metric_atol']:
        raise ValueError('重放指标未恢复选模时的最低验证MSE')
    return dict(count=len(replayed), max_prediction_abs_error=float(np.max(np.abs(actual-expected))),
                max_metric_abs_error=float(max(differences)), metrics=result)


def replay_one(run_dir, manifest, frame, graphs, pairing, device, root=ROOT):
    """单个checkpoint内部接口；完整阶段CLI负责来源/配对审计和完整矩阵。"""
    run_dir = Path(run_dir)
    before = run_identity(run_dir, root)
    config = manifest['config']
    if device.type != torch.device(manifest['device']).type:
        raise ValueError('严格验证重放必须使用原训练设备类型；GPU训练不能用CPU替代证明')
    summary, history = read_json(run_dir/'summary.json'), read_json(run_dir/'history.json')
    seed = int(run_dir.parent.name.removeprefix('seed'))
    arm = next(a for a in config['arms'] if a['name'] == run_dir.name)
    if summary['arm'] != arm['name'] or summary['seed'] != seed or summary['test_evaluated']:
        raise ValueError('summary身份不符或已评估test')
    checkpoint = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=True)
    best = check_checkpoint_metadata(checkpoint, summary, history)
    set_seed(SimpleNamespace(seed=seed))
    if arm['variant'] == 'direct':
        model = GraphCliffRegressor(38, 13, hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=0)
    else:
        model = PairRegressor(arm['variant'], config['hidden_size'], config['num_layers'], config['heads'], arm['readout'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if parameters(model) != summary['parameters']:
        raise ValueError('重放模型参数量不一致')
    model.to(device)
    records = evaluate(model, arm, pairing['valid_pairs'], graphs, frame, config['batch_size'], device)
    comparison = compare_predictions(records, pd.read_csv(run_dir/'validation_predictions.csv'), summary, best)
    if before != run_identity(run_dir, root):
        raise ValueError('重放期间权重或记录被修改')
    return dict(dataset=run_dir.parents[1].name, seed=seed, arm=arm['name'], **before,
                epoch=checkpoint['epoch'], batch_size=config['batch_size'], device=str(device),
                training_device=manifest['device'], status='matched', **comparison)


def validate_replay(path, expected_models=None, root=ROOT):
    """冻结和test入口验证重放证据的范围以及每项原始文件身份。"""
    evidence = read_json(path)
    records = evidence.get('models', [])
    keys = [(m['dataset'], m['seed'], m['arm']) for m in records]
    if (evidence.get('status') != 'validation_replay_passed' or evidence.get('test_evaluated') is not False
            or evidence.get('runs') != len(records) or not records or len(set(keys)) != len(keys)
            or evidence.get('tolerances') != TOLERANCES):
        raise ValueError('验证重放证据不完整或容差改变')
    if expected_models is not None:
        expected = {(m['dataset'], m['seed'], m['arm']): m for m in expected_models}
        if len(expected) != len(expected_models) or set(keys) != set(expected):
            raise ValueError('重放证据没有覆盖冻结模型矩阵')
    else:
        expected = None
    checked = set()
    for record in records:
        run_dir = protected(record['run_dir'], root)
        identity = run_identity(run_dir, root)
        if any(record.get(k) != value for k, value in identity.items()):
            raise ValueError('重放后checkpoint、预测、history或运行身份改变')
        if (record.get('status') != 'matched' or record.get('count', 0) < 1
                or record.get('epoch', 0) < 1 or record.get('batch_size', 0) < 1
                or not np.isfinite(record.get('max_prediction_abs_error', float('nan')))
                or not np.isfinite(record.get('max_metric_abs_error', float('nan')))
                or record['max_prediction_abs_error'] < 0
                or not 0 <= record['max_metric_abs_error'] <= TOLERANCES['metric_atol']):
            raise ValueError('重放数值核对未通过')
        summary = read_json(run_dir/'summary.json')
        predictions = pd.read_csv(run_dir/'validation_predictions.csv')
        max_allowed = TOLERANCES['prediction_atol'] + TOLERANCES['prediction_rtol']*float(predictions.prediction.abs().max())
        if (record['count'] != len(predictions) or record['count'] != summary['count']
                or record['epoch'] != summary['best_epoch']
                or record['max_prediction_abs_error'] > max_allowed):
            raise ValueError('重放数值范围或选模元数据不一致')
        if expected is not None:
            model = expected[(record['dataset'], record['seed'], record['arm'])]
            if any(model.get(k) != value for k, value in identity.items() if k in model):
                raise ValueError('重放权重与冻结权重不一致')
        stage = run_dir.parents[2]
        manifest = read_json(stage/'manifest.json')
        if (torch.device(record.get('device','cpu')).type != torch.device(manifest['device']).type
                or record.get('training_device') != manifest['device']):
            raise ValueError('重放设备与原训练设备类型不一致')
        if stage not in checked:
            sources = check_training_sources(manifest, root)
            if sources != evidence.get('training_source_sha256'):
                raise ValueError('重放证据的训练源码身份不一致')
            checked.add(stage)
    for name, expected_hash in evidence.get('replay_source_sha256', {}).items():
        if digest(protected(name, root)) != expected_hash:
            raise ValueError(f'重放工具来源改变: {name}')
    if 'tools/replay_validation.py' not in evidence.get('replay_source_sha256', {}):
        raise ValueError('缺少重放工具来源身份')
    return evidence


def replay(folders, csv_root, output, device=None):
    output = Path(output)
    relative(output)
    if output.exists():
        raise FileExistsError('重放证据已存在，拒绝覆盖')
    torch.set_num_threads(2)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cache, seen, records, sources, data_hashes = {}, set(), [], None, {}
    for folder in map(Path, folders):
        manifest = read_json(folder/'manifest.json')
        current = check_training_sources(manifest)
        if sources is not None and current != sources:
            raise ValueError('各阶段训练来源不一致')
        sources = current
        config = manifest['config']
        if device.type != torch.device(manifest['device']).type:
            raise ValueError('严格验证重放必须使用原训练设备类型；GPU训练不能用CPU替代证明')
        expected = set(itertools.product(config['datasets'], config['seeds'], [a['name'] for a in config['arms']]))
        complete, summary = read_json(folder/'completed.json'), read_json(folder/'summary.json')
        if (complete['runs'] != len(expected) or complete['test_evaluated'] or manifest['test_evaluated']
                or len(summary) != len(expected)
                or {(r['dataset'], r['seed'], r['arm']) for r in summary} != expected):
            raise ValueError('只接受完成整个预定阶段的重放')
        for dataset in config['datasets']:
            source = Path(csv_root)/f'{dataset}.csv'
            pairing = read_json(folder/dataset/'pairs.json')
            data_hashes[dataset] = digest(source)
            if pairing['input_sha256'] != data_hashes[dataset]:
                raise ValueError('重放数据身份不一致')
            cache_key = (dataset, config['split_seed'])
            if cache_key not in cache:
                cache[cache_key] = read_development(source, config['split_seed'])
            frame, graphs, train_rows, valid_rows, tp, vp = cache[cache_key]
            tp, vp = tp[:config.get('limit_train_queries', len(tp))], vp[:config.get('limit_valid_queries', len(vp))]
            if (pairing['train_rows'] != train_rows.tolist() or pairing['valid_rows'] != valid_rows.tolist()
                    or pairing['train_pairs'] != tp or pairing['valid_pairs'] != vp):
                raise ValueError('重放配对或划分不一致')
            if any(a['readout'] == 'fppool' for a in config['arms']):
                for row in {p[k] for p in vp for k in ['query', 'reference']}:
                    graph = graphs[row]
                    if getattr(graph, 'atom_fp', None) is None:
                        graph.atom_fp = membership(SimpleNamespace(smiles=frame.at[row, 'smiles'], x=graph.x,
                                                                  edge_index=graph.edge_index, edge_attr=graph.edge_attr))
            for seed, arm in itertools.product(config['seeds'], config['arms']):
                key = (dataset, seed, arm['name'])
                if key in seen:
                    raise ValueError('重放矩阵包含重复模型')
                seen.add(key)
                run_dir = folder/dataset/f'seed{seed}'/arm['name']
                record = replay_one(run_dir, manifest, frame, graphs, pairing, device)
                records.append(record)
                print(f"REPLAY_MATCH {dataset} seed={seed} arm={arm['name']} count={record['count']} max_abs={record['max_prediction_abs_error']:.9g}", flush=True)
    result = dict(status='validation_replay_passed', runs=len(records), test_evaluated=False,
                  tolerances=TOLERANCES, training_source_sha256=sources, data_sha256=data_hashes,
                  replay_source_sha256={'tools/replay_validation.py': digest(Path(__file__))},
                  runtime=dict(torch=torch.__version__, pyg=__import__('torch_geometric').__version__,
                               rdkit=__import__('rdkit').__version__, device=str(device)), models=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    validate_replay(output)
    print(f'REPLAY_OK {len(records)} saved checkpoints; test not evaluated', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--csv-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    replay(args.runs, args.csv_root, args.output, args.device)
