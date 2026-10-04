"""冻结验证后的单次最终评估；先完成全部无查询标签预测，再读取test标签计算指标。"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
from freeze_validation import validate_freeze
from graphcliff_pair.data import digest
from graphcliff_pair.predict import run as predict_smiles, training_rows_only
from graphcliff_pair.train import metrics
import numpy as np
import pandas as pd
from rdkit import Chem


def save(path, value):
    """同目录临时文件+原子替换，保留中断前完整状态。"""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze_identity(frozen):
    # 同一冻结文件的复制或JSON空白变化不会创建新的评估身份。
    payload = json.dumps(frozen, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@contextmanager
def evaluation_lock(identity):
    registry = ROOT / 'artifacts' / 'test_evaluation_registry'
    registry.mkdir(parents=True, exist_ok=True)
    lock_path = registry / f'{identity}.lock'
    with lock_path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('同一冻结评估正在运行，禁止并发重复评估') from exc
        try:
            yield registry / f'{identity}.json'
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def child_path(output, name):
    path = output / name
    if path.is_symlink() or path.resolve().parent != output or (path.exists() and not path.is_file()):
        raise ValueError('预测输出路径越界或不是普通文件')
    return path


def model_key(entry):
    parts = [str(entry['dataset']), str(entry['seed']), str(entry['arm'])]
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', part) for part in parts):
        raise ValueError('冻结模型名称包含非法路径字符')
    return f'{parts[0]}_seed{parts[1]}_{parts[2]}'


def check_prediction(path, queries):
    frame = pd.read_csv(path)
    required = {'query_index', 'smiles', 'prediction', 'reference_row'}
    if not required.issubset(frame) or len(frame) != len(queries):
        raise ValueError('测试预测字段/数量不匹配')
    if frame.query_index.tolist() != list(range(len(queries))) or frame.smiles.tolist() != queries.smiles.tolist():
        raise ValueError('测试预测行号/结构不匹配')
    if not np.isfinite(frame.prediction).all():
        raise ValueError('测试预测非有限')
    return frame


def predict_atomic(entry, csv_root, query_path, prediction_path, queries):
    # 未登记state的已完成/半写入文件不可信，重新预测后原子替换此单文件。
    temporary = prediction_path.parent / f'.{prediction_path.name}.{uuid.uuid4().hex}.pending'
    try:
        predict_smiles(ROOT / entry['run_dir'], Path(csv_root) / f"{entry['dataset']}.csv", query_path, temporary)
        check_prediction(temporary, queries)
        os.replace(temporary, prediction_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate(freeze_path, csv_root, output, resume_predictions=False):
    frozen = validate_freeze(freeze_path)
    identity = freeze_identity(frozen)
    output = Path(output).resolve()
    with evaluation_lock(identity) as registry_path:
        _evaluate_locked(freeze_path, frozen, csv_root, output, resume_predictions, identity, registry_path)


def _evaluate_locked(freeze_path, frozen, csv_root, output, resume_predictions, identity, registry_path):
    freeze_sha = digest(freeze_path)
    registration = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else None
    if registration is not None:
        if registration['freeze_identity'] != identity or Path(registration['output']).resolve() != output:
            raise ValueError('同一冻结身份已绑定其他输出目录，禁止换目录重复test')
        if registration['phase'] != 'predictions':
            raise ValueError('已进入标签评估阶段，禁止重复读取test标签')
        if registration['freeze_sha256'] != freeze_sha:
            raise ValueError('冻结文件字节身份改变，禁止恢复')
    if (output / 'completed.json').exists() or (output / 'test_metrics.json').exists():
        raise FileExistsError('test已评估，禁止重复运行或按test调整配置')
    if output.exists() and not resume_predictions:
        raise FileExistsError('输出已存在；仅可显式恢复未完成的预测阶段')
    if registration is None:
        registration = dict(freeze_identity=identity, freeze_sha256=freeze_sha, output=str(output), phase='predictions')
        save(registry_path, registration)
    output.mkdir(parents=True, exist_ok=True)
    state_path = child_path(output, 'prediction_state.json')
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'freeze_sha256': freeze_sha, 'predictions': {}, 'labels_loaded': False}
    if state['freeze_sha256'] != freeze_sha or state['labels_loaded']:
        if state.get('labels_loaded'):
            registration['phase'] = 'labels_loaded'
            save(registry_path, registration)
        raise ValueError('冻结身份改变或已进入标签评估阶段，禁止恢复')
    expected = {model_key(entry) for entry in frozen['models']}
    if len(expected) != 78 or not set(state['predictions']).issubset(expected):
        raise ValueError('预测状态没有匹配完整固定矩阵')
    save(state_path, state)
    rows, queries_by_dataset = {}, {}
    for dataset, sha in frozen['data_sha256'].items():
        source = Path(csv_root) / f'{dataset}.csv'
        if digest(source) != sha:
            raise ValueError('源数据哈希改变')
        # 此阶段不读取y/cliff_mol，仅核对结构、分区和canonical重复。
        structures = pd.read_csv(source, usecols=['smiles', 'split'])
        train_valid, test = set(), set()
        test_rows = structures.index[structures['split'] == 'test'].tolist()
        if not test_rows:
            raise ValueError('test分区为空')
        for i, row in structures.iterrows():
            mol = Chem.MolFromSmiles(row.smiles)
            if mol is None or not mol.GetNumAtoms():
                raise ValueError(f'test前结构检查失败: {dataset} row={i}')
            canonical = Chem.MolToSmiles(mol, canonical=True)
            (test if row['split'] == 'test' else train_valid).add(canonical)
        overlap = train_valid & test
        if overlap:
            raise ValueError(f'{dataset} test与开发集有{len(overlap)}个canonical重复；停止、不静默改划分')
        rows[dataset] = test_rows
        query_path = child_path(output, f'{dataset}_query_smiles.csv')
        queries = structures.loc[test_rows, ['smiles']].reset_index(drop=True)
        queries_by_dataset[dataset] = queries
        if query_path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(query_path), queries)
        else:
            temporary = query_path.with_name(f'.{query_path.name}.{uuid.uuid4().hex}.pending')
            try:
                queries.to_csv(temporary, index=False)
                os.replace(temporary, query_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
    for entry in frozen['models']:
        key = model_key(entry)
        prediction_path = child_path(output, f'{key}_predictions.csv')
        queries = queries_by_dataset[entry['dataset']]
        if key in state['predictions']:
            if not prediction_path.is_file() or digest(prediction_path) != state['predictions'][key]:
                raise ValueError('预测缓存身份不匹配')
            check_prediction(prediction_path, queries)
            continue
        predict_atomic(entry, csv_root, output / f"{entry['dataset']}_query_smiles.csv", prediction_path, queries)
        state['predictions'][key] = digest(prediction_path)
        save(state_path, state)
    if set(state['predictions']) != expected:
        raise ValueError('预测未覆盖完整固定矩阵')
    # 训练参考标签仅用于无训练Top-1对照；全部模型必须选中相同参考。
    reference_predictions = {}
    for entry in frozen['models']:
        dataset = entry['dataset']
        frame = check_prediction(output / f'{model_key(entry)}_predictions.csv', queries_by_dataset[dataset])
        references = frame[['query_index', 'reference_row']]
        if dataset not in reference_predictions:
            pairs_path = (ROOT / entry['run_dir']).parents[1] / 'pairs.json'
            train_rows = json.loads(pairs_path.read_text(encoding='utf-8'))['train_rows']
            bank = training_rows_only(Path(csv_root) / f'{dataset}.csv', train_rows)
            if not frame.reference_row.isin(bank.index).all():
                raise ValueError('test参考行不属于冻结训练参考池')
            values = bank.loc[frame.reference_row, 'y'].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError('训练参考标签非法')
            reference_predictions[dataset] = (references, values)
        else:
            pd.testing.assert_frame_equal(references, reference_predictions[dataset][0])
    # 注册表先进入不可恢复阶段：在state落盘前中断，也不能换输出目录重新读取标签。
    registration['phase'] = 'labels_loaded'
    save(registry_path, registration)
    state['labels_loaded'] = True
    save(state_path, state)
    labels = {d: pd.read_csv(Path(csv_root) / f'{d}.csv', usecols=['y', 'cliff_mol']).loc[ids].reset_index(drop=True)
              for d, ids in rows.items()}
    records, nearest_reference = [], []
    for dataset, truth in labels.items():
        if not np.isfinite(truth.y).all() or not truth.cliff_mol.isin([0, 1]).all():
            raise ValueError('测试标签非法')
        baseline = truth.assign(prediction=reference_predictions[dataset][1])
        nearest_reference.append(dict(dataset=dataset, **metrics(baseline.to_dict('records'))))
    for entry in frozen['models']:
        dataset, seed, arm = entry['dataset'], entry['seed'], entry['arm']
        key = model_key(entry)
        predictions = check_prediction(output / f'{key}_predictions.csv', queries_by_dataset[dataset])
        predictions['y'] = labels[dataset].y
        predictions['cliff_mol'] = labels[dataset].cliff_mol
        result = metrics(predictions.to_dict('records'))
        records.append(dict(dataset=dataset, seed=seed, arm=arm, **result, test_evaluated=True,
                            checkpoint_sha256=entry['checkpoint_sha256'], predictions_sha256=state['predictions'][key]))
    save(child_path(output, 'test_metrics.json'), {'records': records, 'nearest_reference': nearest_reference,
         'freeze_sha256': freeze_sha, 'test_evaluated': True, 'model_selection_on_test': False})
    save(child_path(output, 'completed.json'), {'runs': 78, 'test_evaluated': True, 'model_selection_on_test': False})
    registration['phase'] = 'completed'
    save(registry_path, registration)
    print('TEST_COMPLETE 78 fixed checkpoint evaluations; no test-based selection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--csv-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume-predictions', action='store_true')
    args = parser.parse_args()
    evaluate(args.freeze, args.csv_root, args.output, args.resume_predictions)
