"""用冻结训练配对重建报告诊断，不修改模型或读取测试标签。"""
import numpy as np
import torch

from graphcliff_pair.loss import DeltaLoss
from graphcliff_pair.train import metrics


def nearest_reference(frame, pairs):
    records = [dict(**p, y=float(frame.at[p['query'], 'y']),
                    prediction=float(frame.at[p['reference'], 'y']),
                    cliff_mol=int(frame.at[p['query'], 'cliff_mol'])) for p in pairs]
    return metrics(records)


def check_metrics(actual, recorded, label):
    for name, value in actual.items():
        other = recorded.get(name)
        if (value is None and other is not None) or (value is not None and
                (other is None or not np.isclose(value, other, rtol=1e-9, atol=1e-9))):
            raise AssertionError(f'{label} 指标不一致: {name}')


def reconstruct_weights(frame, pairs, config, mode, seed, scale, epochs):
    # 与batches一致：先各自转float32，再相减；独立Generator只用于shuffle。
    yq = torch.tensor([frame.at[p['query'], 'y'] for p in pairs], dtype=torch.float32)
    yr = torch.tensor([frame.at[p['reference'], 'y'] for p in pairs], dtype=torch.float32)
    target = (yq - yr).reshape(-1, 1)
    loss = DeltaLoss(mode, scale=scale, alpha_max=config['alpha_max'],
                     warmup_epochs=config['warmup_epochs'], cap=config['weight_cap'])
    generator = torch.Generator().manual_seed(seed)
    results = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(pairs), generator=generator)
        chunks = []
        for start in range(0, len(pairs), config['batch_size']):
            batch = target[order[start:start + config['batch_size']]]
            _, weights = loss(torch.zeros_like(batch), batch, epoch)
            chunks.append(weights.flatten().numpy())
        values = np.concatenate(chunks)
        quantiles = np.quantile(values, [0, .25, .5, .75, .95, 1])
        results.append(dict(epoch=epoch, count=len(values), mean=float(values.mean()),
                            **dict(zip(['min', 'q25', 'median', 'q75', 'q95', 'max'], map(float, quantiles)))))
    return results


def check_weight_history(distributions, history):
    if len(distributions) != len(history) or not history:
        raise AssertionError('权重分布与实际训练轮数不一致')
    for distribution, row in zip(distributions, history):
        if distribution['epoch'] != row['epoch']:
            raise AssertionError('权重分布epoch不一致')
        for name in ['min', 'max']:
            # 原训练可能在CUDA；只允许float32归约量级的舍入差异。
            if not np.isclose(distribution[name], row[f'weight_{name}'], rtol=2e-6, atol=2e-6):
                raise AssertionError(f'权重分布与训练history不一致: epoch={row["epoch"]} {name}')


def nearest_markdown(baselines):
    lines = ['', '## 无训练的最近邻参考标签对照', '',
             '按固定结构Top-1直接预测训练参考活性。同一划分下各训练seed相同，每任务只报告一次，不计算三种子SD。', '',
             '|任务|Overall RMSE|Cliff RMSE|Non-cliff RMSE|MAE|查询数|',
             '|---|---:|---:|---:|---:|---:|']
    for row in baselines:
        values = ['NA' if row[name] is None else f'{row[name]:.4f}'
                  for name in ['overall_rmse', 'cliff_rmse', 'noncliff_rmse', 'mae']]
        lines.append(f'|{row["dataset"]}|' + '|'.join(values) + f'|{row["count"]}|')
    return lines
