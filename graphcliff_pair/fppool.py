"""GraphCliff的可选Morgan读出；不负责生成指纹或修改数据。"""
import torch
from torch import nn
from .external import load_fppool
FingerprintPool = load_fppool()


class FPPoolReadout(nn.Module):
    """输入[N,H]及节点归属[N,1024]，输出[B,2H]。"""
    def __init__(self, hidden_size):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError('hidden_size必须为正整数')
        self.hidden_size = hidden_size
        # 官方内部添加全原子分支；外部仍只传1024位Morgan归属。
        self.pool = FingerprintPool(hidden_size, hidden_size,
                                    torch.tensor([1024]), atoms_repr=True)
        self.projection = nn.Linear(hidden_size, 2 * hidden_size)

    def forward(self, x, batch, atom_fp):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise ValueError(f'x必须为[N,{self.hidden_size}]')
        if x.shape[0] == 0 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('x必须是非空且有限的浮点张量')
        if not isinstance(batch, torch.Tensor) or batch.shape != (x.shape[0],) or batch.dtype != torch.long:
            raise ValueError('batch必须为[N]的torch.long张量')
        if not isinstance(atom_fp, torch.Tensor) or atom_fp.shape != (x.shape[0], 1024):
            raise ValueError('atom_fp必须为[N,1024]，与原始图节点顺序一致')
        if batch.device != x.device or atom_fp.device != x.device:
            raise ValueError('x、batch与atom_fp必须在同一设备')
        if atom_fp.is_complex() or not torch.all((atom_fp == 0) | (atom_fp == 1)):
            raise ValueError('atom_fp必须是有限的二值归属矩阵')
        ids = torch.unique_consecutive(batch)
        if not torch.equal(ids, torch.arange(ids.numel(), device=batch.device)):
            raise ValueError('batch必须排序且编号从0连续递增，每个图非空')
        # 官方-1e12遮罩不适合半精度；明确拒绝而非返回NaN。
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError('官方FPPool适配器仅支持float32/float64')
        if x.device != self.projection.weight.device or x.dtype != self.projection.weight.dtype:
            raise ValueError('x的device/dtype必须与模块参数一致')
        # 禁用本模块自动混合精度，保证官方遮罩数值稳定。
        with torch.autocast(device_type=x.device.type, enabled=False):
            # 按分子移除完全未激活的位；官方注意力中这些位权重为零。
            # 每次仍调用官方模块，避免[B,1025,Nmax,H]的巨大中间张量。
            # 空Morgan分支保留一个零列，使其零向量语义与官方一致。
            outputs = []
            original_length = self.pool.fp_length
            try:
                for graph_id in range(ids.numel()):
                    nodes = batch == graph_id
                    local_fp = atom_fp[nodes]
                    active = local_fp.bool().any(dim=0)
                    compact_fp = local_fp[:, active] if active.any() else local_fp[:, :1]
                    self.pool.fp_length = torch.tensor([compact_fp.shape[1]], device=x.device)
                    value, _, _ = self.pool(x[nodes], torch.zeros_like(batch[nodes]), compact_fp)
                    outputs.append(value)
            finally:
                self.pool.fp_length = original_length
            pooled = torch.cat(outputs, dim=0)
            if pooled.shape != (ids.numel(), self.hidden_size) or not torch.isfinite(pooled).all():
                raise RuntimeError('FPPool输出尺寸错误或包含非有限值')
            result = self.projection(pooled)
        if result.shape != (ids.numel(), 2 * self.hidden_size) or not torch.isfinite(result).all():
            raise RuntimeError('投影输出尺寸错误或包含非有限值')
        return result
