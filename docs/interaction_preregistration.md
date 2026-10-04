# 首轮交互验证冻结记录

日期：2026-10-04。实现里程碑 bbff297 已通过 8 项契约测试及 16 组低预算 smoke。smoke 只验证管线，不据其指标选择 arm 或参数。本记录先于完整预算训练。

## 唯一问题

固定 GraphCliff 编码器、原 SAG/MaxMean 读出、train-only Top-1 参考及普通差值 MSE 后，跨分子原子交互是否超过全局表示差和相近新增参数的分子对 MLP？

## 固定矩阵

- 开发任务：CHEMBL234_Ki、CHEMBL244_Ki，已有历史探索，不能称未见任务。
- 训练种子：42、43、44；划分种子固定42。2×3×4=24 次训练；Top-1 参考标签为额外无训练对照。
- arm：direct、global、pair_mlp、cross。首轮全部原读出和普通 MSE，不加入 FPPool 或加权 Loss。
- 第一队列先执行 seed42 的8次，仅验证完整预算运行稳定性；不能凭单种子决定有效性或调参。稳定后补齐43/44的16次。
- 256 hidden、3 layers、4 attention heads、100epoch上限、patience15、batch32、AdamW 1e-4、weight_decay1e-5、betas0.9/0.95、warmup10、cosine到1e-6、clip1。参数沿用已记录本地预算候选，不称论文最优或精确官方复现。
- checkpoint 一律验证 Overall MSE 最低。训练对、参考池和验证查询固定，编码器及兼容 head 初值显式共享并记录哈希。
- 报告参数数、训练图前向数、optimizer steps、时间及峰值显存。direct 每次一步只处理一个分子，pair 两个；相同epoch不是等计算量。
- test 不构图、不评估、不用于决策。

## 预定判断

资源扩展 Go 门槛：每个任务上 cross 三 seed 平均 Cliff RMSE 相对 global 和 pair_mlp 均至少下降2%，Overall RMSE 相对各对照恶化不超过1%；每任务至少2/3 seed方向一致。同时完整报告 direct 与近邻标签对照，不因只超过弱对照就宣称机制有效。

这是后续资源分配的实用门槛，不是统计显著性或发表标准。跨任务仅两项，阳性仍须在预定 CHEMBL264_Ki、CHEMBL233_Ki 确认。完整矩阵不满足时保存阴性结论，先诊断或停止加模块；不靠调 test 或堆 FPPool/Loss 修饰结果。

候选整体目标继续保留三项模块：交互获得支持后，再冻结 attention/FPPool/dynamic 的2×2×2消融和静态权重对照；若交互阴性，三项方案需报告该限制，而非强行宣称成功。

## 可复现命令

```powershell
python -m graphcliff_pair.train --config configs/interaction_seed42.json --csv-root D:/GraphCliff-main/benchmark_data --output artifacts/interaction_seed42_20261004
python -m graphcliff_pair.train --config configs/interaction_seed43_44.json --csv-root D:/GraphCliff-main/benchmark_data --output artifacts/interaction_seed43_44_20261004
```

输出路径只作本机例子，拒绝覆盖已存在目录。原始预测、配对、权重留本地；公开提交摘要和核对结果。尚未完成的训练不得记为已完成里程碑。
