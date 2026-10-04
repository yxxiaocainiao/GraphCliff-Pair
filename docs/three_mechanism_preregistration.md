# 三项机制的最小完整消融

日期：2026-10-04。先于本阶段任何完整预算训练。首轮 interaction_seed42 队列正在运行；不根据它的成绩选择 FPPool/Loss 参数。

## 目的与范围

最终目标要求回答 Cross-Attention、FPPool、动态加权 Loss 各自贡献。因此，当前两项开发任务上的最小完整消融属于必要验收，不用单种子交互成绩删掉某个机制。首轮交互 Go 门槛决定是否扩展到另外两项确认任务；阴性时仍交付当前两任务的固定消融和阴性结论，不追加新参数、新数据或新模块寻找涨点。

这澄清 interaction_preregistration.md 中“交互支持后再冻结/停止加模块”的资源顺序：停止的是额外探索和规模扩展，不能把终极目标缩成只测试注意力。本文件冻结其余两项候选的必要比较，训练公式和预算不变。

## 固定设计

两任务 CHEMBL234_Ki、CHEMBL244_Ki；种子42/43/44；保持同一 split_seed42、训练参考、预算、checkpoint 规则及初始化。

2×2×2 组合：

- 表示交互：global_diff / cross_attention。
- 读出：原 SAG+MaxMean / Morgan-only 官方 FPPool 适配。
- 目标：普通差值 MSE / 已冻结动态权重。

普通 MSE+原读出的 global/cross 已在交互队列中，不重复训练。余下6组×2任务×3seed=36次。另增加 FPPool 下的 global/cross 静态权重对照（12次）来区分动态日程与加权本身；增加 FPPool 下参数量匹配的 pair_mlp+MSE（6次）排除池化条件下的简单容量解释。此阶段共54次，三份每种子18次的独立配置。

alpha_max=1、warmup10、cap3；scale=max(median(abs(train_delta)),1e-6)。这些是候选设置，不是论文最优；不根据验证/测试逐组合调参。除三个因素及静态/容量对照外，不改编码器、参考、shuffle、优化器、选模、训练预算。

## 分析

各任务分别报告每组3seed的 Overall/Cliff/Non-cliff RMSE、MAE，以及有符号差值辅助指标。报告8个组合的全表和条件差值：同一读出/Loss下 attention 对 global，同一交互/Loss下 FPPool 对原读出，同一交互/读出下动态对普通 MSE；动态还对静态权重。不能把组合对基线的总差值归因给某一模块。

每项任务至少2/3seed方向一致和均值改善才描述“开发集方向较一致”；仅两任务不能推出广泛有效。训练任务×seed 是重复单位，分子对不是独立重复。不依据显著性检验筛选组合。

同一 checkpoint 报告全部指标；说明 Overall 与 Cliff 的权衡、峰值显存、时间、训练图前向与 optimizer steps。负贡献、任务间不一致、静态优于动态均保留，不能称“三板斧都有效”。

## 最终测试

完成验证及消融后冻结最终交付报告，不再改配置，再用已保存 checkpoint 一次评估 test。预定主要报告原 direct、global、pair_mlp、cross，以及全部8组合；不按 test 选方法。先核对 test 结构、canonical 重复、图构建和索引；问题原样报告，不静默改划分或重训。训练参考只来自 train，不包括 validation。

用户只授权新项目边界；所有原始结果留 artifacts，公开上传摘要、源码身份、配置与核对记录。原项目只读。
