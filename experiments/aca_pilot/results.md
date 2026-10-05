# GraphCliff＋ACA-MSE：三种子退化，提前暂停

结论：CHEMBL234_Ki三种子Cliff均值恶化3.01%，Overall均值恶化4.55%；仅1/3种子Cliff改善，触发已固定的任务退化否决条件。按用户“成果不好立即暂停”要求停止。

这次正式完成并审计6/18组：一个任务×三个种子×两个Loss。剩余12组未完成，其中下一任务244/seed42/mse已启动后中断、保存16轮现场，其余11组未启动。不能将其称为三任务完整实验，也不能推断另外两个任务的效果。所有已有产物保留，没有创建completed.json。

## 完整六组结果

| Seed | MSE Cliff RMSE | ACA Cliff RMSE | Cliff变化 | MSE Overall | ACA Overall |
|---|---:|---:|---:|---:|---:|
| 42 | 0.632783 | 0.674997 | +6.67% | 0.580060 | 0.625023 |
| 43 | 0.644257 | 0.665636 | +3.32% | 0.589142 | 0.629467 |
| 44 | 0.640981 | 0.635186 | -0.90% | 0.603735 | 0.599035 |

Cliff均值±样本标准差：MSE 0.639340±0.005910；ACA 0.658606±0.020816。Overall均值分别为0.590979与0.617842。292个验证分子，128个cliff分子。

## 为什么停止

方案固定的条件包括：任一任务Cliff均值不能恶化超过3%，Overall均值不能恶化超过1%。第一个完整三种子任务已经同时违反，剩余任务不可能使整个固定方案满足这两条条件。停止使用已有否决阈值；没有改阈值、调系数或换一个有利任务。

执行时点确有调整：原方案写的是完整18组后计算效果门槛；根据用户优先止损指令，将判定提前到第一个完整三种子任务。该提前停止是执行决定，不能将未运行的任务填成负结果。Cliff均值只略高于3%界限，Overall退化则明显超过1%；本次决定不等于显著性证明或ACA普遍无效。

## 核对

- 42项单元测试通过；计划三个任务六组两轮GPU冒烟及重放通过。另有两组被排除2835预检，保留但不作为性能证据。
- 原GraphCliff初始化/推断保持不变；两组参数均为6,021,198，三个seed的匹配初始化一致。alpha0与MSE的值和梯度一致；alpha0.1加入官方表征项。
- 保存预测逐项重算；六个最佳检查点在训练GPU实际重放，最大绝对预测差8.881784197001252e-16。检查点按最低验证Overall MSE选择，未按Cliff挑选。
- 训练目标由每轮reg＋alpha×tsm独立重构；三种子各匹配组共同轮数上的候选triplet与batch计数完全一致，源文件/数据/校准身份核对通过。triplet总数是重复训练中的遭遇计数，不是独立样本数。
- 保存原始18组manifest和完整矩阵auditor，不降低其完整性要求；另用early_audit.py明确核对已完成任务，绑定审计程序自身SHA。
- CSV仅解析开发标签；浓度校准只解析划分后的训练行。没有解析测试活性或cliff标签，没有测试推断。

## 解释限制与下一步

该结果仅针对当前GraphCliff、官方alpha0.1、squared=True、p2、label-only ACA迁移；不是ACANet原论文MAE设置的精确复现，不能证明ACA在所有骨干或系数下无效。squared=True也会平方embedding距离；当前试验没有分离距离形式或梯度尺度的贡献。现在停止这次迁移，不自动转入系数搜索、FPPool或30任务。

alpha0对照也计算零权重triplet项以记录统计，因此运行时间不是优化后的纯MSE训练成本；两组推断容量和接口相同。原项目GPL/许可界限见README；官方Loss复用不能当作原创贡献。

这些局部负结果不足以支撑“30任务多数涨点”或中科院二区投稿主张。当前交付的是可复用实现、正确标签单位和经过审计的停止决定，不是完成整套发表目标。

## 输入、输出与改动

输入：D:/GraphCliff-main/benchmark_data。代码和聚合输出：D:/GraphCliff-Pair/experiments/aca_pilot。原始产物：D:/GraphCliff-Pair/artifacts/aca_pilot_20261005（Git忽略，不发布权重/逐分子预测）。

新增ACA独立目录及tests/test_aca_pilot.py，更新根README/task_plan/notes和文献建议；针对性更正long_branch_swap的三个文档中的单位。原GraphCliff-main、graphcliff_pair核心及vendor、旧绑定report.py、旧数据和队列均未修改。

正式训练源码提交：`ccc570bf3cb23e356150774132bbe2c9085b0a09`；启动时工作树干净：`True`。实现和固定方案已公开提交ccc570b。

仅复算已完成阶段：

```powershell
python -m experiments.aca_pilot.early_audit --output artifacts/aca_pilot_20261005 --dataset CHEMBL234_Ki
```

该命令不恢复训练、不补齐剩余任务、不执行测试集。
