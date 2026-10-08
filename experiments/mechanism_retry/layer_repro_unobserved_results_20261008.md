# M55：原worker两次20epoch一致，历史分叉未重新出现

2026-10-08。直接执行原M52 worker两次，234/self、seed42各20epoch，两份1700事件的完整轨迹逐字节一致，且都与历史repeat2一致。**这是本次有限范围的重复一致证据，不是根因修复；历史repeat1仍未复现，M52失败记录保留。**

## 实际执行

执行前计划提交`e2d2b8e`。直接调用`tools/check_layer_repro_long.py --worker`，不增加训练适配器、模块hook、捕获快照或模型改动。原worker仍逐batch执行输入/预测/梯度/权重/RNG哈希，故这里“未插桩”仅指未增加模块观测，不是完全无诊断或无CPU同步。

固定原M52的20epoch配置、100epoch调度、native amax、strict deterministic、high精度、MSE及seed42。两个进程顺序执行，仅234/self；未启动M52其余6次，也未恢复旧收益队列。

|本次运行|epoch / optimizer步骤|轨迹事件|完整匹配|worker耗时|
|---|---:|---:|---|---:|
|new1|20 / 1660|1700|历史repeat2|213.87秒|
|new2|20 / 1660|1700|历史repeat2|213.99秒|

共同完整trace SHA256：`2d921dac4aa235171ada965a4e4cceb0765219a3b644911947c7d5b07292f60b`。两次全部训练batch、epoch排序/生成器状态、验证哈希、优化器状态哈希及调度事件一致，不是只比较最终loss。

两次与历史repeat1的首差异仍在事件496、epoch6、全局batch485：prediction、loss、gradient、gradient_norm、weight不同。该批次本次loss为0.3401349186897278，历史repeat1为0.34009838104248047。这里的loss是分叉身份辅助证据，不作性能排序。

共新增3320个诊断optimizer步骤，3400条轨迹事件，11680条验证查询预测参与哈希，worker合计427.86秒（约7.13分钟，包含数据读取/哈希/验证开销）。不计算效果指标、不选择最优checkpoint、官方test评估0。

## 核验与解释

- 四条新旧轨迹的summary/trace SHA、初始化/数据/来源身份核验通过，24个来源文件各核验4次，共96次哈希检查。GPU、Torch/CUDA/PyG/RDKit与M52 manifest一致。
- 核对20个epoch起止、1660个连续batch编号、完整事件覆盖；复用既有compare完成新旧轨迹六组比较。
- 两个新进程均退出、stderr空，峰值CUDA分配均565.65 MiB。核验脚本可重复运行，不训练。
- 现在已有三条完整20epoch轨迹匹配历史repeat2，但这些运行只覆盖同一机器/任务/算子/seed，不能据此估计分叉概率或保证更长训练。
- 无需额外模块hook也可得到repeat2，不能再把M53单分支现象简单归因于额外hook。仍不能排除原worker哈希同步或未记录执行上下文的影响。
- M51并列max梯度修正的局部证据、M52失败及机制No-Go均保留。未修改FPPool/LongPoly候选，不声明新算法有效，也不声明相关模块普遍无效。

## 文件与下一步

输入为只读`D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv`官方train开发拆分。输出位于`D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_repro_unobserved_20261008/`及同前缀日志/进程记录。原始逐批次轨迹留本地。

本轮新增本报告、[聚合JSON](layer_repro_unobserved_results_20261008.json)、`tools/report_layer_repro_unobserved.py`，更新本方向README/worktree，共5个唯一文件。原模型/训练器/配置/历史结果不改，可靠性目录/计划未写，不合并。

复核命令（只读原始轨迹，会重写本轮聚合JSON）：

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe tools/report_layer_repro_unobserved.py
```

停止追加同配置234/self重复。若继续根因检查，下一项应是对现有第485批次捕获中的排序/选择操作做并列与近并列检查；这只是诊断假设，不先认定SAG或某个算子有错。没有可复现触发器前，不据猜测改架构/精度/Dropout，不放宽原收益门槛。
