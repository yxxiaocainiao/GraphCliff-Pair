# 第485批次捕获完成，前向分叉算子尚未定位

2026-10-08。本轮成功捕获历史分叉前的模型、图输入及CPU/CUDA/Python/NumPy随机状态，并建立直接重放入口。但所有捕获和重放只得到历史repeat2的预测，没有重新出现repeat1分支；因此**未定位具体算子，也未解决M52的长程重复失败**。不因观测一致而宣布修复。

## 执行与证据

沿用234/self、seed42、原生amax与strict deterministic，复用M52固定worker、MSE、AdamW和100epoch调度周期。诊断仅走6epoch以到达全局batch485，不选验证最优权重，不计算收益指标。原模型、Dropout和矩阵计算精度均未更改。

|观测方式|独立诊断|实际范围|第485批次结果|
|---|---|---|---|
|逐模块即时CPU哈希，前向前保存快照|a、b|各6epoch/498步|均匹配历史repeat2|
|前向中保留张量引用，结束后哈希|c、d|各6epoch/498步|均匹配历史repeat2|
|快照CPU复制与哈希均推迟到前向结束|e、f|各6epoch/498步|均匹配历史repeat2|

六次前484个batch及此前epoch事件，与M52两条历史轨迹的前495行完全一致。捕获模型权重SHA均为`cc9c9bb3a426abe3c7e3fde8a52bfefe1e9a22c0162845d7f025f1edbec6c5a5`，即历史step484更新后的状态。捕获的x、edge_index、edge_attr、batch逐项与原位模块输入匹配；每个目标前向记录143个nn.Module调用的张量值、布局、关键字参数、训练模式和RNG。

六份捕获文件逐字节相同，SHA256均为`f47c54b26281e471e14680a869c595d850ba6aa4b5fe8184b04be67aea6ede28`。这说明捕获状态可核验，不说明已捕获底层执行上下文的全部状态。

主要重放核验：3次独立同步观测前向、2次独立延迟观测前向，143个模块调用输出均与捕获一致；固定状态下另做50次无模块hook前向，预测仅1种，仍是历史repeat2。早期探索还做过2次延迟重放及50次无hook重放，结果相同；这些辅助记录保留，聚合报告以最后固定代码下的主要重放为准。

本轮累计6×498=2988个诊断优化步骤，无新增正式收益筛查。验证仅沿用固定forward重放以保留训练/验证模式切换，官方test评估0。M52未启动的6次验收仍未运行。

## 能与不能得出的结论

- 可以直接重放真实分叉前状态，后续检查该状态不必每次重新训练到epoch6。
- 本轮六次捕获的前缀、权重、输入和RNG身份成立，观测不是在另一个初始化上做的构造试验。
- 同步、延迟观测和无hook重放均只得到一个历史分支，**尚不能证明观测改变了分叉，也不能排除执行上下文或采样范围的影响**。
- 未观察到模块间输出差异，因此不能指定某个attention、Dropout、SAG选择或GPU内核为原因，不据此修改精度或关闭Dropout。
- 143是每次目标前向的模块调用记录数，不是全部底层functional/CUDA算子的覆盖数；保留张量引用也不是完全无扰动的观测。
- M51并列max梯度修正的局部证据仍保留，M52长程失败及原机制No-Go均保留。本轮没有新的有效算法或性能结论。

## 文件与复核

执行前依次提交`50d756d`（同步捕获）、`b4ad762`（延迟哈希）、`25087d6`（延迟快照CPU复制），各阶段运行时冻结对应来源，不修改M52的工具/配置。报告核验六段轨迹完整覆盖、捕获SHA、历史前缀与预测身份，累计156次来源哈希检查（重复检查同一共享来源，不是156个不同文件）。

输入是只读`D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv`官方train开发拆分。原始输出均在独立worktree的`artifacts/mechanism_retry_forward_20261008_*`，包含快照、模块观测、训练轨迹、日志和进程元数据；权重、图特征和逐样本预测不推送GitHub。可用的主捕获为`artifacts/mechanism_retry_forward_20261008_a/step485.pt`。

本方向新增[执行范围](repro_forward_screen.json)、3个薄观测工具、核验脚本与[聚合JSON](layer_forward_results_20261008.json)，更新README及worktree日志。原模型/训练器/Loss/历史配置/历史结果未改，可靠性目录和计划未写入，不合并。

已有证据复核，不训练：

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe tools/report_layer_forward.py
```

一次延迟观测重放，无优化步骤（输出目录必须是新目录）：

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe tools/trace_layer_forward_deferred.py --replay artifacts/mechanism_retry_forward_20261008_a/step485.pt --output artifacts/new_forward_replay
```

下一步应围绕同一捕获状态检查未插桩执行上下文及底层算子，先重新暴露历史的另一分支，再做归因；本轮停止追加同类warmup。不自动恢复效果矩阵，不用新模块掩盖复现问题。
