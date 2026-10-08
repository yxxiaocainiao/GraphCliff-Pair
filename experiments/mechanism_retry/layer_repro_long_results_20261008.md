# 二十epoch重复性验收：未通过，后续队列停止

2026-10-08。结论：原生amax修正的长程重复性仍未通过。234/self两次独立运行在第6epoch、全局第485个训练batch首次出现前向预测差异；该批次记录的输入和前后RNG相同，此前权重与全部追踪记录一致。具体前向算子尚未定位，不直接归因某个GPU内核、Dropout或Cross-Attention。

## 预定范围与实际执行

按[执行配置](repro_long_screen.json)，先提交`4c54fbd956a7907b68f447da02f0919257b0275e`，启动时Git干净。计划234/244 × self/centered × 两次独立进程，每次固定20epoch、seed42、原生amax、严格deterministic。每对运行完成即比较，发现差异便停止后续队列。不是seed43/44效果扩训，不新增模块，不选验证最优权重。

|项目|实际执行|
|---|---|
|234/self|两次均完成20epoch、每次1660个优化步骤|
|234/centered|未运行|
|244/self|未运行|
|244/centered|未运行|
|完成/计划|2/8次；后续6次停止|
|逐批与逐epoch记录|3400条，总计3320个优化步骤|
|验证预测哈希覆盖|11680次查询预测，仅哈希重放，不计算收益指标|
|worker累计计时|427.63秒，约7.13分钟，包含读取、哈希和验证重放|
|峰值CUDA已分配显存|两次均565.65 MiB|
|官方test评估|0|

不能把计划中的第二任务写成已验证，也不能把同seed重复当作两个独立seed的统计证据。

## 首个差异

使用实际模型、配对、batch生成、plain-MSE路径、AdamW与WarmupCosineScheduler。调度器仍采用原100epoch周期，仅将诊断长度固定为20epoch，不早停。每个epoch运行现有验证forward以覆盖训练/验证模式切换，但不计算验证误差、不选模型、不保存最佳checkpoint。

启动子进程时固定`PYTHONHASHSEED=42`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，沿用两CPU线程及原矩阵精度`high`，每次训练前重置seed，再启用`torch.use_deterministic_algorithms(True, warn_only=False)`。本轮严格模式没有报错，但仍观察到差异。

|检查点|两次记录|
|---|---|
|完整初始化|相同，SHA为`30d154fb519f3750e5c24dfcd8efea743b75da78573c7e4eeb4d1c96b113a098`|
|第1–484个batch及此前epoch事件|逐位记录一致，包括前5次验证预测哈希和优化器状态|
|第485batch输入、样本配对、CPU/CUDA/Python/NumPy RNG|一致|
|第485batch forward后RNG|一致|
|第485batch预测|哈希不同|
|该batch训练Loss|0.34009838104248047 / 0.3401349186897278|
|该batch反向梯度、范数、更新权重|不同，出现于前向差异之后|

第一处不同是trace第496行，epoch6内部第70个batch。前一batch更新权重SHA为`cc9c9bb3a426abe3c7e3fde8a52bfefe1e9a22c0162845d7f025f1edbec6c5a5`，两次相同。数值相近不能替代逐位重复验收；当前没有证据证明这种差异对性能无影响或解释了所有历史差距。

本次发生于self对照，不能把它直接归因centered差分算子。相同可记录状态仍不代表所有底层执行状态一致；下一步需要捕获第485batch并观察模块输入/输出，确定首个出现差异的前向算子。

## 与上轮结论的关系

[M51首epoch诊断](layer_repro_results_20261008.md)发现的精确并列max梯度不确定性及其局部修正仍成立。那次首差异是第54batch反向梯度；本次原生amax路径的首差异是更长轨迹中的前向预测。不能把后一结果当作max修正无效，也不能把前一局部修正扩大成完整训练已确定性复现。

当前停止收益对照扩展。保留原FPPool、centered、context各自No-Go，不改门槛，不调Loss、epsilon或组合，不将self改选为有效最终方法。第二任务验收尚缺，不自动补跑剩余6次。

## 文件与复核

- 输入：只读`D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv`官方train开发拆分；官方test标签不解析。
- 原始输出：独立worktree的`artifacts/mechanism_retry_repro_long_20261008`及同名前缀stdout/stderr/process.json。stderr为空，进程已退出。
- 聚合证据：[结果JSON](layer_repro_long_results_20261008.json)。报告工具核对24份冻结来源、两份轨迹SHA、每次20epoch/1660batch完整覆盖、首差异与失败即停范围。
- 修改范围：本方向的执行配置、长轨迹工具/比较检查、结果报告、README与worktree日志；原模型/训练器/Loss/历史配置/历史报告未改，可靠性目录和计划未写入，不合并。

复核已有结果，不训练：

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe -m unittest discover -s tests -p test_retry_repro_long.py -v
& D:\Tools\conda-envs\graphcliff\python.exe tools/report_layer_repro_long.py
```

这里只证明当前执行范围未通过重复性验收，没有新的FPPool或LongPoly效果结论。下一步先定位第485batch的前向差异，再决定是否恢复尚未运行的验收项目。
