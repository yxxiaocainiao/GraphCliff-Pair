# LongPoly 训练分叉：首个真实批次差异与最小修正

2026-10-08。结论：在CHEMBL234、centered、seed42的首epoch追踪中，定位到原读出GPU max pooling在精确并列最大值处的梯度不确定性。使用原生`scatter_reduce_(amax)`替换该执行路径后，两次独立进程的83个batch全部逐位一致。这里修复训练协议，未证明机制有效，也未覆盖完整训练或其他任务的全部分叉原因。

## 证据

输入为只读`D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv`的官方train开发拆分，沿用H256、3层、4头、batch32、seed42、AdamW及plain MSE。复用实际读取、配对、初始化、批处理与Loss代码；仅追踪首epoch，不执行验证选模或官方test。完整初始化SHA仍为`30d154fb519f3750e5c24dfcd8efea743b75da78573c7e4eeb4d1c96b113a098`。

|检查|观测|
|---|---|
|strict / warn各两个独立进程，前8batch|输入、RNG、预测、梯度、权重完全一致|
|原路径，两个独立进程，完整83batch|前53batch一致；第54batch预测、Loss、RNG仍一致，但梯度及更新权重开始不同|
|第54batch模块输入/输出观测|所有记录的前向张量一致，head输入/输出梯度一致；SAG输出节点梯度不同|
|捕获第54batch，恢复实际精度和RNG|预测SHA与真实追踪严格一致后，才接受离线诊断|
|query / reference max输入|各有8个精确并列通道，且上游梯度非零|
|固定输入和上游梯度，原路径各重放100次|两侧各出现23种梯度SHA；前向值始终一致|
|同输入，原生amax各重放100次|两侧梯度各仅1种SHA，且每次前向与原路径逐位一致|
|原生amax原型，两个进程各83batch|全部追踪记录一致|
|实际修正函数，两个进程各83batch|全部追踪记录一致，当前5份诊断相关源码SHA复核通过|

以上次数与SHA由[可重算JSON](layer_repro_results_20261008.json)和`tools/report_layer_repro.py`核验。固定输入重复不是独立分子样本，不作统计显著性计算。

本轮共18段短程诊断、991个optimizer步骤，每段不超过首epoch；另有不更新权重的反向重放。没有新的完整效果筛查fit，没有验证选模，没有官方test评估。不得把诊断步骤描述为零训练或性能实验。

## 修正的数学含义

对一个图、一个通道，设$m=\max_i x_i$、$I=\{i:x_i=m\}$。唯一最大值时两条路径的导数相同。精确并列时，max存在多种合法次梯度：

$$g_i=p_i\,g,\quad p_i\ge0,\quad \sum_{i\in I}p_i=1,\quad p_i=0\ (i\notin I),$$

其中$g$为上游梯度。当前安装的`torch_scatter 2.1.2+pt27cu128`路径在本例将梯度给一个被选中的argmax，重复执行时选择可以变化；严格PyTorch deterministic开关在此次测试没有报错。原生amax在本例对并列最大值平均分配，即$p_i=1/|I|$。

因此修正保持有限输入的最大值输出和参数量，但**改变精确并列时的训练次梯度规则**。这是执行与复现修正，不是新增FPPool、理论创新或泛化证明。不能把新旧训练值混入同一未注明协议变更的排行榜。

## 实现与边界

- [repro.py](repro.py)：仅本方向的可选入口；复用已有context/centered/FP runner，在进程内将共享读出的max函数路由到原生amax，manifest记录梯度策略，退出或异常均恢复。沿用顺序执行限制，不支持同进程并发运行。
- `tests/test_retry_repro.py`：CPU/GPU检查负值、并列最大值的准确前向与平均梯度，以及异常后路由恢复。
- `tools/trace_layer_repro.py`：记录真实样本、输入、CPU/CUDA RNG、预测、全部参数梯度及更新权重，支持首个差异批次捕获和模块观测。
- `tools/check_layer_max_backward.py`：先验证捕获预测逐位重放，再检查真实max输入和固定上游梯度。
- `tools/report_layer_repro.py`：无训练重算六对追踪比较、隔离证据、来源SHA与18段诊断清单。

既有`model.py`、`context_scale.py`、原训练器、Loss、历史配置/报告/manifest未修改；可靠性worktree和计划未写入。原8fit和6fit的No-Go保留。新入口尚未执行正式效果筛查。

## 诊断更正与限制

初版离线工具遗漏训练中的`set_float32_matmul_precision("high")`，用默认精度重放得到不同隐藏状态，错误地未发现并列。因此早期“max没有并列”的观测不能排除真实路径；这些初版产物保留但已降级。修正工具直接调用原seed设置，并断言预测SHA匹配捕获记录，再形成上述证据。初版短程工具使用直接MSE均值，后续完整对照及修正验收均改为复用实际DeltaLoss的plain-MSE路径；不把初版作为精确训练复现。

已确认这一真实批次的可复现原因与修正，但未保存历史完整训练的逐批梯度，不能断言它解释所有历史差异。首epoch重复通过也不保证后续epoch、244任务或所有GPU内核逐位重复。当前没有修正后性能收益结果。

下一步先在冻结的修正协议下验证更长轨迹及另一任务，再讨论是否需要重做有限的同期机制对照；不恢复旧8次seed队列，不新增模块、Loss、任务搜索或FP组合。

复核已有证据（不训练）：

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe -m unittest discover -s tests -p test_retry_repro.py -v
& D:\Tools\conda-envs\graphcliff\python.exe tools/report_layer_repro.py
```

报告目录为`D:/GraphCliff-Pair-FPPool-LongPoly/experiments/mechanism_retry`；真实输入、捕获权重、逐batch记录和原始重放证据仅保存在本worktree被忽略的`artifacts/mechanism_retry_repro_20261008_*`，不提交分子级记录和权重。
