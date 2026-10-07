# self上下文共同尺度先导：算子动机成立，预测收益未通过

2026-10-08。新三臂先导6/6训练、6个最佳checkpoint独立重放通过。**主候选context的8项严格比较只通过2项；它在两任务都改善旧centered的Overall，但Cliff未改善，且相对self的双指标均退化。因此本候选No-Go，不扩训、不组合FP。** 旧先导No-Go保留。此结论限定当前方案、任务与运行，不否定所有差分或归一化方法。

方案与真实隐藏状态诊断：[执行前方案](self_context_scale_execution_20261008.md)。算子与条件推导：[设计说明](self_context_scale_proposal_20261008.md)。完整机器核验：[结果JSON](context_scale_results_20261008.json)。训练从提交aa1381fb7f3a2da6ab8cd4de4f56706f97adbb9c、干净工作区启动，没有中途改模型/参数/Loss或评价规则。

## 1. 完整同期结果

234/244 × self/旧centered/context × seed42，全部从相同初始化同期重跑。H256、3层、4heads、普通MSE、batch32、lr1e-4、最多100epoch、patience15，与原预算一致。三臂均6,810,654参数，完整初始化hash一致。checkpoint仅按validation Overall MSE选择。

|任务|臂|Overall RMSE|Cliff RMSE|Noncliff RMSE|最佳/实际epoch|
|---|---|---:|---:|---:|---:|
|234 Ki|self|0.724281|0.753279|0.700816|18/33|
|234 Ki|centered|0.739219|0.810960|0.677971|7/22|
|234 Ki|context|0.731176|0.830053|0.643533|14/29|
|244 Ki|self|0.897745|1.074404|0.698037|21/36|
|244 Ki|centered|0.926760|1.096952|0.737451|6/21|
|244 Ki|context|0.916479|1.097131|0.712172|9/24|

相对旧centered，context的Overall改善234为1.09%、244为1.11%，Cliff分别退化2.35%、0.016%。244的Cliff差极小，不能解释为统计上可辨别的差距；但No-Go并非仅由这个极小差驱动：相对self，context两任务Overall分别退化0.95%、2.09%，Cliff分别退化10.19%、2.12%。单次结果的百分比仍不是显著性证据。

|context严格低于对照|234 Overall|234 Cliff|244 Overall|244 Cliff|
|---|---|---|---|---|
|self|未通过|未通过|未通过|未通过|
|centered|通过|未通过|通过|未通过|

按照执行前规则不扩展此候选，不调整eps、指标权重或Loss，不恢复8次seed扩训，不运行FP组合，不另选self为本研究的新方法。没有full/cross同期臂，不能将本表扩展成原全对照门槛的通过证明。

## 2. 同seed重跑差异使历史线索需要降级

本轮两个旧臂与原先导的已记录源码hash、优化参数/划分、torch/PyG/RDKit版本及完整初始化一致，但最终权重和预测没有保持一致。原先就只承诺best-effort deterministic，不承诺GPU逐位训练复现。**目前未定位分歧原因，不能直接归因GPU，也不能说所有运行条件已经完全相同。**

|任务|臂|旧Overall→本轮Overall|旧Cliff→本轮Cliff|平均绝对预测差|最大绝对预测差|
|---|---|---:|---:|---:|---:|
|234 Ki|self|0.733442→0.724281|0.783217→0.753279|0.274169|1.127962|
|234 Ki|centered|0.741989→0.739219|0.781515→0.810960|0.317097|1.090019|
|244 Ki|self|0.931855→0.897745|1.141951→1.074404|0.374662|1.647570|
|244 Ki|centered|0.927017→0.926760|1.118174→1.096952|0.108887|0.758444|

234 self验证轨迹前16epoch完全一致，第17epoch起分歧；旧centered第1epoch即不同。两个运行选择的最佳epoch也可能不同。上表是已选checkpoint的预测差，不能当成浮点舍入误差量级、训练状态逐步差或跨seed置信区间。

旧[M48诊断](layer_tradeoffs_20261008.md)对原保存预测的恒等式和计数仍正确，但“centered相对self在两个任务都获得Cliff收益”的模式未在本轮保持：本轮两个任务self的Cliff均更低。因此只能将其保留为原单次运行上的描述，不能写成稳定机制。当前评价使用本轮同期对照，不用历史较弱self分数替代它。

后续若继续研究，应先定位本方向同seed重跑差异并明确允许的复现范围，再评价微小结构收益；本轮没有启动额外训练来排查，也没有修改另一可靠性对话的代码或计划。

## 3. 理论和实际收益分别得到了什么证据

设计的固定上下文响应界、同输入零注入及构造输入检查成立；在旧centered的真实隐藏状态中，后两层沿差分方向的幅度压缩也确实被观察到。共同self尺度改变了这一算子性质，没有增加参数。

但避免差分自身归一化，并不意味着这个变化对活性预测必然有益。本轮context相对旧centered在234的Noncliff改善而Cliff退化，在244的Noncliff也改善但Cliff没有改善。不能把恢复幅度当成提升活性悬崖预测的理论证明，不能据此推断旧幅度压缩是某个化学机制，也不能把数值性质当成论文已建立的主贡献。

目前最准确的定位是：一个有明确反例动机、参数不增加、可推导且可证伪的小改动候选，未通过当前预测对照。原创性亦未确认，不包装为最终模型，不因这次失败继续堆模块。

## 4. 核验、成本与产物

- 当前6组审计检查19份训练源码、固定外部依赖、来源清单及记录环境；历史8组审计也通过，历史预测hash保存在结果JSON。
- 对完整训练/验证行、结构Top-1参考、预测身份、初始化、参数量、预算、普通MSE权重、最小Overall选权重核验通过；重算Overall/Cliff/Noncliff等保存指标。
- 6个checkpoint重放全部1617条验证预测，最大差8.89e-16；每个首个batch的32对检查交换反对称和同分子零差值，共192对/项，最大误差均0。
- 6fit累计1109.97秒，即18.50分钟，不含数据准备、诊断、预检及重放。stderr为空。官方test评估0，不解析其标签；训练参考活性仅在差值输出后恢复预测。

|任务|臂|fit秒|峰值CUDA MiB|
|---|---|---:|---:|
|234 Ki|self|205.39|537.24|
|234 Ki|centered|158.12|657.04|
|234 Ki|context|227.83|656.56|
|244 Ki|self|207.27|930.19|
|244 Ki|centered|138.04|1460.20|
|244 Ki|context|173.33|1484.36|

各臂早停epoch不同，不用fit总时长推断单步效率。self每层双向2次MHA，旧centered/context为4次；参数相同不等于计算量相同。显存为max allocated MiB，不是整机占用。

只读输入：D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv与CHEMBL244_Ki.csv，覆盖和hash见context_scale_preflight_20261008.json。输出：D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_context_scale_screen_20261008及同名前缀日志/process.json，权重/数据/逐分子预测不提交。

配置SHA256：549578417396838fbd8dd1ec3cd43f17c4180638c7ae6cd2781fad560afd4fa9。模型/训练来源在启动后冻结；结果工具在tools/中编写，不属于训练捕获的实验源码。复算命令：

```powershell
Set-Location D:\GraphCliff-Pair-FPPool-LongPoly
& D:\Tools\conda-envs\graphcliff\python.exe -m tools.report_context_scale artifacts/mechanism_retry_context_scale_screen_20261008 --json-output artifacts/context_scale_reaudit.json
```

执行前提交9个本方向文件，清单见方案与Git aa1381f。结果提交仅限本报告、context_scale_results_20261008.json、tools/report_context_scale.py、本目录README.md和worktree.md，共5文件。不改旧模型/训练器/旧实验结论、不写可靠性目录、不自动合并。当前候选停止扩展。
