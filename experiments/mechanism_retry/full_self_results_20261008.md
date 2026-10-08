# M60 full/self多seed确认

No-Go; stop this architecture round

**结论：234通过任务级开发筛查，244未通过，整体不通过；按执行前协议停止本轮架构方向。** self相对full的三seed均值在234的Overall/Cliff改善2.32%/2.36%，在244恶化5.47%/6.37%。这是一项任务条件下的性能差异，不支持把self替换作为跨两任务稳定改进，也不能推断所有self/Cross-Attention普遍无效。

本轮是对基础信号的确认，不是救回centered/anchor候选；未匹配容量，不识别LongPoly替换的因果机制。结果应作为与老师讨论下一研究问题的依据，不再自动添加门控、尺度、FPPool或调整early stopping补救。

固定2任务×3seed×2臂，全部新训练；按验证Overall选模，100epoch上限/patience15。仅开发集，不是独立test，也不宣称显著性。

|任务|seed|臂|Overall RMSE|Cliff RMSE|best/实际epoch|
|---|---:|---|---:|---:|---:|
|CHEMBL234_Ki|42|full|0.749560|0.817259|9/24|
|CHEMBL234_Ki|42|self|0.717925|0.784896|20/35|
|CHEMBL234_Ki|43|full|0.749644|0.820427|8/23|
|CHEMBL234_Ki|43|self|0.715253|0.789091|38/53|
|CHEMBL234_Ki|44|full|0.713416|0.776672|11/26|
|CHEMBL234_Ki|44|self|0.728013|0.783468|25/40|
|CHEMBL244_Ki|42|full|0.946027|1.134052|3/18|
|CHEMBL244_Ki|42|self|0.916168|1.092291|26/41|
|CHEMBL244_Ki|43|full|0.854230|1.031005|55/70|
|CHEMBL244_Ki|43|self|0.948682|1.143291|2/17|
|CHEMBL244_Ki|44|full|0.861906|1.018825|37/52|
|CHEMBL244_Ki|44|self|0.943044|1.151149|4/19|

均值±样本标准差(ddof=1)，同一固定划分的3个训练seed：

|任务|臂|Overall|Cliff|
|---|---|---:|---:|
|CHEMBL234_Ki|full|0.737540 ± 0.020892|0.804786 ± 0.024399|
|CHEMBL234_Ki|self|0.720397 ± 0.006730|0.785818 ± 0.002923|
|CHEMBL244_Ki|full|0.887388 ± 0.050928|1.061294 ± 0.063304|
|CHEMBL244_Ki|self|0.935965 ± 0.017374|1.128910 ± 0.031956|

门槛：每任务双指标均值均优于full，且至少2/3 seed双指标同时改善；两任务均通过才继续容量/算子混杂核验。

- CHEMBL234_Ki：均值比较{'overall_rmse': True, 'cliff_rmse': True}，双指标同时改善2/3 seed，任务通过=True。
- CHEMBL244_Ki：均值比较{'overall_rmse': False, 'cliff_rmse': False}，双指标同时改善1/3 seed，任务通过=False。

核验：12个checkpoint重放，最大绝对差0；实际418epoch/31873 optimizer步骤，fit合计52.20分钟。

边界：full6021198、self6810654参数，未匹配容量；两臂均使用训练参考活性恢复查询预测，不是单分子官方GraphCliff。三个seed不能证明统计显著性、化学机制或创新；通过筛查也不能直接归因去除LongPoly。GPU采用strict可用算子/native amax，但历史同seed分叉仍未解决，不承诺全程逐位。旧候选No-Go不改，不自动添加模块或扩训。逐seed差值、nearest-reference及耗时/显存/初始化/来源核验见同名JSON。

上下文：仅使用最近训练参考活性预测的Overall/Cliff为234的0.786663/0.884097、244的0.955152/1.147978。它不是full/self之间的容量对照，不能用其较弱成绩证明替换有效。12个checkpoint共重放3234条验证预测；另有12个真实批次预检，均为0 optimizer更新。训练来源为干净提交f5269a0，32个suite来源文件及21个原audit源码检查通过。全部原始产物位于本worktree的artifacts/mechanism_retry_full_self_20261008，官方test0。
