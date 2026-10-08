# M54：三种执行上下文仍未重现历史前向分叉

2026-10-08。结论：真实step485捕获在三种执行上下文、各两个独立进程中，120次无模块hook前向全部匹配M52历史repeat2，未得到repeat1。**具体差异算子仍未知，M52长程重复失败仍保留。** 本轮完成诊断，没有模型修正或收益结论。

## 执行范围

|上下文|独立进程|每进程测量|额外操作|结果|
|---|---:|---:|---|---|
|fresh|2|20次前向|无|均为历史repeat2|
|allocator_churn|2|20次前向|模型构造前分配/释放CUDA块，保留8,454,403字节|均为历史repeat2|
|warm_backward|2|20次前向|先做2次预测平方均值的反向，再清空梯度|均为历史repeat2|

总计120次测量前向，另有4次预热前向及4次反向；optimizer步骤0，官方test评估0。预热目标仅用于启动反向执行路径，不是训练损失试验。每次测量恢复相同CPU/CUDA/Python/NumPy RNG；120次不是120个独立统计样本。

沿用234/self、原生amax、strict deterministic、high矩阵精度及现有TF32设置，未修改模型/Dropout/精度。无模块hook，但整个前向结束后仍进行CPU哈希同步；这是受控重放，不是完整历史轨迹。

显存扰动确实改变布局：query.x地址模4096由0变为3072，首参数由0变为2048。三个上下文的存活分配字节分别为61,389,824、69,845,504、94,944,256；该差异仅证明执行上下文存在变化，不用于归因。所有进程输入/权重SHA、测量前RNG一致，测量后RNG匹配历史repeat2，预热未改变模型状态或图输入。

## 核验与边界

执行前工具和计划提交为`7d7950e`。6个worker来源冻结，156次来源SHA检查通过；另外核验M52两条历史trace文件SHA及其来源未变。父进程退出、stderr空，聚合核验通过。原始逐样本预测只保留在本地artifacts。

- 输入：`D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_forward_20261008_a/step485.pt`，SHA为`f47c54b26281e471e14680a869c595d850ba6aa4b5fe8184b04be67aea6ede28`。
- 原始输出：`D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_forward_context_20261008/`。
- 可提交输出：[聚合JSON](layer_forward_context_results_20261008.json)、本报告；新增`tools/check_layer_forward_context.py`，更新本方向README/worktree日志，共5个唯一文件。
- 原模型、训练器、Loss、历史工具/配置/结果不改；可靠性目录/计划不写，不合并，不恢复收益队列。

这些结果只能说明本次采样的三种上下文不足以触发另一分支，不能排除其他分配历史、底层执行状态或观测影响。M51并列max梯度修正的局部证据、M52失败及机制No-Go均不改变。不通过增加FP、改精度或关闭Dropout绕过失败。

已有结果复核（不会训练、不会重放模型；会重写本地artifacts聚合文件）：

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe tools/check_layer_forward_context.py --report --output artifacts/mechanism_retry_forward_context_20261008
```

停止同类固定状态重放。下一项有价值的检查是原M52 worker的未插桩完整前缀对照，先判断历史分叉是否仍能出现，再决定在哪段执行底层算子观察；本轮未启动该对照。完整轨迹复现验收通过前，仍不做收益排序。
