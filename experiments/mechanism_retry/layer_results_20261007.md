# LongPoly逐层替换先导：保留局部收益，主候选不扩训

2026-10-07。8/8组正式训练及8个最佳checkpoint独立重放完成。**centered在两任务均优于保留LongPoly的full对照，但234 Overall不胜同容量self；12项严格比较通过11项，未达到执行前门槛。本轮不扩训、不运行4次FP组合消融。** 这不是所有Cross-Attention普遍无效的证据。

输入方案：[执行前方案](layer_execution_20261007.md)、[完整预检](layer_preflight_20261007.json)。机器核验：[layer_results_20261007.json](layer_results_20261007.json)。既有实现与完整公式：[README第3节](README.md)。

## 1. 固定问题与完整结果

两任务×full/self/cross/centered×seed42，H256、3层、4heads、batch32、lr1e-4、普通MSE，最多100epoch、patience15。唯一主候选centered在训练前固定；没有中途更换参数、Loss、任务或指标。所有checkpoint按validation Overall MSE选择，不按Cliff选择。

full是保留原LongPoly的**Pair对照**，不是单分子GraphCliff官方test成绩。查询和训练参考的结构进入模型，参考活性只在差值输出后用于恢复预测。234为2632train/292valid/128valid-Cliff，244为2229train/247valid/118valid-Cliff。

RMSE越低越好；相对full改善百分比为正表示下降。保留全部臂，不只列阳性结果。

|任务|臂|Overall RMSE|Cliff RMSE|Noncliff RMSE|Overall改善%|Cliff改善%|最佳/实际epoch|
|---|---|---:|---:|---:|---:|---:|---:|
|234 Ki|full|0.745144|0.822569|0.678604|0.00|0.00|5/20|
|234 Ki|self|0.733442|0.783217|0.692111|+1.57|+4.78|20/35|
|234 Ki|cross|0.756964|0.846891|0.678546|−1.59|−2.96|4/19|
|234 Ki|centered|0.741989|0.781515|0.709611|+0.42|+4.99|21/36|
|244 Ki|full|0.949138|1.147950|0.720755|0.00|0.00|2/17|
|244 Ki|self|0.931855|1.141951|0.685424|+1.82|+0.52|3/18|
|244 Ki|cross|0.929353|1.123008|0.707205|+2.08|+2.17|6/21|
|244 Ki|centered|0.927017|1.118174|0.708341|+2.33|+2.59|6/21|

234上centered的Cliff改善是真实保存结果；同时其Overall比self高0.008547，不能将Cliff收益解释为跨分子增量在双指标上已得到一致支持。244上centered双指标均胜三个对照，保留这一局部阳性信号。234 cross退化、244 cross改善，也不支持“Cross-Attention一概无效”。这里没有跨seed置信区间或显著性结论。

## 2. 按执行前规则决策

|centered严格低于对照|234 Overall|234 Cliff|244 Overall|244 Cliff|
|---|---|---|---|---|
|full|通过|通过|通过|通过|
|self|未通过|通过|通过|通过|
|cross|通过|通过|通过|通过|

既定门槛要求两任务、两指标、三个对照全部通过；不能在结果后放宽为“只胜full即可”。因此`optional_fp_ablation_allowed=false`，未执行full_fp/centered_fp，不能填充2×2交互表或宣称组合有效/无效。也没有将self改选为新主候选。

这项No-Go是本轮投入门槛，不是对方法族的统计否定。两个任务已反复用于开发，只有一个seed；full参数更少，胜full不能单独归因于cross-self机制。当前不足以把此候选写成已确认的论文核心。残差FP沿用其[独立先导结论](fp_results_20261007.md)，不与本表混成同一最终模型或统一排名。

## 3. 理论边界：零增量不保证小扰动稳定

既有候选使用

$$D_q=A(X_q,X_r)-A(X_q,X_q),\qquad B_q=T(D_q)-T(0).$$

共享MHA、无attention dropout时，相同输入使$D_q=0$，继而$B_q=0$；逐节点变换扣除$T(0)$保证学习到非零bias后这一性质仍成立。该恒等式不保证泛化提升，也不恢复原LongPoly在同分子对上的贡献。

执行中补充检查一个与当前代码直接对应的限制，**不修改候选或决策规则，不将其当作结果失败归因**。初始化时group_scale=1、group_bias=0、RMSNorm逐通道权重为1，因此对单个节点的$d$维增量，

$$B(D)=\operatorname{SiLU}\!\left(\frac{D}{\sqrt{\|D\|_2^2/d+\varepsilon}}\right),\qquad B(0)=0.$$

令$s(D)=\sqrt{\|D\|_2^2/d+\varepsilon}$。归一化的Jacobian为

$$J_R(D)=\frac{I}{s(D)}-\frac{DD^\top}{d\,s(D)^3}.$$

由$\operatorname{SiLU}'(0)=1/2$，得到

$$J_B(0)=\frac{1}{2\sqrt{\varepsilon}}I.$$

本环境float32、RMSNorm默认eps=None，实际有效$\varepsilon=1.1920928955078125\times10^{-7}$；预测零点导数1448.154687870，直接对当前`LayerSwap.transform`求自动微分得到1448.154663086，误差符合float32精度。零点输出仍严格为0，未做任何优化步骤；复现检查包含在核验脚本和结果JSON中。

所以“相同输入不注入”与“邻近输入只注入很小的量”是不同命题。上述导数只描述初始化零点的局部性质：训练后的scale/bias会改变它，真实增量未必处在这一邻域，最终输出还经过门控、残差及读出。不能据此声称实际梯度爆炸、化学扰动被放大1448倍，或已定位234的差距来源。当前也没有据此调eps、加门控或启动新的补救实验。

## 4. 核验、成本与复现

- 8/8完成，无异常stderr；8个最佳checkpoint均从磁盘重新构建模型并重放全部验证查询，共2156条模型预测。相对保存预测的最大绝对差为8.89e-16。
- 重算Overall/Cliff/Noncliff指标；核对完整train/valid行及结构Top-1参考、训练引用身份、查询/参考活性、Cliff标签、预算、普通MSE权重和最低Overall选择epoch。18份训练源码hash、固定外部FPPool文件及两份继承依赖来源校验通过。
- self/cross/centered的全部初始化hash一致，并与预检一致；四臂head初始化一致。full为6,021,198参数，其余各6,810,654，不声称对full等容量。
- 每个checkpoint在首个验证batch的32对上检查交换反对称和同分子零差值，共256对/项；最大绝对误差均为0。此项不是遍历所有图置换，也不是化学解释验证。
- 8fit实际累计1167.33秒，即19.46分钟；是训练器各fit计时之和，不含数据准备、预检和独立重放。官方test评估0次，不解析官方test标签；未新增划分或原子遮罩解释。

|任务|臂|fit秒|峰值CUDA MiB|每batch attention调用|训练attention调用|
|---|---|---:|---:|---:|---:|
|234 Ki|full|112.68|517.46|0|0|
|234 Ki|self|220.21|537.24|6|17430|
|234 Ki|cross|120.18|537.99|6|9462|
|234 Ki|centered|259.04|660.17|12|35856|
|244 Ki|full|89.96|610.84|0|0|
|244 Ki|self|103.93|929.28|6|7560|
|244 Ki|cross|123.24|927.44|6|8820|
|244 Ki|centered|138.09|1455.98|12|17640|

attention调用按当前三层双方向forward计数，训练调用乘实际训练batch数和epoch，不含验证/重放；它不是FLOPs测量。各臂早停epoch不同，fit总耗时也不能单独代表单步计算效率。峰值为训练器记录的allocated memory，不是整机显存占用。

训练来源提交：`151cda243d792b3ca596a8cda0e9dcd6e600ba61`，启动时Git干净。配置SHA256：`811a62aeae9a6729800b2fa1acfa50dc30f6b3324a34b85de0fbf607a614b5a7`。环境：torch2.7.1+cu128，CUDA，best-effort deterministic，不宣称跨硬件逐位一致。

```powershell
Set-Location D:\GraphCliff-Pair-FPPool-LongPoly
& D:\Tools\conda-envs\graphcliff\python.exe -m tools.report_layer_retry artifacts/mechanism_retry_layer_core_screen_20261007 --json-output artifacts/layer_reaudit.json
```

## 5. 文件和隔离边界

只读数据：`D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv`、`CHEMBL244_Ki.csv`，完整输入SHA见预检。输出：`D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_layer_core_screen_20261007`及同名前缀stdout/stderr/process.json；产物、权重、数据及逐分子预测不提交。

本轮执行前提交改动5个文件：本目录`layer_core_screen.json`、`layer_execution_20261007.md`、`layer_preflight_20261007.json`、`README.md`、`worktree.md`。本轮结果提交限定5个文件：本报告、`layer_results_20261007.json`、`README.md`、`worktree.md`及`tools/report_layer_retry.py`，合计8个唯一任务文件。

模型/训练器/原六臂配置/既有测试未修改，未改可靠性对话的计划、源码或工作目录。只推送分支`codex/residual-fp-centered-cross-20261007`，不自动合并。当前保留局部信号与完整反证，停止本轮候选扩展。
