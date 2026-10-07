# 逐层替换的事后诊断：Cliff收益与Noncliff代价

2026-10-08。沿用已完成的8组验证预测，没有新增训练。**centered相对同容量self，在两个开发任务都降低Cliff MSE、提高Noncliff MSE；234的代价超过收益，244的收益超过代价。** 这是当前样本的描述性分解，不是独立确认、显著性检验或化学机制解释；不撤销原No-Go，不改候选、评价指标或停止规则。

来源：[原始先导与停止结论](layer_results_20261007.md)、[本次聚合JSON](layer_tradeoffs_20261008.json)。下表保留centered对全部三个对照的结果，不能只看相对self的共同模式。

## 1. 哪一部分决定了总体结果

定义MSE收益为“对照MSE − centered MSE”，正数表示centered改善，负数表示退化。这里拆分MSE，不能把两个子集RMSE直接按人数加权。

|任务|对照|Cliff MSE收益|Noncliff MSE收益|Overall MSE收益|平方误差改善查询数/全部|
|---|---|---:|---:|---:|---:|
|234 Ki|full|+0.065854059|−0.043044494|+0.004691858|146/292|
|234 Ki|self|+0.002662511|−0.024530531|−0.012610293|135/292|
|234 Ki|cross|+0.106458721|−0.043123226|+0.022446943|145/292|
|244 Ki|full|+0.067476121|+0.017741417|+0.041501316|119/247|
|244 Ki|self|+0.053740312|−0.031940687|+0.008991936|124/247|
|244 Ki|cross|+0.010833840|−0.001607584|+0.004336092|116/247|

234有128个Cliff、164个Noncliff。相对self，Cliff对总体MSE收益的加权贡献为+0.001167128，Noncliff贡献为−0.013777421，合计−0.012610293。这解释了234为什么Cliff略好、Overall却输给self。

244有118个Cliff、129个Noncliff。相对self，两项加权贡献分别为+0.025673509和−0.016681573，合计+0.008991936。因此同样存在Noncliff代价，但该任务的Cliff收益足以抵消它。

这个模式必须限定对照：244相对full的Cliff和Noncliff均改善，不能说centered必然牺牲Noncliff。244相对full只有119/247个查询改善而Overall仍下降，也说明“改善的样本占比”和“平均平方误差收益”不是同一个量；误差变化幅度同样起作用。表中的查询数是描述计数，不把相关分子当作独立统计重复。

## 2. 与代码和预测对应的精确分解

对任一对照$b$，令$e_b=y-\widehat y_b$，$c=\widehat y_{centered}-\widehat y_b$。对既定子集$S$，逐样本平方展开给出

$$G_S=\overline{e_b^2}_S-\overline{(e_b-c)^2}_S
=2\overline{e_bc}_S-\overline{c^2}_S.$$

因此预测变化必须与对照误差有足够乘积，才能抵消自身变化能量。这里$c$只是两个独立训练模型的预测差，并不是新训练的补偿分支；不能将这个恒等式当成已验证残差学习或因果机制。

令$p=n_{Cliff}/n_{Overall}$，两子集互斥且覆盖验证集，故

$$G_{Overall}=pG_{Cliff}+(1-p)G_{Noncliff}.$$

六组对照的18项子集恒等式及6项人数加权恢复核验均通过，绝对容差1e-12。没有按结果重新划分子集、搜索相似度阈值或改变Cliff权重。

## 3. 对研究和论文的含义

可以写：“在两个已使用的开发任务、seed42下，centered相对self呈现Cliff与Noncliff的误差权衡；234总体退化来自Noncliff代价超过Cliff收益，244则相反。”这句话限定了样本、对照和指标，能由保存预测复算。

不能写“跨分子相减已学会非共有官能团”“理论证明Cliff泛化提升”或“所有Cross-Attention无效”。self与centered的处理方式不同，训练后注意力参数也不同；误差分组和零增量性质均不能单独识别收益的化学来源。此前RMSNorm零点导数只是初始化局部性质，本次没有把它与逐查询误差建立因果联系。

本轮最合适的处理是保留这条研究线索，维持不扩训，而非通过提高Cliff权重、换主指标、加Loss或组合FP来补救原门槛。若用户明确要求重新讨论训练，应另立执行前方案，说明新问题与预算；不得重写原先导的通过/失败结论。本报告没有启动这些后续实验。

## 4. 复现与文件范围

只读输入：`D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_layer_core_screen_20261007`内8份`validation_predictions.csv`及原manifest/完成记录。训练来源提交`151cda243d792b3ca596a8cda0e9dcd6e600ba61`，本次分析起点`1939dbc`。聚合JSON记录8份预测文件及manifest的SHA256，复用已有审计检查完整8组、18份源码与保存指标，并核对比较双方查询、参考、活性与Cliff身份一致。

```powershell
Set-Location D:\GraphCliff-Pair-FPPool-LongPoly
& D:\Tools\conda-envs\graphcliff\python.exe -m tools.report_layer_tradeoffs artifacts/mechanism_retry_layer_core_screen_20261007 --json-output artifacts/layer_tradeoffs_reaudit.json
```

只改本目录`worktree.md`，新增本报告、`layer_tradeoffs_20261008.json`及`tools/report_layer_tradeoffs.py`，共4个本方向文件。原模型、训练器、配置、评价门槛和已有结果保持原样；不写可靠性worktree，不合并分支。新增fit为0，官方test评估为0，未发布逐分子预测、数据或权重。
