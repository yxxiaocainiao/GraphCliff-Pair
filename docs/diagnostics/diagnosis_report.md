# 参考差值路线的阶段诊断与暂停决策

日期：2026-10-05。状态：**在D2按用户授权提前暂停**。D0输入核对、D1相似度诊断、D2差值/cliff构成检查已完成；D3学习曲线、后续实验设计和任何新增训练均未启动。本报告不是完整新目标的完成声明。

## 已得到的信息

42组已审计输出的预测、history和checkpoint字节身份与历史审计一致，指标独立重算和相同验证分子对齐通过。所有诊断仅读取已保存输出，没有打开原始来源CSV、读取test标签、运行模型推理或训练。

在CHEMBL234中，最低参考相似度层Q1有64分子，最高层Q4有65分子。global−direct的层间平方误差差定义为`Q1处理−对照MSE差 − Q4同一差`，三个训练seed分别为−0.1237、−0.2416、−0.2288；即相对direct的劣势在Q4更大，而非简单集中在低相似度参考上。D1各层删除最大绝对误差差样本后方向仍一致。这是新的分层描述，但不证明参考选择就是退化原因。

![参考相似度](reference_similarity.png)

来源：[固定分析计划](analysis_plan.json)、[42组输入核对](input_check.json)、[D1完整结果](reference_results.md)。所有任务、种子和候选均保留，不只展示通过的比较。

## 为什么在D2暂停

D2仅沿着分析前固定优先级选出的234/global−direct继续，按分子cliff标记和绝对差值大小分组。差值界限为训练配对中位绝对差0.451545，不按验证误差调整。

|分组|Q1 / Q4分子数|三个seed的Q1−Q4差|信息与限制|
|---|---:|---|---|
|非cliff分子|46 / 32|+0.1527、+0.0516、+0.0158|方向与D1总体相反，不能推广总体模式到所有分子|
|cliff分子|18 / 33|−0.6011、−0.8221、−0.6080|有探索性信号，Q1未达到预设20分子门槛|
|小绝对差值|25 / 33|−0.1663、−0.1941、−0.1691|删除每层最大绝对误差差样本后，seed42变为+0.0091|
|大绝对差值|39 / 32|−0.1612、−0.3036、−0.3265|同样删除后，seed43变为+0.0884|

![差值与cliff构成](delta_cliff_composition.png)

因此，D1的稳定总体方向并未成为符合预设规则的稳定条件模式：两类边际分组都没有通过D2门槛。按用户“中途成果不怎么样就立刻暂停”的要求，不进入D3、不改门槛、不改选其他D1候选、不开始新实验。

20分子、5%的D1筛选及删除极端样本检查是本轮预先固定的实用判断规则，不是统计显著性、样本量功效或普遍无效证明。cliff子组18分子的不足只说明未通过这项保守规则，不能据此断言信号不存在。删除检查说明描述依赖极端样本，也不能把极端样本直接判定为错误或噪声。

来源：[D2分析计划](composition_plan.json)、[全部子组和反例](composition_results.md)、[原始精度汇总](composition_results.json)。标准库csv/math独立核对24个子组×seed记录的样本数、原始差和删除差，与汇总一致。

## 当前研究决策

这轮诊断增加了两点信息：退化不应仅归咎于低结构相似度；分子cliff构成及少数极端误差与总体模式有关。它尚未提供足以启动新训练的机制解释。建议保留现有实现与阴性结果，将参考差值组合路线保持暂停；本目标不自动提议第二轮阈值搜索或扩大数据范围。

结果只限两个已有开发任务、同一固定数据划分、三个训练seed与当前实现。FPPool/Loss完整消融只有一个seed，不因本轮参考诊断形成稳定贡献结论。官方test仍未评估。

## 复现已完成部分

需具有三个已审计输出目录和历史审计文档；源码仓库不包含数据或checkpoint。使用原运行输出，安装可选绘图依赖`pip install -e ".[diagnostics]"`后，选用全新输出目录：

```powershell
python tools/diagnose_reference.py --phase prepare --output artifacts/my_diagnostics
python tools/diagnose_reference.py --phase reference --output artifacts/my_diagnostics
python tools/diagnose_composition.py --phase prepare --output artifacts/my_diagnostics
python tools/diagnose_composition.py --phase composition --output artifacts/my_diagnostics
```

D2入口只允许D1已通过；报告存在时拒绝静默覆盖。若D1未通过，应立即停止而不调用D2。没有重新训练、重新推理或打开来源CSV的步骤。Git会规范化文本换行，所以已跟踪文档身份使用明确的LF规范化UTF-8哈希；ignored artifact的预测/history/checkpoint仍核对原始字节。原冻结计划及初次发布commit也保留，换行修正未改变任何分析门槛或数值。
