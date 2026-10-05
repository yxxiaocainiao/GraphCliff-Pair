# 历史实验台账与框架纠错

日期：2026-10-05。**这是事后整理，不是实验前注册。** 本轮只读取三个独立目录的既有验证产物，未训练、未改参数、未评估test。原项目与my_work保持只读。

## 核心纠错

原 GraphCliff 的直接预测＋FPPool **已经做过**，不是待启动候选。CHEMBL3979_EC50，seed42/43/44，固定810train/89validation；六组预测按历史float32标签、float64误差累计独立复算通过。平均Overall/Cliff：Baseline 0.584489/0.675169，FPPool 0.593744/0.690374；Cliff仅1/3seed改善，未达旧门槛。停止直接替换路线，不重复这六组训练。

另有残差FPPool、跨任务、GMT容量近似对照、归属扰动和解释诊断。新项目只查Pair内部时遗漏了这些旧结果，因此此前“直接组合未测”的推荐不完整，本页及当前README纠正；训练源码和旧结果均不改。

## 检索范围与计数

输入目录：`D:/GraphCliff-main/eval_out`、`D:/GraphCliff-Pair/artifacts`、`D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines/results`。发现223个具备验证预测、元数据和检查点的记录（原项目68、新项目128、旧基线27），另有2处history无验证预测的中断现场。874个所读输入SHA256复核不变。

223是**产物记录数，不是223次独立有效正式实验**：包括冒烟、复制/恢复产物、重跑和重复权重。同权重目录分组见inventory.json，不把复制目录加为新种子；阶段不完整不因有14个结果就称完整18组。

台账重新计算Overall/Cliff/Non-cliff；有同名历史指标时最大差4.45e−16。每条记录包含路径、任务/种子/模型、预测形式、冒烟标记、轮数、参数、实际保存耗时、配置/manifest/权重来源哈希、历史源码哈希、可获得的提交版本及核验范围。缺少耗时、版本或指标时保留null，不编造。Baseline目录没有有效HEAD，版本未知；其产物/源码哈希是身份依据。

本轮不是所有历史实验的穷尽搜查：仅覆盖上述三目录可发现的验证产物。作者checkpoint和my_work既有SVM/Chemprop30任务另列为背景资产；未读取其test预测参与本轮排名。之前234的SVM/Chemprop验证推断可直接复用，见baseline_reuse_234报告。

## 逐阶段产物范围

|来源|目录阶段|记录数|冒烟记录数|保存拟合耗时合计秒|
|---|---|---:|---:|---:|
|baselines|GCN/GAT/MLP_saved_validation|27|0|0.0（有缺失）|
|original|component_pair_pilot|24|0|48092.9|
|original|cross_task_residual|19|0|4758.0|
|original|fppool_multiseed|4|0|759.9|
|original|fppool_pilot|3|0|670.7|
|original|fppool_residual|2|0|485.0|
|original|gmt_residual|6|0|309.5|
|original|loss_pilot|4|0|208.9|
|original|membership_ablation|6|0|1453.8|
|pair|ablation_seed42_20261004|18|0|21011.9|
|pair|ablation_seed43_20261004|14|0|18169.8|
|pair|ablation_seed43_recovered_20261005|14|0|18169.8|
|pair|aca_coverage_smoke_20261005|2|2|0.8|
|pair|aca_pilot_20261005|6|0|1431.6|
|pair|aca_smoke_20261005|6|6|1.7|
|pair|branch_swap_screen_20261005|6|0|1053.0|
|pair|branch_swap_smoke_20261005|6|6|2.3|
|pair|interaction_seed42_20261004|8|0|1569.8|
|pair|interaction_seed43_44_20261004|16|0|3770.7|
|pair|smoke_20261004|16|16|52.9|
|pair|smoke_20261004_verified|16|16|52.5|

耗时是保存日志的拟合耗时之和，包含复制目录时会重复计数，不代表累计GPU小时；早期异步/后续同步运行、验证与checkpoint开销各异，不能直接做效率排名。逐记录数据见inventory.json。

## 已回答与尚缺的对照

|候选|已回答|关键缺口／边界|当前决定|
|---|---|---|---|
|直接FPPool替换|3979三seed及seed42无SAG对照；平均Cliff/Overall未胜基线|单任务；初期GPU并列max梯度与后期固定首节点max协议不同|停止，不重做|
|残差FPPool|2047/235/287三seed，只有287达旧门槛；GMT与归属扰动对照|2047反例；GMT仅参数近似；小收益不足以确立普遍化学归属收益|停止扩展|
|结构分量/分子对辅助Loss（旧项目）|234/244四臂三seed24组；冻结报告No-Go|不能混作Pair参考差值模型；CUDA首批故障根因未定|停止扩展|
|编码后Cross-Attention（新项目）|两任务三seed，与direct/global/同容量pair MLP比较|均未胜direct/MLP；参考任务形式、容量与计算预算要分开|停止扩展|
|替换LongPoly的Cross-Attention|两任务seed42共6正式组＋重放，旧门槛未过|只有一个种子；残差/归一化/容量不同，不是注意力因果消融|停止扩展|
|Pair内FPPool/动态Loss|seed42完整条件消融；seed43存在14/18部分产物|未完成三seed，不以部分目录补足稳定贡献|保持暂停|
|ACA迁移|234三seed6正式组，恶化触发提前停止|244对照中断；其余未启动；原18组执行时点提前，数值门槛未改|停止|
|BMC/LDS/FDS/IA-MoE|官方来源/许可/接口审查；DIR/BMC合成检查|没有本地分子性能证据；IA-MoE许可与完整入口未确认|不启动训练|

## 可解释性已有成果

原项目v4已有12checkpoint、157个任务内分子、942次解释；方法为**encoder隐藏表示Gradient×Input**，20%原子隐藏表示置零，20次同规模随机遮罩，跨seed集合Jaccard/归因秩相关。本轮重新读取942条记录，重算随机float32均值、差值与36行四项分组摘要（144项）一致。

它不是图输入遮罩，不是结合机制因果证据，也没有确认FPPool解释优势。稳定性汇总已读取/绑定哈希，本轮未重新从原子分数重算；没有重放12checkpoint，旧重放证明仅作为历史记录。技术兼容性与最终必需解释规范在下一里程碑报告。

## 本轮异常与限制

- 首次审计把旧NumPy float32随机均值按Python float64计算，核验阻断；查看原解释脚本后恢复原精度，没有放宽容差、改数据或挑病例，首次空输出目录保留。
- 第二次在my_work无有效Git HEAD处阻断；版本未知被显式记录，未初始化仓库或创造历史版本，第二次现场保留。
- v3完成核心复算；v4补齐最近manifest、历史源码身份及重复权重组。均未触发训练或覆盖旧产物。
- 当前字节哈希证明读取期间未变化；不自动证明所有旧源码与训练时一致，严格历史匹配须引用对应审计，未知状态不升级。

## 核验与文件边界

本轮新增tools/audit_framework.py与本目录；更新README/task_plan/notes、milestones、publication_feasibility及阶段证据解读。旧model/Dataset/Loss/Optimizer、三目录原始输入与所有权重不改。

```powershell
python tools/audit_framework.py artifacts/framework_review_reproduce
```

输出必须新目录。输入哈希和核验摘要见verification.json；本轮审计工具版本绑定analysis_tool_sha256.json。当前任务目标是审查并决定止损，不宣称发表目标完成。
