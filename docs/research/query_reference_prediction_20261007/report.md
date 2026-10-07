# M57：固定模型的15个训练参考预测与有限案例诊断

结论：完成一次固定CHEMBL234_Ki/full模型的CPU推理，15个参考预测与15个查询重放均通过核验。两角色所选案例的差值误差项占比超过一半，但参考训练内误差并未接近零。当前只支持保留这些有限案例；尚无能由合法部署信息识别的稳定失误规则，也没有新增可信算法核心。不追加训练、任务、阈值或第二候选。

## 输入与执行合同

输入沿用[M56报告](../query_reference_activity_support_20261007/report.md)已固定的87个结构对及事后一log活性差子集：calibration 5对、evaluation 10对，15不同查询和15不同full-fit参考。原0.9≤Morgan Tanimoto<1、标签阈值与名单均未改变，没有重算外部风险特征。查询真值参与过案例筛选，结果仅是开发分区上的探索证据；这份名单不得作为部署时邻居或特征规则。

参考标签来自oof.csv的full-fit身份行，只读取其标签/结构/身份，未使用其中OOF预测。参考点预测全部来自同一个已有full checkpoint，属于**训练内预测**。查询点预测继续使用原角色缓存，没有用本次CPU输出替换。身份及canonical检查确认full-fit与两查询角色不交叠；官方test没有读取。

checkpoint：`artifacts/reliability_phase_a_20261005_v3/CHEMBL234_Ki/full/training/model_0/best.pt`，SHA256 `35523862aeb1fb5d028d0c7bb104433e0b649008bbc90be00ece867f80e2fcdd`，与历史artifact_binding一致。复用[既有runner](../../../experiments/reliability_base/run_phase_a.py)的run_command及M54 hash/JSON函数，经AST只抽取所需函数；模型推理直接调用已安装Chemprop官方CLI，没有自写点模型推理代码。

[协议](protocol.json)在提交 `0ccd029bd56e9b45a2cc51e31ba2fe652f9a5984` 推送后执行。Chemprop 2.2.3 / Torch 2.12.1+cu130 / Lightning 2.6.5 / RDKit 2026.3.4；CPU、devices1、workers0、batch64，单checkpoint路径、单CLI调用、30行，15参考新预测+15查询重放。实际CLI墙钟6.922秒，预算180秒CLI/240秒总执行；0训练、0风险拟合。CLI内部构图属于推理，不宣称零分子特征处理，也没有审计内部低层权重反序列化次数。

15个查询的CPU与原缓存预测最大绝对差 `2.9999999995311555e-7`，通过预先冻结的 `1e-4` 门槛。该门槛只是项目CPU/GPU float32数值检查，不是论文超参数、统计置信界或所有分子的误差保证。

## 全部两角色聚合结果

误差e=p−y，单位沿用缓存活性尺度；每对c=(e_q+e_r)/2、d=(e_q−e_r)/2，故平均两端平方误差等于c²+d²。表内共同/差值项先分别对全角色所选对求均值，占比为均值之比。

|角色|对数|查询RMSE|参考训练内RMSE|查询MAE|参考训练内MAE|差值MAE|差值符号准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|calibration|5|0.966553|0.578457|0.934801|0.496700|1.126174|60%|
|evaluation|10|0.673063|0.625873|0.471887|0.512498|0.928356|90%|

差值MAE为mean| (p_q−p_r)−(y_q−y_r) |；预测差为0时符号判错。符号准确率高不代表差值幅度准确，也不是接受决策收益。

|角色|平均两端平方误差|共同项均值c²|差值项均值d²|差值项占比|令参考误差为0时的差值项均值|
|---|---:|---:|---:|---:|---:|
|calibration|0.634419|0.231860|0.402559|63.453163%|0.233556|
|evaluation|0.422366|0.111769|0.310597|73.537478%|0.113254|

末列是同一查询误差下的代数极限mean(e_q²)/4，未调用第二个模型、不是公平算法消融、不是可部署oracle改进。全部未舍入数字见[results.json](results.json)。没有根据结果新增分组或阈值。

## 证据分层与方法缺口

**已观察事实：** 在这5/10对中，参考训练内RMSE分别0.578/0.626，没有接近0；evaluation参考误差与查询误差量级相近。两角色差值项占比分别63.45%/73.54%，与M56所述“参考误差近0时两项各半”的极限不同。有限病例中存在预测差幅度误差；10对evaluation有9对方向正确，仍有0.928的差值MAE。

**支持的解释：** 不能把这些病例的分解直接归结为参考误差趋零带来的各半退化。由c²−d²=e_qe_r，差值项均值较大只等价于所选病例平均两端误差乘积为负；它没有证明化学机制、哪种粗糙度特征有效、或者误差异号能够提前识别。训练内参考误差不小也不能证明训练乐观偏差不存在，因为没有这些参考的合法样本外误差对照。

**未验证假设：** 是否能用不含查询标签、且没有跨折泄漏的参考信息区分“变化学会/未学会”，以及这种区分能否超越普通RF或局部偏差统计改善接受名单，均未检验。本轮没有比较风险臂、归因原增强7维收益、检查原子忠实度或提供跨seed机制证据。15对虽端点不重复，仍不证明化学系列或模型误差独立，不作显著性与跨任务推断。

可靠性主线保留，当前三点几何/组件均衡/残差方差等停止或暂停结论不变，可信最终算法核心仍0。现在不值得继续堆公式或训练。下一最小研究动作应是**先审查查询与参考同时样本外的预测合同和最低成本**：必须排除查询折及相应参考数据的训练依赖，不能拼其他折OOF预测；如果只能重复基础局部残差统计，或必须大幅重训才可反证，就按方法缺口停止。此项尚未执行，且不会自动触发训练。

## 核验、异常和文件范围

26冻结输入hash、角色及15对名单、checkpoint历史绑定、30行SMILES次序、预测有限性、15查询重放、所有误差分解与聚合回放通过。独立标量复算最大逐对恒等式残差 `2.220446049250313e-16`。selfcheck1次、prepare1次、真实推理1次、脚本audit1次；独立标量审核尝试2次/成功1次。

首次独立标量审核命令遗漏UTF-8，Windows默认GBK读取含中文协议失败，算术检查尚未开始；显式UTF-8修正后通过。仅重做审核，真实推理没有失败、补跑或重启。CLI出现可选Triton缺失、GPU可用未使用、低worker及弃用提示，正常退出0；未安装软件或切换设备。详见[verification.json](verification.json)。

公开新增本目录diagnose.py、protocol.json、results.json、report.md、verification.json；仅追加task_plan.md、notes.md、docs/milestones.md、docs/research/reliability_publication_plan.md。逐分子输入、身份、标签、预测、误差及CLI日志留在忽略目录 `D:/GraphCliff-Pair/artifacts/query_reference_prediction_20261007/`。训练代码、checkpoint、原缓存、M56及历史报告/六图、两个原项目未修改。

```powershell
& 'D:/Tools/conda-envs/chemprop_baseline/python.exe' docs/research/query_reference_prediction_20261007/diagnose.py selfcheck
& 'D:/Tools/conda-envs/chemprop_baseline/python.exe' docs/research/query_reference_prediction_20261007/diagnose.py audit
# run禁止覆盖execution.json，不能用来自动补跑
```
