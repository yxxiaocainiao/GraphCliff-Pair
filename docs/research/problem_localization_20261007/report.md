# M43：可靠性主线是否保留与唯一问题定位

2026-10-07。身份：接续M40—M42的有界问题讨论，非训练协议、非新算法验收。起点本地及远端main均为e12fd1d3e20c9c4c6eeed9c918dffa2a3b0af28d，初始工作树干净。

## 结论与推荐

**保留可靠性研究主线和Chemprop底座；保持具体候选停止；本轮仍保留0个可信新算法核心。** 保留主线是有界选题判断，不是恢复训练授权。推荐仅讨论一个问题：**在点预测完全相同的条件下，折外风险学习与部署使用不同合法参考集，是否会改变接受排序并造成可复核的选择损失？**

选择它的理由：现有协议确有reference差异，影响尚未被同查询设计隔离；可以不训练点模型或风险模型就做第一次辨别。它优于立即加新模块的地方在于能直接淘汰一个具体动机。它目前的弱点也明确：没有已观察的损失证据，没有经近作比较成立的校正运算。因此，这是**待定位问题，不是可信算法候选**，不将未知升格为论文核心。

不建议再以平方目标、低相似度门控或组件均衡作默认下一方法。M30/M31已限制前两项的直接动机；M38/M39限制第三项的具体运算。也不因此恢复旧FPPool、Cross-Attention、动态Loss或ACA。

## 证据与判断分开

| 层次 | 当前可说什么 | 当前不能说什么 |
|---|---|---|
| 观察事实 | 三任务seed42，增强7维RF对普通5维RF六接受率RMSE均值改善2.08%/2.49%/3.76% | 独立确认、最终算法创新、点预测RMSE改善 |
| 观察事实 | 组件均衡相对同9维同支持量去核心：234 +0.768%，244 -2.577%，4792 -1.273%；宏均值 -1.125% | 所有组件方法或粗糙度普遍无效 |
| 支持的判断 | M39九例/180随机掩蔽未给出推翻强消融的证据；该候选停止合理 | 九例证明不存在任何适用条件，或已完成化学机制解释 |
| 观察事实 | M30固定18组对无MAE/MSE逆序；M31六个低/高相似度组增强均改善 | 所有排序目标或支持估计都无价值 |
| 观察事实 | OOF与full参考数不同，M31九个fold/full比较的同查询配对数均0 | 已证实reference差异导致失准 |
| 支持的判断 | 现有底座和风险信号使可靠性值得保留一次有界问题讨论 | 值得继续多seed/30任务扩训或保证可投稿 |
| 未验证假设 | 仅reference变化可能影响风险排序及接受损失 | 退化原因已确认为参考迁移、过拟合或噪声 |

数值来源：[M25](../../../experiments/reliability_base/phase_a_report.md)、[M30](../../../experiments/reliability_base/objective_diagnostic_report.md)、[M31](../../../experiments/reliability_base/support_diagnostic_report.md)、[M38](../component_roughness_pilot_20261006/report.md)、[M39](../component_roughness_cases_20261006/report.md)。其中宏均值改善按聚合误差比值定义，不能与三任务相对改善的简单均值混用。离散六点均值不改名AURC。

M26当时的“粗糙度优先补证建议”已被后续M29—M40的具体审查和停止决定收紧，不能当作当前扩训许可。[M40](../publication_assets_20261006/report.md)仍是资产与方法缺口依据；[M41](../../reports/teacher_brief_20261006/report.md)与[M42图册](../../figures/graphcliff_changes_20261006/gallery.html)是汇报资产，不增加有效性证据。

## 一个小而明确的问题

现有风险模型可写为：

`r_R(x) = g(f_full(x), mol_size(x), rf_var_full(x), h(x; R))`

其中h仅包含合法训练reference计算的nn_sim、local_dens、nbr_disp、sali_mean；不使用查询y。g在OOF误差上学得；OOF的f、辅助RF及reference均来自各折训练子集。full查询则使用全fit模型及全fit reference。

| 任务 | full reference | 三折reference | calibration / evaluation |
|---|---:|---|---|
| CHEMBL234_Ki | 1755 | 1184 / 1147 / 1179 | 426 / 448 |
| CHEMBL244_Ki | 1462 | 970 / 977 / 977 | 375 / 402 |
| CHEMBL4792_Ki | 702 | 460 / 472 / 472 | 179 / 187 |

这些是M31/M40已经核对的数量，本轮未重新加载真实样本重算。关键不是比较不同角色的特征直方图，而是在**同一查询、同一点预测、同一风险权重**下比较R_full与已有R_fold。两个reference均不含calibration/evaluation查询，使用其合法训练标签。

第一步只隔离h的reference效应；prediction、mol_size、rf_var固定。特别地，rf_var来自作者辅助活性RF的树预测，不是Chemprop ensemble方差。不能为了更新rf_var去调用作者process，因为它会重新fit辅助RF。固定rf_var意味着这只是部分受控输入干预，不是完整fold部署流水线，不等于证明训练规模效应或风险目标迁移。

## 与最接近方法的关系：有问题差别，尚无新算法差别

“最近”分机制接近和时间较新；本轮是限定核查，非穷尽SOTA综述。

| 原始来源 | 已有内容 | 本问题的边界与贡献缺口 |
|---|---|---|
| [DEUP，2023版本，§3.1/3.2](https://arxiv.org/html/2102.08501v4) | 用样本外误差训练误差预测器；讨论训练数据变化带来的非平稳性，用密度/方差等数据集相关特征缓解 | 不能把“误差学习+reference感知”称首次提出；本项目仅缩小到固定预测器下的分子邻域参考变化，尚无优于该思路的新运算 |
| [UNIQUE官方方法表](https://opensource.nibr.com/UNIQUE/indepth/available_inputs_uq_methods.html) | 距离、密度、ensemble方差与RF/LASSO误差模型 | 拼入参考规模或稳定性特征仍是常规误差模型；必须与同信息RF比较，不能只胜5维RF |
| [粗糙度作者当前仓库](https://github.com/krishnatheaverage/qsar-landscape-roughness) | README已覆盖局部粗糙度、无活性标签变体、组合和粗糙度条件保形校准 | 本地固定版本nbr_disp/SALI依赖reference活性；不是作者全部方案。简单组合或条件校准已重叠，正式发表身份仍未知 |
| [jackknife+/CV+](https://arxiv.org/abs/1905.02928) | 样本外残差与相应预测器配对，用于预测区间 | 本问题只变邻域参考、固定点预测，目标是接受排序；不同于CV+，但“不同”本身不证明新颖性，也不继承区间覆盖保证 |
| [JCIM较新分子UQ研究](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02381) | 出版方索引明确为分布变化下分子UQ比较，使用UNIQUE，含模型/特征输入 | “分子误差模型+数据变化”不是空白；本轮ACS全文403、PMC验证拦截，只按索引信息作重叠提醒，不推断其未做reference控制 |

DEUP本轮读取HTML相关方法节，不宣称全篇精读；UNIQUE读取官方文档及本地固定RF源码；粗糙度读取当前README及旧固定源码。当前README与历史固定版本必须分开，不根据标题变化推断算法发布日期。CV+本轮摘要与M29既有审查分层使用，无新增全文核验。

**差异审查结论：** 尚未找到可辩护的新核心。“reference匹配”“参考子采样平均”或“增添规模特征”目前只能作为已有思想下的基线/辨别手段，不能命名一个新方法就宣布解决创新缺口。

## 可复用代码与最低成本反证设计（仅讨论，未执行）

复用现有仓库，不建项目、不安装依赖：

- [run_phase_a.py](../../../experiments/reliability_base/run_phase_a.py)：GENERIC/EXTRA定义、OOF/full身份、metrics接受排序与校准工具。不能直接运行main；prepare/run都会触发模型操作。
- [run_component_pilot.py](../../../experiments/reliability_base/run_component_pilot.py)：已存在的feature_function只提取作者纯featurize；保存模型、round_trip读取和reference重放做法可借用。不能运行其prepare/run来做本问题，它有组件特征或拟合路径。
- 私有固定作者build_features.py：MIT，91522b8b81a52d5c95ae216d01a00cbbf0c66209，复用radius2/2048、k=10及原统计。process内含fit，不能当纯特征函数调用。
- 私有UNIQUE RF源码：BSD-3-Clause，c6d65b9c63102bc18e68c25c98d76e24650c3e4a；本地源码及LICENSE.md已读，官方固定LICENSE.md已核。整套UNIQUE未运行，不为一次检查新接全流水线。
- 现有M28保存增强7维风险RF与对应full缓存可作为首选；正式检查前核对保存路径、配置、特征顺序、hash和原分数重放。保存模型是否满足此检查合同本轮未加载核验，缺失就报告阻塞，不能暗中重新fit。

建议一次有限检查设计如下，**未冻结成可执行协议，未执行**：

1. 固定234/244/4792和已有三个reference折，使用全部现calibration/evaluation查询；不新增随机划分、不换任务、不选择最有利折。
2. 先做y盲检查：原full统计及风险重放；对同一查询用R_full和三个R_fold重算四个h特征，固定其余三特征与g。记录风险差、排序变化、六接受率集合交并比。恒等参考必须完全重放；更改查询y不得改变特征或风险。
3. 只有前一步完成并保存分数后才读evaluation误差作描述性评价；主指标沿用六接受率RMSE均值，列全部任务/折/coverage，100%接受时误差必须相同。不能先看误差选择reference。
4. 如需一个可部署对照，仅固定三个fold-reference风险的算术均值，比较原full-reference风险；它是简单reference边际化基线，不是新算法。分别评价每折只是诊断，禁止按查询误差选折。首轮不增加校准/区间实验或新训练。
5. 不反复改变比例、k、seed、阈值或加更多子采样。拟议预算：0任何fit、0点模型推理、0官方test、0搜索；三任务共8个风险predict批次/任务（full重放+三个折，各cal/eval），24调用、8068行风险输出，CPU wall上限10分钟；完整统计及核验计入此上限，达到即停并记未完成。工程是否在10分钟内完成未知，不伪称实测。

8068 = 4 × (874 + 777 + 366)，平均组合无需额外predict。独立指标核验只读取保存分数，不暗加模型调用。所有逐分子分数及特征保留忽略目录；公开仅聚合/hash。此预算是讨论建议，**不是本轮执行次数，也不是新增训练额度**。

### 如何被反证，什么结果才改变判断

- 若改变reference后接受集合/损失实质不变，当前“reference变化损害排序”的动机没有支持；结束该入口，不改找另一比例。
- 若分数或集合变动，但没有可复核损失，最多证明敏感性，不能据此开发校正模块。分数整体缩放却不改排序尤其不构成排序算法动机。
- 若简单均值就有收益，先承认标准子采样/平均基线足以解释，不将它改名为创新。收益不自动授权多seed或训练。
- 若有同查询可复核失效且简单基线不足，才值得定义一个针对该失效、与DEUP/UNIQUE/普通子采样不同的具体运算，以及同信息去核心消融。仍需新独立条件验证，已见缓存不作确认集。
- 阴性只结束固定h/reference入口；没有同步改变point model与rf_var，不能否定完整模型规模迁移。阳性也不能证明这是M38退化原因。

不设事后3%/5%录用阈值，不用九例或一任务涨点投票；讨论看的是具体失效及方法可识别性。若本次限定检查也无法形成可定位增量，就收束该选题轮，不再接一串预检。本轮并未自动授权它运行。

## 解释验证应围绕新增运算

此问题的首层解释是可审计证据链：同一查询 → reference删去了哪些合法近邻 → nn_sim/local_dens/disp/SALI怎样变化 → 风险分数及接受身份如何变化。恒等reference、查询y不变性、固定预测和100%误差一致性是必要接口检查。

只有未来保留新运算，才比较其声称重要的reference证据与同规模随机删除；随机次数和案例规则先冻结，成功/失败/一般全部展示，组件依赖按组报告。还须证明“减小风险变动”与“更准确接受排序”相连，不能把不敏感本身当优点。

最终模型仍需独立的Chemprop原子/子结构归因、重要区域对同规模随机遮罩、跨训练seed稳定性；reference敏感性不能代替这些要求。训练reference扰动不等于化学干预或化学因果，本轮不生成热图，不宣称完整忠实度。

## 本轮交付、限制及三层复核

本轮只阅读计划/报告/源码、查官方来源与写讨论；0fit、0模型推理、0真实特征重算、0新角色划分、0test标签读取、0搜索参数。未复核历史359项资产或重算历史预测，不把过去验收计为本轮结果。

信源复核：报告数值与M25/M38/M31/M40聚合相符；网页证据区分源码/README/摘要/索引，失败保留。逻辑复核：候选停止不外推主线，受控输入敏感性不冒充因果或新算法。未知复核：效应、校正公式、跨seed、强ensemble、忠实度和投稿可行性继续未知，可信新核心数为0。

异常：expression-skill指定路径缺失，标准.codex/.agents递归查找也未找到它或planning-with-files，复用现有task_plan/notes；未安装技能。早期批量阅读输出有截断，关键报告分次重读。UNIQUE LICENSE初次用错误文件名404，定位LICENSE.md后官方固定版本核对成功；误猜models.py路径未获取，改读已有本地真实RF源码。ACS403、PMC/OpenReview验证拦截和CV+ HTML访问失败保留；DEUP改用arXiv HTML相关节。无训练失败或中断，不伪装网络失败为方法反证。

改动仅当前仓库文档：task_plan.md、notes.md、docs/milestones.md、docs/research/reliability_publication_plan.md及本目录report.md/evidence.json。原GraphCliff/my_work、代码、模型、Loss、配置、数据、预测、权重、历史报告及六图不改。核验记录和本地输入SHA见[evidence.json](evidence.json)；验证后按已授权里程碑commit/push，核对远端。

下一步最小选择：讨论是否值得做上述一次reference受控检查；若要求立即给出投稿级算法，本轮答案是没有，缺口是“实际失效证据+可识别新运算”，继续训练不能补这个缺口。
