# GraphCliff-Pair

## 2026-10-06 M31：当前状态

M31支持代理诊断完成：三任务低/高相似度组增强均改善，原始及尾部移除后0/3任务支持低组退化假设，结束本诊断。第三问题固定缓存没有同查询fold/full配对，不能判断规模效应；算法核心仍0，暂停扩展训练，发表目标未完成。 见[完整结果与缓存可行性](experiments/reliability_base/support_diagnostic_report.md)。M30及更早状态均作历史记录。

## 2026-10-06 M30：当前状态

M30零拟合目标失配诊断完成：三任务18组对原始/稳健MAE-MSE逆序均0；本项筛查未通过，结束该诊断，不训练平方目标、不补种子或扩大任务。该结果不证明其他问题或排序方法无效；M29仍0算法核心、暂停扩展，发表目标未完成。 见[完整结果与异常](experiments/reliability_base/objective_diagnostic_report.md)。下文M29及更早状态均作历史记录。

## 2026-10-06 M29：唯一当前状态

M29有界三问题审查完成，保留0个算法核心；当前直接改法缺明确方法增量或已观察瓶颈，暂停本选题扩展训练。M28正向信号保留，但置乱gate因4792缺一次重复仍未确认；不增加第19次、不运行seed43/44或30任务。发表目标未完成。 见[三问题审查与停止报告](docs/research/reliability_core_review_20261006/report.md)。后文较早的“当前状态”均为历史记录，不覆盖本节决定。

## 2026-10-06 M28：当前状态

M28预算内完成18次RF拟合，1次重放核验失败、修复后17次结果有效；234/244五置乱、4792四置乱，正式gate未确认。真实特征现有结果均优于置乱，但不称完整通过或算法创新。不扩大训练，继续有界三问题审查。 见[全部结果与异常](experiments/reliability_base/permutation_report.md)。

## 2026-10-06：完整计划已授权，M28运行前

可靠性算法创新主线、8—12周投稿材料目标和分阶段预算已确定。当前算法核心未定；先18次RF内同维度置乱，再三问题有界审查。旧队列暂停，未通过不扩大。见[唯一当前总计划](docs/research/reliability_publication_plan.md)，历史状态保留。

## 2026-10-06 M27：当前状态

M27零拟合简单对照通过：增强风险RF对作者式等权组合三任务改善2.67%/1.68%/0.37%，但方法特异性/跨seed/解释仍缺；当前只继续有限补证准备，0新fit/0test，旧队列暂停。下一步先固定同维度置乱对照，不叠加模块。作者特征seed固定0可候选复用，跨seed草案预算可降至36fit，尚未执行。 见[结果与边界](experiments/reliability_base/incremental_evidence_report.md)。

## 2026-10-06 M26：当前判断（优先于历史状态）

M26历史模块复核完成：实际退化、条件性收益、不完整实验及文献筛查已分开；5%仅为历史项目筛查规则。旧No-Go不改，旧队列继续暂停。唯一优先补证建议为三任务小幅正向的粗糙度风险增强，需先固定方法差异和新方案，本轮0训练/0test。目标为有可靠增量贡献的二区或核心论文，不保证录用。 见[完整复核报告](docs/research/module_pause_review_20261006/report.md)。

最终目标：形成可复现、有明确贡献的论文并投稿，争取中科院二区发表。研究范围允许限定到有证据支持的任务和条件，不要求彻底解决活性悬崖；三项机制是可删减的候选手段。具体投稿口径和证据要求见[项目计划](task_plan.md)。

**研究优先级：可复用底座上的算法方法论文。** [总体框架](docs/research/algorithm_framework.md)已更新为可靠预测方向，GraphCliff可选；核心尚未确定，不宣称新算法已完成，不恢复旧三板斧训练。

**当前有限基线阶段A已完成；粗糙度增强候选未过继续门槛，B不启动。** [执行计划](docs/research/algorithm_research_plan.md)已更新。

**M25最新状态：** M25完成：阶段A 30/30fit，0test；未通过项目筛查，停止粗糙度增强候选；不运行seed43/44，不追加模块/任务或调参救结果。 结果与原因分层见[阶段A报告](experiments/reliability_base/phase_a_report.md)。发表目标仍未完成；原项目/my_work/旧队列不变。

**M21历史状态：** M21：用户确认以成熟公开项目为底座，不再限定GraphCliff。MIT粗糙度作者代码已原样跑通3次RF小样本接口，128train/32validation，部署特征标签依赖及作者纯校准接口通过，官方test使用0；UNIQUE为已审查但未运行参考。进入有限基线方案准备，尚无新方法效果结论。见[底座验收](experiments/reliability_base/README.md)。发表目标不变，旧队列暂停。

**M20历史状态：** M20粗糙度方向差异审查完成：固定两仓库/14文件/9份Python静态核验；作者已有风险组合、GIN和粗糙度条件校准，UNIQUE已有误差模型，普通联合拒答亦有近邻。当前迁移/组合方案未形成算法贡献，保留0，不启动三任务训练；复核价值与算法贡献分开。见[差异审查](docs/research/roughness_method_review_20261005/report.md)。发表目标仍未完成。

**M19历史状态：** M19：重新核查单MCS匹配解释监督。3组合成控制/72次重编号通过；两任务48训练结构近邻对中43完成、5超时、0不同排名多重集充分证据，未达运行前结构门槛，停止该候选、保留0；无训练/test。见[结构审查报告](docs/research/mcs_mask_audit_20261005/report.md)。M18及旧No-Go保留，创新核心仍空缺，发表目标未完成。

**M18历史状态：** 有界G1关键近邻审查、G2三个问题和G3筛查完成：非共有幅度监督、共有上下文差值、高低频交互加动态Loss均存在直接公开重叠，保留0个候选，暂停本轮选题；不进入G4，不恢复旧队列。发表目标未完成。见[近邻审查与停止报告](docs/research/neighbor_audit_20261005/report.md)。

[文献边界初表](docs/research/literature_29_20261005/method_boundary.md)保留历史证据范围；本轮只做有界深核验，不代表全库穷尽。

历史候选框架为基于 GraphCliff 编码器的参考分子差值回归研究项目。逐项验证跨分子 Cross-Attention、FPPool、动态加权 Loss，保留原短长程编码机制。

**状态（2026-10-05）：旧实验目标已暂停，后台训练与自动收尾均已停止。独立数据接口、三项模块和预测入口已实现，36项测试与16组真实数据smoke已通过。完整三种子消融、78模型权重重放及最终test未完成。**

首轮两任务三种子24次交互实验已完成并独立审计：[完整验证结果](docs/interaction_three_seed_results.md)。Cross-Attention在两任务上均未超过原GraphCliff或容量匹配的pair MLP，预定额外任务扩展为No-Go；用户已暂停余下消融，test尚未评估。

[阶段性证据解读](docs/development_evidence_summary.md)分别说明三个模块已有的支持、反证和未解决问题。用户选定[新诊断目标及提前暂停条件](docs/diagnostic_goal.md)，本轮手动完成D0/D1/D2并在D2按预设规则提前暂停：[阶段报告与暂停原因](docs/diagnostics/diagnosis_report.md)。D3和新增训练未启动，系统旧目标仍保持暂停。

[D1参考相似度诊断](docs/diagnostics/reference_results.md)发现234/global−direct在最高相似度层反而更不利；[D2构成检查](docs/diagnostics/composition_results.md)显示其子组证据未达到继续门槛。分析没有确认退化原因，保留所有反例和样本数。

[已有基线复用诊断](docs/diagnostics/baseline_reuse_234/results.md)已完成：234任务14组验证预测逐行对齐；原GraphCliff在这些既有对照中表现最好，但预算不完全匹配，尚未形成值得新增训练的改进假设。旧my_work基线直接复用，不重建、不读取旧test预测。

[有限难例核查](docs/diagnostics/hard_case_audit_234/results.md)已完成：234高活性尾部有三个seed重复低估，但已有分子不平衡回归文献覆盖相关机制；尚未形成新方法贡献，诊断到此结束，不据此追加Loss或训练。

[三任务固定分层误差筛查](docs/diagnostics/matched_cliff_20261005/results.md)已完成：复用27组已有验证预测，只有MLP/3979满足两种设计；未达到跨模型、跨任务继续门槛，No-Go。不据此启动新Loss训练，论文目标仍未完成。

[历史实验台账及框架纠错](docs/research/framework_review_20261005/inventory.md)：原GraphCliff直接FPPool已完成3979三seed，未过旧门槛；另有残差/跨任务/归属/解释试验。不重复已失败路线；可解释性为最终论文必需验收。

**[最新方法审查：No-Go](docs/research/framework_review_20261005/decision.md)**：本轮保留0个新训练候选，停止继续当前三模块/迁移路线；发表目标未完成。[可解释性必需验收与已有案例](docs/research/framework_review_20261005/explainability.md)、[完整阶段/异常日志](docs/research/framework_review_20261005/milestone_log.md)已保存。旧训练队列及自动收尾继续暂停。

终极目标与进度见 [task_plan.md](task_plan.md)，来源与决策见 [notes.md](notes.md)。本仓库公开，只发布源码、配置、来源记录和验证摘要。数据与训练权重不随仓库发布。

完整复现顺序、数据哈希及验证后测试步骤见[复现说明](docs/reproduction.md)。

预测契约：`delta(query, reference) = y_query - y_reference`，`y_hat = y_reference + delta_hat`。参考来自同任务训练集，按结构选择；查询活性不传入模型。

## 使用

Python 3.10+。先按设备安装匹配 PyTorch 2.7 的版本，再在本 checkout 中运行：

```powershell
python -m pip install -e .
python tools/fetch_fppool.py
python -m unittest discover -s tests -v
python -m graphcliff_pair.data --csv "D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv" --output artifacts/prepare/CHEMBL234_Ki
python -m graphcliff_pair.train --config configs/smoke.json --csv-root "D:/GraphCliff-main/benchmark_data" --output artifacts/my_smoke
python tools/verify_smoke.py artifacts/my_smoke --report docs/my_smoke_validation.json
```

将示例数据路径替换为自己的 MoleculeACE CSV 目录，至少包含 smiles/y/split/cliff_mol。输出目录必须全新，拒绝覆盖和静默重用模型。安装采用 editable checkout，因为固定来源和外部源码相对项目根目录加载。

`configs/smoke.json` 是 2 任务×8 组×3 轮的基础检查，每个任务只用 64 个训练查询和 64 个验证查询；参考池仍是完整训练集。它不能用于效果排名。`configs/development.json` 是完整预算候选，正式运行前须冻结阶段配置，不默认启动全部组合。

正式配置为 `configs/interaction_seed42.json`、`interaction_seed43_44.json` 与 `ablation_seed42/43/44.json`；设计见 [交互冻结记录](docs/interaction_preregistration.md) 和 [三机制完整消融](docs/three_mechanism_preregistration.md)。训练按队列串行执行，各阶段审计通过后继续，错误即停止；本机 Windows 入口为 `tools/run_queue.ps1`。共24次交互训练与54次后续消融，是否完成以实际输出和独立审计为准。

实现与权重公式见 [docs/design.md](docs/design.md)，提交记录见 [docs/milestones.md](docs/milestones.md)。公开仓库不包含数据或权重，数据应自行从 [GraphCliff 官方仓库](https://github.com/dmis-lab/GraphCliff) 获取并核对 SHA256。FPPool 通过固定版本下载脚本获取，来源说明见 [docs/sources.json](docs/sources.json)。

在本机已验证的环境为 torch 2.7.1+cu128 / PyG 2.6.1 / RDKit 2025.03.6。保留上游 best-effort 确定性策略，不保证 GPU 跨硬件逐位相同；attention 使用 PyTorch math backend。

## 独立预测

查询 CSV 只需 `smiles` 列，附带的 `y` 不读取。参考库使用训练时同一份 CSV，并核对文件哈希与训练行号；仅加载这些训练行的活性值。

```powershell
python -m graphcliff_pair.predict --run-dir artifacts/my_smoke/CHEMBL234_Ki/seed42/cross_fp_dynamic --source-csv "D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv" --query-csv queries.csv --output artifacts/query_predictions.csv
```

输出保留输入顺序，包含预测活性、训练参考行号、结构相似度和预测差值。`direct` arm 的 `delta_prediction` 为空；它仍输出结构选出的参考信息作审计，但原模型预测不使用参考标签。

完成正式队列后使用 `python tools/audit_runs.py --runs <run-folder> --csv-root <csv-root> --output <audit.json>` 独立核对。验证内部 Morgan 相似对的差值指标只是辅助诊断，不等于官方 Cliff RMSE。

## Sources

- [GraphCliff official code](https://github.com/dmis-lab/GraphCliff)：复用分子特征、编码器及原读出。
- [FPPool official code](https://github.com/shenwxlab/FPPool)：固定版本按需下载，已接入指纹分层池化，并核对适配后输出与梯度等价性。
- [SQRL paper](https://arxiv.org/html/2501.09103v1)：参考相对回归的任务定义；不是官方复现。
- [Siamese-Regression-Pairing](https://github.com/AstraZeneca/Siamese-Regression-Pairing)：参考近邻配对流程，是否借用具体源码需逐文件检查。
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：使用库内注意力运算。
- [Scientific Agent Skills](https://arxiv.org/abs/2609.00065)：实验设计及 PyG 接口的程序性指导。

三种子矩阵形成后运行 `python tools/summarize_results.py --runs <run-folders> --csv-root <csv-root> --phase interaction --output-prefix <report-prefix>`；完整消融使用 `--phase full`。工具先独立审计再汇总，拒绝不完整矩阵。

最终test入口为 `tools/freeze_validation.py` 与 `tools/evaluate_test.py`。只有完整78次验证及报告核对通过后才能冻结；冻结后核对所有权重、来源和数据身份。当前test尚未执行，不能用单种子或smoke跳过完整验证。

最终test完成后使用 `python tools/summarize_test.py --freeze <freeze.json> --test-output <test-folder> --csv-root <csv-root> --output-prefix <report-prefix>`。它核对完整78个模型、数据/权重/预测身份与结构行序，重新计算各指标并汇总三种子；不按test选模型，不在test上执行扩展门槛。当前只完成合成流程检查，真实端到端核对待test评估后进行。

## 当前验收与运行记录

24次交互及seed42的18次消融，共42次联合审计通过（[审计记录](docs/validation_through_seed42_audit.json)）。当前36项测试覆盖模型接口、指标汇总、冻结重放、测试评估生命周期和恢复契约；通过这些测试不代表正式78模型已经全部验收。历史测试数量与提交见[里程碑](docs/milestones.md)。

[seed42完整消融阶段表](docs/ablation_seed42_results.md)包含26组及全部条件差；仅单种子开发证据，不能替代完整三种子结论。

[42次已完成训练的补充诊断](docs/validation_diagnostics_42.md)包含预定最近邻参考标签基线与加权Loss的训练分布；逐轮极值已与history核对。

最终freeze要求全部保存权重的validation重放，须使用原训练设备类型。最终test通过原子状态保存及唯一输出登记处理中断和重复执行。正式全78重放及真实test尚未完成。

seed43在14/18组完成后发生CUDA非法指令，随后在新目录复用14完整组并从头重训剩余组。用户暂停时，首个重训组完成至第46轮，尚未形成该组正式完成记录；已停止恢复队列并保留现场。当前恢复目录不能直接重新运行以覆盖结果；未来如另行恢复，须重新核对完整组与中断组。原失败目录与恢复目录不能重复计数。

本机 `tools/finish_frozen_experiment.ps1` 原用于训练完成后串行收尾，现已随用户暂停停止。入口已通过前置与并发检查，但正式78模型重放及test未执行；不能将工具实现完成等同于实验验收完成。

## 最新授权试验：LongPoly替换（2026-10-05）

新增独立逐层替换试验：保留ShortGINE、门控、残差、原池化和MSE。两个任务×三组、seed42共6组正式训练及GPU重放审计已完成；未通过预注册门槛，停止扩展。原核心代码及旧队列不改，测试标签未解析，未评估测试集。[中文结果与停止理由](experiments/long_branch_swap/结果报告.md)。

## ACA-MSE迁移已提前暂停（2026-10-05）

原GraphCliff保留，仅借用官方ACA表征Loss。第一个任务的三种子六组Cliff均值恶化3.01%、Overall恶化4.55%，按用户止损要求触发固定否决条件后提前暂停；剩余12组未完成，未评估测试集。[六组审计与停止报告](experiments/aca_pilot/results.md)。当前没有活跃训练；不把借用Loss当原创或二区发表成果。

实现默认优先复用已审计的本地与GitHub项目，必要时核对原论文；新增工作限定为薄适配和贡献必需的差异。[三个官方候选与复用边界](docs/research/github_reuse_candidates_20261005.md)已核对，尚未启动新性能试验。
