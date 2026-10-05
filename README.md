# GraphCliff-Pair

最终目标：形成可复现、有明确贡献的论文并投稿，争取中科院二区发表。研究范围允许限定到有证据支持的任务和条件，不要求彻底解决活性悬崖；三项机制是可删减的候选手段。具体投稿口径和证据要求见[项目计划](task_plan.md)。

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
