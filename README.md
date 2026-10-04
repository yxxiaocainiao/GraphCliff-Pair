# GraphCliff-Pair

基于 GraphCliff 编码器的参考分子差值回归研究项目。逐项验证跨分子 Cross-Attention、FPPool、动态加权 Loss，保留原短长程编码机制。

**状态：独立数据接口、三项模块与只输入 SMILES 的预测入口已实现，19 项契约测试与 16 组真实数据 smoke 已通过；正式验证正在运行，尚无三项机制的效果结论。**

首轮两任务三种子24次交互实验已完成并独立审计：[完整验证结果](docs/interaction_three_seed_results.md)。Cross-Attention在两任务上均未超过原GraphCliff或容量匹配的pair MLP，预定额外任务扩展为No-Go；继续完成全部54次已冻结消融，test尚未评估。

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
- [FPPool official code](https://github.com/shenwxlab/FPPool)：计划复用指纹分层池化，不自行重写算法。
- [SQRL paper](https://arxiv.org/html/2501.09103v1)：参考相对回归的任务定义；不是官方复现。
- [Siamese-Regression-Pairing](https://github.com/AstraZeneca/Siamese-Regression-Pairing)：参考近邻配对流程，是否借用具体源码需逐文件检查。
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：使用库内注意力运算。
- [Scientific Agent Skills](https://arxiv.org/abs/2609.00065)：实验设计及 PyG 接口的程序性指导。

三种子矩阵形成后运行 `python tools/summarize_results.py --runs <run-folders> --csv-root <csv-root> --phase interaction --output-prefix <report-prefix>`；完整消融使用 `--phase full`。工具先独立审计再汇总，拒绝不完整矩阵。

最终test入口为 `tools/freeze_validation.py` 与 `tools/evaluate_test.py`。只有完整78次验证及报告核对通过后才能冻结；冻结后核对所有权重、来源和数据身份。当前test尚未执行，不能用单种子或smoke跳过完整验证。

最终test完成后使用 `python tools/summarize_test.py --freeze <freeze.json> --test-output <test-folder> --csv-root <csv-root> --output-prefix <report-prefix>`。它核对完整78个模型、数据/权重/预测身份与结构行序，重新计算各指标并汇总三种子；不按test选模型，不在test上执行扩展门槛。当前只完成合成流程检查，真实端到端核对待test评估后进行。

当前完成24次交互及seed42的18次消融，共42次联合审计通过（[审计记录](docs/validation_through_seed42_audit.json)）。20项契约测试通过。Windows队列可用 `-ResumeAfterSeed42` 在重新审计后继续剩余36次；它拒绝重复训练及已有输出。完整78次报告与test仍待完成。

[seed42完整消融阶段表](docs/ablation_seed42_results.md)包含26组及全部条件差；仅单种子开发证据，不能替代完整三种子结论。

[42次已完成训练的补充诊断](docs/validation_diagnostics_42.md)包含预定最近邻参考标签基线与加权Loss的训练分布；逐轮极值已与history核对。
