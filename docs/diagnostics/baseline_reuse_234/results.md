# CHEMBL234_Ki：已有基线复用与验证误差诊断

2026-10-05。结论：已有基线可以复用，无需重新建立。逐行对齐后，近期GraphCliff普通MSE对照的三个训练种子均在Overall和Cliff指标上优于这里的既有SVM、D-MPNN及匹配种子的GCN/GAT/MLP。这个任务上的主要问题是新机制尚未超过原GraphCliff，而不是缺少能运行的强基线。本轮不新增训练、不调整模型或Loss、不读取官方test标签或旧test预测。

## 输入身份与来源

- 旧基线：`D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines`；当前项目：`D:/GraphCliff-Pair`；原数据：`D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv`。三个目录保持区分。
- 两套划分函数均为split_seed=42、validation fraction=0.1；2632训练分子、292验证分子，其中128个cliff。训练/验证行号、SMILES、标签和cliff逐项一致；SVM训练行、Chemprop保存train/val输入、九组神经模型训练/验证行均核对一致。
- 原CSV和旧processed文件仅读取分区身份后，再跳过官方test行读取标签。文件全文SHA256用于身份检查，不解析test活性。
- 标签为`y=−log10(nM)`；全部指标在同一尺度上计算。GraphCliff保存标签为float32，统一使用源float64标签重算，数值较原报告约有1e−8差异，排序未变。
- GraphCliff使用最新ACA试验的alpha=0普通MSE三组完整对照，而非从历次GraphCliff运行中挑最高分。模型核心不变；alpha=0路径仍计算零权重辅助项统计，不能用其耗时代表优化过的纯MSE训练。
- SVM：原MoleculeACE特征代码、原pickle权重，半径2/1024位ECFP；确定性模型只有一次预测，不伪造三种子。
- D-MPNN：官方Chemprop 2.2.3预测入口加载原best.pt，仅推断292验证SMILES。CLI参数名称test_path是预测输入名称，本轮实际输入只含validation。
- GCN/GAT/MLP：直接读取既有42/43/44验证预测。本轮未重新执行这九个权重的前向重放。

## 验证结果

| 模型 | 运行数 | Overall RMSE | Cliff RMSE | Noncliff RMSE |
|---|---:|---:|---:|---:|
| graphcliff | 3 | 0.590979 ± 0.011944 | 0.639340 ± 0.005910 | 0.550182 |
| svm | 1 | 0.619922 | 0.660906 | 0.585946 |
| mlp | 3 | 0.649018 ± 0.014253 | 0.663172 ± 0.011950 | 0.637520 |
| chemprop | 1 | 0.706142 | 0.725707 | 0.690486 |
| gcn | 3 | 0.861973 ± 0.010753 | 0.877512 ± 0.012913 | 0.849597 |
| gat | 3 | 0.918243 ± 0.005768 | 0.928748 ± 0.012513 | 0.909903 |

三个运行的均值±样本标准差仅表示固定划分下训练随机性；SVM和Chemprop为单运行，没有标准差。逐seed数据见validation_metrics.json。

这不是等预算算法排名：GraphCliff为最多100epoch、batch32、约602万参数；旧GCN/GAT/MLP为50epoch、batch64、默认超参数，D-MPNN为单seed50epoch。全部是验证选模，但训练预算、设备与环境并不相同。当前数据支持本地候选筛选，不支持“GraphCliff普遍优于所有方法”或论文录用判断。

## 逐分子错误：先看最接近的SVM

以下统一用GraphCliff seed42，随机基线也使用seed42；没有拟合融合模型或按标签构建oracle预测。

- 128个cliff分子中，SVM在55个分子上绝对误差更低；总体292个分子中为125个。逐分子获胜数不是可直接识别的推断规则。
- 各自绝对误差最高的20% cliff分子取ceil(128×0.2)=26个，其中18个重合，Jaccard为0.5294；cliff有符号残差相关系数0.8874。
- GraphCliff−SVM的cliff平均平方误差差为−0.036383；删除绝对误差差最大的一个分子后为−0.053619，优势方向保留。该删除检查是敏感性描述，不是显著性检验或因果证明。
- 其余四种模型全部比较也已保存，包含不利于GraphCliff的逐分子结果，不只保留有利案例。
- `cliff_case_ranking.csv`保存全部128个cliff分子按GraphCliff−SVM平方误差差排序；`shared_hard_case_ranking.csv`按两者绝对误差的较小值从大到小排序，定位共有难例。附上已保存的最近训练分子行号和结构相似度。排序仅用于下一步化学核查，不能据此宣称某个结构机制成立。

结论：SVM确有局部更好的分子，但高误差分子重叠和残差相关也较高。目前没有证据表明简单融合一定改善，也没有形成值得立即训练新模块的具体化学假设。

## 来源核对与限制

- SVM/Chemprop权重与历史哈希一致；九组GCN/GAT/MLP的权重、预测、配置、history、metrics字节与历史manifest一致。
- 发现旧`models/neural.py`当前哈希与训练manifest不同，详见historical_binding_audit.json。未修改或恢复旧脚本。旧保存产物可用于本次比较，但不能称为当前旧源码的严格训练复现。
- 标准库独立复算14组×3个RMSE共42项、10条错误重叠及相关系数，最大RMSE差1.11e−16；75个输入文件的审计后SHA256未变。没有声称对整个旧目录逐文件验收。
- SVM运行环境与旧config记录一致；Chemprop当前主要包版本与旧environment/chemprop_pip_freeze.txt一致，具体版本见chemprop_runtime.json。与GraphCliff及旧CPU神经基线的环境不相同。
- 同一开发验证集已反复用于方案筛选，本轮是探索性诊断，不能当作独立确认实验。旧基线30任务报告已有历史test评估；本轮没有读取其测试成绩来选模型，后续论文应如实披露历史评估范围。

## 决策与最小下一步

完成本轮复用核对，旧训练保持暂停。不扩展30任务，不启动direct+FPPool、融合或新Loss。下一步仅核查排序中的少量共有难例与SVM占优案例，区分具体结构变化、训练邻居覆盖和可能的数据问题；没有可检验、可跨任务确认的规律，就暂停这项诊断，不为涨点追加搜索。化学案例不能从单任务事后排序直接推广。

## 输出与复算

本地完整输出：`D:/GraphCliff-Pair/artifacts/baseline_reuse_234_20261005`（Git忽略）；公开输出：本目录的报告、汇总JSON和哈希审计。逐分子SMILES/预测、原数据和权重不发布。

```powershell
python -B tools/reuse_baselines.py prepare --baselines "D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines" --project "D:/GraphCliff-Pair" --data "D:/GraphCliff-main/benchmark_data" --output "D:/GraphCliff-Pair/artifacts/baseline_reuse_234_20261005"
python -B tools/reuse_baselines.py analyze --baselines "D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines" --project "D:/GraphCliff-Pair" --data "D:/GraphCliff-main/benchmark_data" --output "D:/GraphCliff-Pair/artifacts/baseline_reuse_234_20261005"
python -B tools/verify_baseline_reuse.py artifacts/baseline_reuse_234_20261005
```

上述analyze复用已导出的验证预测。缺少SVM预测时，用旧SVM解释器运行同一工具的svm模式。缺少Chemprop预测时，用旧Chemprop解释器调用官方predict入口，输入chemprop_validation_input.csv，输出chemprop_validation_raw.csv，加载原checkpoints/chemprop/CHEMBL234_Ki/best.pt；smiles_columns=smiles、num_workers=0、accelerator=gpu、devices=1。Windows须设置PYTHONIOENCODING=utf-8；首次GBK进度显示失败已重试完成，未改模型。先prepare核对身份再导出预测。

新增两个工具及本报告/汇总，更新根README、计划、notes、milestones及文献建议状态。未修改原GraphCliff-main、my_work基线、当前模型/Loss/训练配置或既有训练产物。
