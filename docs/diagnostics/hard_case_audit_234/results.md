# 有限难例核查：高活性尾部偏差有信号，尚未形成论文方法贡献

2026-10-05。已完成固定的五加五难例核查，去重后8个验证cliff分子。本阶段结束，不继续追溯所有难例，不新增训练。当前唯一可保留的候选问题是：高活性尾部预测是否系统性收缩，以及它与cliff指标的关系；这仍是探索问题，不是已确认的新机制。

## 发现和反例

- 8个案例均在2632训练分子的活性最小/最大范围内，不能直接称为标签范围外推；8个案例的GraphCliff三个seed误差符号一致，但“方向重复”不证明误差原因。
- 5个案例活性高于训练q90。其余3个不在高尾部，因此尾部不能解释所有案例。
- 8个案例与所选最近训练分子均非相同连接结构，未发现特征完全碰撞。在官方train开发池中，亦未发现同连接结构、不同立体SMILES且活性跨度至少1 log单位的组。这只淘汰了当前数据上这一种解释，不证明立体信息不重要。
- 本地GraphCliff原子特征没有显式原子手性字段，键特征包含立体信息；Morgan指纹本地配置未开启chirality。缺少某字段与某个错误的原因不能画等号。RDKit官方文档说明Morgan可选择加入原子手性，[RDKit Book](https://rdkit.org/docs/RDKit_Book.html)。

## 高活性尾部：重复偏差与边界

分位数仅由训练行计算：q10=−3.204120，q90=−0.113943。y=−log10(nM)，因此y更大表示活性更强。以下组别按真实validation y划分，只是评估描述，不能作为推断时可用的路由标签。

| seed | 高尾cliff数量 | Cliff RMSE | 平均预测−真实 | 删除本组最大绝对误差后平均偏差 |
|---|---:|---:|---:|---:|
| 42 | 20 | 1.014477 | -0.685458 | -0.579051 |
| 43 | 20 | 1.015827 | -0.588591 | -0.460369 |
| 44 | 20 | 0.952178 | -0.474649 | -0.353383 |

SVM的高尾cliff RMSE为0.977407，平均偏差−0.714435。GraphCliff seed44在该组比SVM好，seed42/43比SVM差；不能声称SVM在高尾部一致占优。低尾cliff仅9个、中间区间99个，完整总体与cliff分组均已公开，不隐藏反例。删除单一极端样本后三个seed仍低估，但剩余各19个，不能当作充分的跨任务确认或显著性结果。

高尾cliff的真实标签用于划分评估组，若模型训练或推断直接用查询真实活性选择尾部专家，将造成标签泄漏；任何后续方案必须只依靠训练或推断可用信息。

## 与发刊目标有关的文献核对

1. ICML2021的[Delving into Deep Imbalanced Regression](https://proceedings.mlr.press/v139/yang21m.html)已提出标签与特征分布平滑；标签稀疏区域重加权不是本项目新概念。
2. CVPR2022的[Balanced MSE](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html)已有针对不平衡回归的损失设计。不能把GraphCliff加一个现成Loss直接包装成原创。
3. 2026的[Toward Imbalanced Molecular Property Regression: A Benchmark Study and Interval-Aware Mixture of Experts](https://pubs.acs.org/jcisd8/article/doi/10.1021/acs.jcim.6c01847/5350961/Toward-Imbalanced-Molecular-Property-Regression-A)，DOI 10.1021/acs.jcim.6c01847，已直接研究分子回归的稀疏标签区域，设计区间感知专家机制。官方ACS检索结果可核对方法与题名；直接页面403，PMC全文入口触发验证，未声称读完整全文或复现其方法。
4. [Rationalizing the Formation of Activity Cliffs in Different Compound Data Sets](https://pmc.ncbi.nlm.nih.gov/articles/PMC6644420/)已研究活性分布与结构关系对cliff形成的共同影响。“活性分布与cliff有关”本身也不是新发现。

以上是有限创新排查，不能据检索未命中断言任何新问题无人研究。特别是分子不平衡和区间专家路线已有直接先例，应在训练之前说明与这些工作的实质区别，而非仅换编码器或数据集。

## 当前决定及下一步的论文候选

本轮病例核查完成。停止继续解释这8个分子；暂不启动新的加权Loss、立体特征或专家模块，也不恢复旧消融队列。现在没有可宣称原创、可投稿的方法结果，不把这份诊断报告当作发刊目标完成。

若继续投入，优先用现有基线做一个限定的可证伪问题：**在相同真实活性区间及训练邻居覆盖下，cliff分子是否仍有额外预测误差，已有候选的cliff改善是否能与普通尾部回归改善区分？**这是潜在评估研究问题，尚未通过完整创新核对，也不能称作二区论文选题已成立。

最小下一步限定为已有神经基线三个任务234/3979/4792，固定训练分位区间、可用样本数与覆盖条件后，复用验证预测做匹配误差比较。不训练、不使用旧test成绩；对不够支持的分组报告缺失，不合并出有利结论。若不能提供跨任务、不同模型一致的信息，结束该候选问题；若有稳定区别，再决定需要补哪一项实验和论文贡献。相同划分的seed不是独立数据划分，确认阶段仍需独立评价。当前仅记录这个候选，不在本病例阶段追加分析。

## 验证和交付

8个选中案例身份、各seed有符号误差与保存的逐分子预测独立核对；全部6组×4模型RMSE和bias共48项由标准库独立重算，容差1e−12。3个seed单极端删除后bias独立计算。源数据SHA256匹配前阶段；脚本语法通过。

输入：GraphCliff-Pair/artifacts/baseline_reuse_234_20261005及原CHEMBL234_Ki官方train行；本地详细输出：GraphCliff-Pair/artifacts/hard_case_audit_234_20261005；公开输出：本目录的聚合summary、选中行号、哈希和报告。未公开逐分子活性、预测、SMILES或权重。

复算：使用GraphCliff解释器，在项目根执行`python -B tools/audit_hard_cases.py --output artifacts/hard_case_audit_234_recheck`。需要一个新的输出目录，避免覆盖既有结果；脚本不会拟合模型。

新增tools/audit_hard_cases.py及本目录交付，更新README/task_plan/notes/milestones。原GraphCliff、my_work基线、当前模型/Loss/训练配置、已有预测均不改。
