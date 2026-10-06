# M26：暂停模块实验的事后复核

日期：2026-10-06。目的：按用户明确的“可复用底座上的算法增量贡献，争取中科院二区或核心”重新判断历史证据。本文是事后整理与新的研究价值判断；不改旧预注册、原始结果和No-Go，不把放宽项目筛查称为通过旧方案。

**结论：以前的停止不能统一归因于涨点不足。编码后Cross-Attention、直接FPPool和ACA存在实质退化；残差FPPool及部分Pair-FPPool有条件性收益；最新粗糙度增强属于三个任务一致小幅正向但未达旧门槛。唯一优先补证对象建议为粗糙度风险增强，当前仍暂停训练。没有证据支持恢复“三板斧”整套队列。**

## 1. 范围、核验及复现

输入：原只读 `D:/GraphCliff-main/eval_out`，当前项目 `D:/GraphCliff-Pair/artifacts`，只读 `D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines/results`；以M14台账及已有配置/验证预测为入口，不使用官方test选择模型。新版风险结果另取M25的 `artifacts/reliability_phase_a_20261005_v3/results.json`。

复用 `tools/audit_framework.py` 新做保存产物算术核验：223条产物记录、2处仅有未完成history、0解析错误、874份受保护输入未变、18项直接FPPool指标核对、144项解释摘要核对。223包含冒烟与恢复副本，不是223次独立正式训练。进一步排除smoke与seed43 recovered副本，选择117份历史运行，生成39个按任务/对应seed配对的比较；另外逐行预测重算component实验48行分子对MAE及数量。没有新增训练、checkpoint推理、解释优化或test评估。

计算口径：先对对应seed取指标均值，再算 `100*(baseline_mean-candidate_mean)/baseline_mean`。正数代表误差降低；只比较双方都完成的seed。各阶段划分、训练版本与信息条件不同，不能跨阶段直接排行。原GraphCliff标签按float32重算以继承原指标口径。三seed仅证明这几个运行的方向，不能替代跨任务或独立重复；同一分子参与多条pair边，不能把边数当独立样本数做显著性包装。

公开：[比较数值](comparisons.json)、[输入哈希/配置/运行来源](provenance.json)、[复核脚本](../../../tools/review_module_pauses.py)。私有完整台账仍在忽略目录 `artifacts/module_pause_review_20261005_inventory`，不公开逐分子预测/权重/数据。脚本先运行旧审计器再聚合；需本机原产物，公开摘要不足以恢复训练数据。

```powershell
python tools/audit_framework.py artifacts/module_pause_review_20261005_inventory
python tools/review_module_pauses.py artifacts/module_pause_review_20261005_inventory/inventory.json docs/research/module_pause_review_20261006
```

当前分析基准提交为 `8df3c39f5e7d6f87081784cda5c8089e6a66ab28`。历史运行source_commit缺失的地方保持null，配置及源文件哈希作为可用证据，不虚构当时版本。原my_work未确认Git HEAD，不据此声称严格重现作者结果。最新阶段A的逐行核验继承M25已完成审计，本次只重新读取聚合及绑定hash。

## 2. 逐路线判断与停止原因

### 2.1 原GraphCliff直接FPPool：保持归档

观察事实：3979_EC50，seed42/43/44；Cliff RMSE均值0.675169→0.690374，恶化2.25%，只有1/3 seed改善；Overall恶化1.58%，没有seed同时改善两项。seed42的no_sag另有退化。历史训练总耗时约增至8.5倍，但早停轮数/运行条件不同，不解释为单步效率下降8.5倍。

已支持的解释：这个替换方案没有提供稳定收益，因此停止不依赖5%门槛。待验证假设：读出与指纹语义不适配、优化差异等。尚不能确定：性能下降不足以证明过拟合、指纹无效或FPPool普遍无效。继续理由必须是具体读出兼容性机制及同信息/预算对照；当前没有低成本明确路径，不恢复。

### 2.2 残差FPPool：保留条件性研究资产，不优先重启

观察事实：3979单seed Cliff改善6.81%，但Overall恶化3.20%；bridge是相同权重的另一参照记录，不是独立重复。三任务三seed正式主比较：2047 Cliff恶化2.99%、Overall改善3.69%；235 Cliff改善1.00%（1/3 seed）；287 Cliff改善2.24%（2/3 seed），但同时改善两项为0/3。

GMT对照：残差FPPool Cliff优于GMT 3.12%/2.59%（2047/287，各2/3 seed），提供超出某个等参数读出对照的有限证据。GMT和FPPool参数接近，但GMT存在68维瓶颈，参数接近不保证表达能力或计算成本公平。真实membership与置乱相比：287改善3.96%、3/3同向；2047反而恶化4.61%、仅1/3改善。两任务是此前结果引导选择，属探索性补充，不能视为未看结果的独立确认。

已支持的解释：部分任务对真实指纹分组敏感，有条件性正向信号；“所有FPPool都失败”不成立。待验证假设：收益是否由特定结构/指纹覆盖决定。尚不能确定：不能仅据287结果推导化学机制或推广至30任务。解释旧结果亦不增强该主张：top遮罩变化高于随机比例baseline/residual在287为0.654/0.613、2047为0.627/0.542，跨seed Jaccard也未更好；归因针对隐藏表示，不能宣称输入原子机制正确。停止理由是跨任务目标冲突、机制证据混合和额外验证成本，而非涨幅小。保留为备选资产，不新增结构门控模块来救这条路线。

### 2.3 编码后跨分子Cross-Attention：保持归档

观察事实：234/244各三seed，direct/global/pair_mlp/cross共24/24正式运行。Cross对direct的Cliff恶化12.24%/25.54%，各0/3获胜；对pair_mlp恶化1.79%/9.86%，亦各0/3获胜。对较弱global在234改善3.41%，244恶化8.55%。不能只选择234的global比较讲正向故事。

已支持的解释：当前参考机制下注意力并未优于可用的非注意力pair对照。direct与pair信息和图计算量不同，pair_mlp是更直接的交互机制对照，近似参数匹配仍不等于完全公平。待验证假设：参考质量、交互粒度、训练目标。尚不能确定：不从下降直接推导参考噪声/过拟合，亦不推出所有Cross-Attention走不通。停止理由是已有重复反证，不值得只为“仍有创新故事”重跑。

### 2.4 用逐层Cross-Attention替换LongPoly：保持归档

观察事实：两任务各一个seed的full/short/cross，6/6完成；对同期full，234 Overall改善3.79%、Cliff改善0.08%；244 Overall恶化1.16%、Cliff恶化4.93%。对short，Cliff改善2.58%/1.37%，但244 Overall仍恶化。结果不是纯粹“两个任务都涨得不够”。

已支持的解释：替换并未稳定胜过同期双通道；234局部收益值得记载。待验证假设：额外容量/规范化/参考差异。尚不能确定：只有单seed，且cross增加约79万参数，同时改LayerNorm/dropout，不能把差异单独归因于attention。旧direct数值协议存在差异，仅作背景，主判断用同期full/short。继续需要更干净的消融及重复，目前成本高于信号，归档。

### 2.5 Pair-FPPool、动态Loss及三模块组合：矩阵未完，但已有反证

观察事实：原计划78次正式运行中独立完成56次（交互24＋seed42新臂18＋seed43新臂14）；seed43第15次CUDA非法指令中断，恢复目录14份为副本，seed44未开始，不能把恢复复制计为新训练。缺少完整三seed因子矩阵，不能宣布所有组合都充分检验。

对已完成42/43配对：global_fp在234 Cliff改善8.65%，两seed都胜；244恶化12.66%，两seed都败。cross_fp均值在234/244恶化1.79%/6.55%。global_dynamic均值恶化1.06%/14.31%（234仍有一个seed的Cliff改善）；cross_dynamic恶化6.45%/7.61%。整套cross_fp_dynamic对direct：234两seed均值恶化22.60%，244只有seed42完整，恶化33.53%。不能混用2seed均值与1seed值伪装完整矩阵。

动态Loss的确切含义：`weights=1+alpha(epoch)*min(abs(delta)/scale,cap)`，再除以batch权重均值；alpha按epoch线性warm-up。它是标签差值幅度加权的时间调度，不是学习的不确定度、误差自适应或元学习权重。静态对照只在FP上下文已做，不能外推为所有上下文都证明dynamic更好。

已支持的解释：部分global-FP组合有任务条件性收益，三模块叠加没有已见一致优势。待验证假设：差值权重是否放大噪声、参考采样是否失配。尚不能确定：没有梯度/噪声对照，不能据退化断言这些原因。FPPool若干历史运行耗时增至约5–11倍，但早停轮数不同；这是成本提示，不是精确算子效率结论。停止理由是跨任务反证与补全矩阵成本；不为补齐表格恢复剩余22次训练。

### 2.6 Component-aware辅助差值Loss：权重贡献未获支持

观察事实：原单分子GraphCliff加辅助pair Loss，234/244×三seed×mse/balanced/uniform/permuted共24/24；它不同于当前参考驱动的Pair预测。balanced对MSE Cliff改善1.14%/恶化1.57%；234虽2/3 seed的Cliff改善，两项同时改善仅1/3。balanced在两个任务Cliff都不及uniform（恶化2.03%/1.12%）；244也不及permuted（恶化3.22%）。合法Cliff-pair MAE仍有约1.02%/0.26%小幅均值改善（1/3、2/3 seed同向），不写成所有指标全败。

已支持的解释：已有简单uniform/permuted对照削弱了“特定权重有效”的主张，不能仅用对MSE的234收益讲创新。待验证假设：组件尺度/边频率/样本难度是否混杂。尚不能确定：下降不能证明组件结构没有信息；27/18条cliff边关联同分子，不能做独立样本显著性夸大。停止理由是机制特异性与重复性不足，而非仅没达到2%项目门槛。不再搜索lambda救结果。

### 2.7 ACA迁移：保持归档

观察事实：18计划运行中仅234三seed×两臂6次正式完成，244下一MSE到16epoch中断；冒烟不补入正式结果。ACA Cliff均值恶化3.01%、Overall恶化4.55%，只有1/3 seed改善。使用官方类默认alpha=0.1、平方p=2，与本地MSE主任务相结合；不是ACANet论文全部MAE协议的精确复现。有效triplet覆盖已有核对，不据此说“没有三元组”。

已支持的解释：当前系数/目标/协议下转移不值得扩大。待验证假设：损失尺度或梯度失衡。尚不能确定：未测梯度比，其他任务没有正式结果，也不能判定原方法无效。停止理由是同容量对照下的重复退化及成本，不仅是3%门槛；保持归档，GPL来源及归属沿用既有审查。

### 2.8 Chemprop＋粗糙度风险增强：唯一优先补证建议

观察事实：234/244/4792，seed42，12 Chemprop＋12作者特征RF＋6风险RF＝30/30 fit，约7.09分钟，0官方test。generic与augmented使用相同点预测；改善的是固定六个coverage点的选择性RMSE均值，不能称点预测RMSE涨点。增强相对generic改善2.08%/2.49%/3.76%，任务相对改善平均2.78%；三个任务均正向，且macro优于distance/SALI。

原固定规则要求至少两任务≥5%、任务相对平均≥3%、无任务退化>5%、macro胜distance/SALI；前两项未达，后两项满足。M25的No-Go保持事实。新判断：这组证据最符合“可重复的小幅增量”补证价值，统一5%不是期刊录用标准，不能因幅度小直接判定不可发表。也不能通过改门槛事后宣布成功。

已支持的解释：在三任务这一seed/划分，训练邻域粗糙度给generic风险特征带来额外排序信号。待验证假设：这种增量是否跨seed/划分保持，是否是现有误差模型容易吸收的普通特征效应。尚不能确定：新算法差异、强ensemble/MVE对照、时间外推、完整解释忠实度都未建立；OOF参考规模变化也未排除。现有六个固定结构案例只是有限误差分析，不是通过忠实度的完整解释。

建议下一步仅固定一个小补证方案：复用现有三任务/相同超参数/相同seed绑定划分，增加42以外两个独立seed，generic/augmented继续共享点预测；最多新增60次fit、总计90次，不选任务、不调参、不叠加模块。开始前写清目标贡献、最近方法差异、主指标与新筛查规则，把M25作为探索证据，不当确认性证据。新规则应按方向稳定性、相对简单/强对照及成本综合判断，不能承诺单靠1–3%可发二区。**本轮仅建议，不写成已冻结方案，不启动B或恢复旧队列。** 如果找不到可辩护的方法差异，先停止方法论文路线，保留工具与阴性结果，不用故事掩盖缺口。

## 3. 不属于模型效果失败的已停止想法

BMC/LDS/FDS/IA-MoE属于论文/接口/许可筛查，没有分子正式实验，不能称其性能失败。M18的简单非共有幅度监督、共有上下文差值、高低频交互与既有ACES-GNN/PrismNet/MAPCliff等存在公开重叠，停止的是具体简单方案，不是所有任务针对性改进。M19单MCS监督：72次合成重编号检查、48训练结构近邻对中43完成/5超时、没有满足充分诊断条件的不同排名证据，停止是结构假设未获支持，不是训练RMSE失败。M20粗糙度简单迁移缺贡献区别，后续M21–M25实跑底座不撤销该差异缺口。

原3979的MAE/Huber小对照仅单任务单seed，均不及MSE；两个MSE重复具有同一权重，不当独立重复。它们是背景资产，不能推广成所有回归Loss均走不通。

## 4. 五项发表证据检查与最终决定

|路线|贡献差异|对照公平性|跨任务重复性|解释可信度|成本与决定|
|---|---|---|---|---|---|
|直接FPPool|公开读出迁移，任务机制未证|同任务三seed，有读出控制|负向，1任务|不能补性能反证|高成本，归档|
|残差FPPool|287 membership有有限信号|有GMT/置乱，但预算/表达能力未全匹配|3任务混合|已有隐藏归因未更好|保留备选资产|
|编码后Cross|需胜非注意力pair才成立|有pair_mlp近参数对照|2任务3seed负向|接口检查不等于解释成功|归档|
|替换LongPoly|容量/规范化同时变化|同期full/short优先|单seed，1正1负|未完成真实归因|归档|
|Pair FP/动态组合|epoch warm-up非新自适应机制|不完整因子矩阵|已有2seed方向冲突|未通过完整忠实度|不补全队列|
|Component Loss|特定权重不胜简单对照|uniform/permuted较有力|2任务3seed混合|组件统计不等于归因|归档|
|ACA|公开Loss迁移，差异未成立|局部协议同容量，非论文全复现|1任务3seed负向|未构建机制证据|归档|
|粗糙度风险增强|现有风险特征组合，差异仍待落实|共享点预测，需强风险对照|3任务单seed同向|仅有限案例|唯一优先补证建议，仍暂停|

小幅稳定收益可以是证据的一部分，不能单独替代机制差异、公平对照、解释与复现。[Journal of Cheminformatics研究论文官方指南](https://jcheminform.github.io/jcheminform-author-guidelines/guidelines/researchArticle.html)要求说明相对已有工作的贡献并提供可复现的数据/软件/算法；该引用不证明任何固定涨幅或本项目录用/分区。现阶段没有一条路线已达到“论文方法及证据完整”，但也不应把条件性信号一概写成无价值。最终可解释性仍是必需交付，规则继承[M14解释验收](../framework_review_20261005/explainability.md)：固定成功/失败/一般案例，重要区域对同规模随机区域遮罩，跨seed稳定性，兼容性/忠实度不足如实披露；图遮罩只解释模型行为，不是化学干预。

## 5. 日志、异常及验收

本次计划：只读复核、配对聚合、更新判断、公开聚合/来源、核验并提交。实际：1次新完整历史审计；聚合脚本第一次因历史subset字段为`cliff_pairs`而非`cliff`校验失败，修正按原schema后48/48 MAE/数量核对通过，未修改来源或跳过断言。此前两个猜测路径`losses.py`和`docs/publication_feasibility.md`不存在，已定位`loss.py`及带日期的正式文档；不是训练失败。普通权限git状态检查不能识别worktree，升权只核验授权项目目录。没有训练异常、重启或参数搜索。

分析准备在2026-10-05开始，2026-10-06续办提交；私有审计目录保留开始日期，本报告采用用户当前日期。事后补记明确标注，不冒充旧实验前登记。新增本报告/聚合/来源和薄脚本，更新README、框架、研究计划、发表可行性、task_plan、notes、milestones；原GraphCliff/my_work、模型/Loss/数据/权重、旧报告/固定方案不改。提交前核验比较表与JSON、来源hash、链接、diff及忽略目录；推送后核对本地/远端SHA。历史参数和实际耗时见provenance，未记录值保留null。

## 6. 完整配对比较表

正数为误差改善；括号为改善seed数/已配对seed数。风险指标单列，不能与Cliff RMSE混用。

|比较|任务|seed|Overall改善%|Cliff/风险改善%|
|---|---|---|---:|---:|
|direct_fppool_vs_baseline|CHEMBL3979_EC50|42,43,44|-1.583 (1/3)|-2.252 (1/3)|
|residual3979_vs_baseline|CHEMBL3979_EC50|42|-3.199 (0/1)|+6.807 (1/1)|
|residual_fppool_vs_baseline|CHEMBL2047_EC50|42,43,44|+3.692 (3/3)|-2.991 (1/3)|
|residual_fppool_vs_baseline|CHEMBL235_EC50|42,43,44|+0.736 (2/3)|+0.995 (1/3)|
|residual_fppool_vs_baseline|CHEMBL287_Ki|42,43,44|+0.352 (1/3)|+2.239 (2/3)|
|residual_fppool_vs_gmt|CHEMBL2047_EC50|42,43,44|-0.061 (1/3)|+3.118 (2/3)|
|residual_fppool_vs_gmt|CHEMBL287_Ki|42,43,44|-0.536 (1/3)|+2.586 (2/3)|
|true_membership_vs_permuted|CHEMBL2047_EC50|42,43,44|+1.758 (2/3)|-4.610 (1/3)|
|true_membership_vs_permuted|CHEMBL287_Ki|42,43,44|+0.124 (1/3)|+3.956 (3/3)|
|cross_vs_direct|CHEMBL234_Ki|42,43,44|-14.236 (0/3)|-12.239 (0/3)|
|cross_vs_direct|CHEMBL244_Ki|42,43,44|-25.338 (0/3)|-25.543 (0/3)|
|cross_vs_pair_mlp|CHEMBL234_Ki|42,43,44|-2.206 (0/3)|-1.787 (0/3)|
|cross_vs_pair_mlp|CHEMBL244_Ki|42,43,44|-8.759 (0/3)|-9.856 (0/3)|
|cross_vs_global|CHEMBL234_Ki|42,43,44|+4.786 (3/3)|+3.413 (2/3)|
|cross_vs_global|CHEMBL244_Ki|42,43,44|-4.872 (0/3)|-8.546 (0/3)|
|branch_cross_vs_full|CHEMBL234_Ki|42|+3.787 (1/1)|+0.082 (1/1)|
|branch_cross_vs_full|CHEMBL244_Ki|42|-1.159 (0/1)|-4.932 (0/1)|
|branch_cross_vs_short|CHEMBL234_Ki|42|+1.148 (1/1)|+2.581 (1/1)|
|branch_cross_vs_short|CHEMBL244_Ki|42|-1.180 (0/1)|+1.369 (1/1)|
|global_fp_vs_global|CHEMBL234_Ki|42,43|+5.355 (2/2)|+8.648 (2/2)|
|global_fp_vs_global|CHEMBL244_Ki|42,43|-6.613 (0/2)|-12.662 (0/2)|
|cross_fp_vs_cross|CHEMBL234_Ki|42,43|-1.818 (0/2)|-1.794 (1/2)|
|cross_fp_vs_cross|CHEMBL244_Ki|42,43|-2.305 (0/2)|-6.551 (0/2)|
|global_dynamic_vs_global|CHEMBL234_Ki|42,43|-2.605 (0/2)|-1.064 (1/2)|
|global_dynamic_vs_global|CHEMBL244_Ki|42,43|-7.968 (0/2)|-14.313 (0/2)|
|cross_dynamic_vs_cross|CHEMBL234_Ki|42,43|-3.547 (1/2)|-6.450 (1/2)|
|cross_dynamic_vs_cross|CHEMBL244_Ki|42,43|-3.271 (0/2)|-7.608 (0/2)|
|cross_fp_dynamic_vs_direct|CHEMBL234_Ki|42,43|-24.050 (0/2)|-22.596 (0/2)|
|cross_fp_dynamic_vs_direct|CHEMBL244_Ki|42|-28.055 (0/1)|-33.533 (0/1)|
|balanced_vs_mse|CHEMBL234_Ki|42,43,44|+1.642 (1/3)|+1.137 (2/3)|
|balanced_vs_mse|CHEMBL244_Ki|42,43,44|-1.486 (1/3)|-1.569 (1/3)|
|balanced_vs_uniform|CHEMBL234_Ki|42,43,44|+1.508 (2/3)|-2.034 (1/3)|
|balanced_vs_uniform|CHEMBL244_Ki|42,43,44|-2.582 (0/3)|-1.124 (1/3)|
|balanced_vs_permuted|CHEMBL234_Ki|42,43,44|+3.162 (3/3)|+2.285 (2/3)|
|balanced_vs_permuted|CHEMBL244_Ki|42,43,44|-3.174 (1/3)|-3.221 (0/3)|
|aca_vs_mse|CHEMBL234_Ki|42,43,44|-4.546 (1/3)|-3.013 (1/3)|
|roughness_vs_generic|CHEMBL234_Ki|42|—|+2.080 (1/1)|
|roughness_vs_generic|CHEMBL244_Ki|42|—|+2.487 (1/1)|
|roughness_vs_generic|CHEMBL4792_Ki|42|—|+3.760 (1/1)|
