# M20：粗糙度方向的方法差异与复用审查

2026-10-05。用户确认先审查算法差异，不建新仓库。**现有“粗糙度迁移GraphCliff、风险分数组合、普通可学习拒答头”尚未形成独立方法贡献，保留0个算法候选，本轮不进入三任务训练。** 复核问题仍有研究价值；这不是对整个风险学习领域或投稿可行性的最终否决。方法论文和中科院二区目标未完成，旧队列保持暂停。

## 1. 固定来源与证据范围

|对象|固定版本|许可/身份|
|---|---|---|
|[qsar-landscape-roughness](https://github.com/krishnatheaverage/qsar-landscape-roughness)|91522b8b81a52d5c95ae216d01a00cbbf0c66209|原LICENSE核对MIT；缓存作者方法/实验/仓库稿件，未运行|
|[Novartis/UNIQUE](https://github.com/Novartis/UNIQUE)|c6d65b9c63102bc18e68c25c98d76e24650c3e4a|原LICENSE核对BSD-3-Clause；误差模型/管线入口，未安装|
|[SelectiveNet ICML2019](https://proceedings.mlr.press/v97/geifman19a.html)|官方出版摘要|用于核查预测与拒答联合学习已有；未全文复现|

14份固定源码/文档SHA256与Git blob逐项通过，9份Python AST解析通过，2份完整仓库树核验。未执行作者模块，未下载数据/权重，静态语法通过不等于运行复现。原PDF是立项稿；其引用和预期观察不当作已实现结果。来源URL/哈希、原PDF哈希和异常见[provenance.json](provenance.json)，检查见[verification.json](verification.json)。缓存留忽略目录，不发布全文、私人PDF或作者整套源文件。

## 2. 粗糙度的“activity-free”究竟指什么

[build_features.py固定代码](https://github.com/krishnatheaverage/qsar-landscape-roughness/blob/91522b8b81a52d5c95ae216d01a00cbbf0c66209/src/build_features.py)使用ECFP radius2/2048bits，K=10及Hölder K=25。对查询x选训练邻居；“不需要查询真实活性”与“不需要任何活性标签”不同。

|代码特征|计算来源|部署身份|
|---|---|---|
|nbr_disp/highsim_disp|训练邻居标签分散度|不需查询y；仍需合法训练邻居活性|
|sali_mean/sali_max|训练邻居两两标签差除以结构距离|不需查询y；不是纯无标签结构特征|
|holder|训练邻居的log距离/log标签差拟合|不需查询y；支持量不足会NaN|
|nn_sim/local_dens/mol_size|查询与训练结构/查询大小|结构特征|
|dirichlet/lipschitz|显式使用查询真实y与邻居y差|仅回顾性/上限变量，不可放进部署风险输入|
|rf_err/cliff_mol/y|评价误差/子群/真值|不能用来生成查询的部署分数或选择阈值|

作者代码在同一缓存里保存两类特征与评价标签，因此薄适配必须采用明确特征白名单，不能全列输入误差模型。包含标签列不自动说明作者泄漏；需要检查下游到底选了哪些列。本轮`conformal.py`使用sali_mean/rf_var/nn_sim，没有观察到其把查询dirichlet作为该部署风险输入。

## 3. 已有运算与实际迁移差异

|拟议改变|作者/近邻已有|我们新增的实际内容|当前判断|
|---|---|---|---|
|把粗糙度用于GraphCliff|粗糙度仓库已有RF/GB/SVR和GIN相关分析|换骨干、身份/划分适配|可作泛化复核，不足以单独支持方法原创|
|合并方差和粗糙度|conformal.py已有标准化两分数之和及各分数比较|重测或改组合权重|普通组合/调权不是明确新机制|
|粗糙度条件预测区间|conformal.py已有Mondrian分组及局部缩放|独立校准、适配新骨干|有协议价值，不能称首次粗糙度校准|
|用小模型学习预测误差|UNIQUE已有误差模型、预测/UQ/特征组合|增加粗糙度为输入|属于特征扩充；具体学习目标尚无差异|
|联合训练预测与拒答头|SelectiveNet已有分类/回归与拒答联合优化|搬入GraphCliff、加粗糙度输入|尚未定义近邻没有的约束或运算|

[conformal.py](https://github.com/krishnatheaverage/qsar-landscape-roughness/blob/91522b8b81a52d5c95ae216d01a00cbbf0c66209/src/conformal.py)里`comb=z(var)+z(rough)`已实现组合，并比较普通/方差/粗糙度/AD/组合分组区间和局部缩放。不是只与单个距离指标比较。

[gnn_tuned.py](https://github.com/krishnatheaverage/qsar-landscape-roughness/blob/91522b8b81a52d5c95ae216d01a00cbbf0c66209/src/gnn_tuned.py)在官方train中另切验证，训练标签标准化仅基于内部训练子集，按validation MSE选权重；同时存在固定预算`gnn.py`。不能称作者尚未尝试GNN。它们不是当前GraphCliff的特征/预算/固定行协议，本轮也未据此排名模型。

[UNIQUE/pipeline.py](https://github.com/Novartis/UNIQUE/blob/c6d65b9c63102bc18e68c25c98d76e24650c3e4a/unique/pipeline.py)有误差模型特征生成、fit与评价；[误差模型基类](https://github.com/Novartis/UNIQUE/blob/c6d65b9c63102bc18e68c25c98d76e24650c3e4a/unique/error_models/base.py)接受predictions/labels/which_set/input_features，计算误差并调用分区函数；[RF误差模型](https://github.com/Novartis/UNIQUE/blob/c6d65b9c63102bc18e68c25c98d76e24650c3e4a/unique/error_models/models/random_forest_regressor.py)已有监督fit/predict。需要我们保证输入预测/风险特征是合法折外，不因库有TRAIN/TEST列就自动保证无泄漏。本轮未深核其分区helper/全部评价选择入口，兼容与完整协议仍未知。

## 4. 协议缺口不等于新算法

作者`conformal.py`把已有官方test预测样本再次随机分为cal/ev，50次重复用于回顾性校准分析。分出的校准样本与当次评价样本不同，**不能直接指控所有结果为标签泄漏**；但它与我们“先锁独立校准角色、最终test一次评价”的契约不同，不应原样运行到当前已封存test。

此外标准化/缺失填补及部分缩放统计使用整个待分cal/ev样本的分数；这些不使用查询标签，却涉及评价分布统计，迁移时要明确转导还是归纳场景。未知其对结果影响，不能未经对照宣称作者效应由此造成。支持少、NaN、重复canonical、序列依赖等也是需要检查的评价条件。

用户PDF希望做强通用对照、联合系列留出和时间外验证。这些提供有价值的独立复核问题；仅修正分区、增加公平对照和改指标，不自动构成算法机制。三任务validation先导可以回答限定复核问题，但我们当前优先方法创新，故不在本轮自动启动它。缩减范围不改变贡献身份。

## 5. 可证伪问题、最低对照及解释

当前唯一明确问题是：固定骨干、合法训练邻域与独立风险学习条件下，粗糙度是否对通用误差分数提供额外信息？它是复核假设，未通过G3算法差异条件。

若日后授权把它作为复核项目，最低控制应有不改活性模型的通用误差分数、粗糙度、简单分数和、UNIQUE同容量误差模型及增加粗糙度的同模型；风险学习只用折外残差，粗糙度也必须由该折训练邻域计算。保留比例相同，比较风险曲线/误差及cliff保留率，不能只比较cliff分类AUC，也不能挑好任务或只看拒掉全部难例后的误差。

最终解释须分开活性预测归因与拒答依据：结构、真值/预测/误差，合法邻居支持信息，固定成功/错误拒答/漏掉难例；原子/子结构重要对随机mask及跨seed稳定性仍必需。拒答分数变化不代表化学因果，也不能用粗糙度关联代替模型归因忠实度。没有保留模型，本轮不新建完整解释系统。

## 6. 五项与停止报告

- 贡献差异：上述具体改动已有近邻，尚无独立目标/约束；不通过。
- 公平性：已有test再校准、特征维度、OOF缺失和预算差异，明确需要薄适配，未做统一成绩。
- 重复性：本轮0训练/0test，作者声称与历史旧结果都不能冒充新机制复现。
- 解释：未做真实新模型解释/忠实度，只保留必需规范与历史有限案例。
- 成本：有MIT/BSD合法复用入口，未测完整CPU/GPU开销；不采信PDF6–8周/120GPU小时为本机预算。

**观察事实：** 固定2仓库/14文件/9 Python静态解析，检查特征来源、风险组合、GIN和误差模型；0训练、0test、0新仓库、0保留算法候选。

**已支持解释：** 普通粗糙度迁移、组合、校准和误差学习已有直接机制；源码澄清了“无查询标签”与“无任何活性标签”的区别。当前主张缺方法差异，而非性能已失败。

**待验证假设：** 结构支持稀疏、噪声、系列转移或弱标签可能限制风险学习；这里都没有独立对照，不据此编造新模块或认定旧方法失败原因。

**尚不能确定：** 三任务迁移效果、严格系列/时间增量、真实收益成本、解释可信度；直接预印本当前正式发表状态本轮未查实，仓库稿件不替代正式版本核验。

**停止理由：** 当前缩小方案仍属于复核，达不到所选算法差异门槛；不启动三任务先导来弥补原创定义空缺，不把方法迁移视为独立贡献。停止的是上述具体算法包装；复核资产归档保留，不否认其学术价值。不能据此保证或否定二区录用。

**改变判断需要什么：** 独立明确的失败行为与最近方法未具备的运算/目标、最低简单替代和预先低成本反证；例如提出具体约束时还需与SelectiveNet等核对，不能只冠以“悬崖感知”名称。当前不预设新候选存在，不追加第四模块。若希望转为独立复核论文，需要用户明确改变论文类型；不自行转向。

本次只新增审查报告/来源/核验，更新README/框架/计划/notes/里程碑；无模型/Loss/配置变化。原PDF、GraphCliff/my_work只读，输入hash和公开范围检查后提交push。既有[解释规范](../framework_review_20261005/explainability.md)、[M18](../neighbor_audit_20261005/report.md)和[M19](../mcs_mask_audit_20261005/report.md)停止身份保留。
