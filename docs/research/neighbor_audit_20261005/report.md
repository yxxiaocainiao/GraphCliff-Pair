# M18：关键近邻深核验与本轮止损

2026-10-05。**完成有界G1、三个问题G2及G3筛查：保留0个候选，暂停本轮选题，不进入G4训练。** 三个改动均有直接公开重叠，尚不足以提出独立方法贡献。该结论只针对下述具体机制，不是整个活性悬崖领域不可创新，也不是二区期刊录用判断。总体框架底座保留；创新核心仍空缺，发表目标未完成。

## 范围与证据身份

以用户指定29条整理（ACA版本合并为28研究条目）为入口，只深核验ACES-GNN、PrismNet、MAPCliff-WMGR的方法/实验及关键入口；为检查两个公式差异，定向增加UCN与SAGGLR。不是29篇重新阅读全文，也不代表aidd全库或穷尽相关工作。原文、提取全文、作者源码缓存均留忽略目录，不在公开仓库分发。

三份正式英文PDF公式已视觉核对：ACES-GNN PDF第5页式5–8；PrismNet第15–16页式1–10；MAPCliff-WMGR第5页式8–9。其余判断来自注明的原文段落或固定代码。UCN/SAGGLR的下述运算以作者源码为证据，不冒充整篇深读。输入哈希、完整树、每份源码URL/哈希见[来源记录](provenance.json)，核验见[verification.json](verification.json)。

|作者仓库|固定commit|许可核查|复用结论|
|---|---|---|---|
|[Liu-group/XACs](https://github.com/Liu-group/XACs)|889b6596913161490dbade6b2203183a9f41c3bf|MIT|可依法薄适配；当前只读审查|
|[GZU-SAMLab/PrismNet](https://github.com/GZU-SAMLab/PrismNet)|4698a5bc6c3ae0e31284e84676a375f8ed10941b|MIT|可依法复用；特征、划分、检查点不等同当前底座|
|[tfwuSUDA/MAPCliff-WMGR](https://github.com/tfwuSUDA/MAPCliff-WMGR)|56f6653b31d63dcdc65184fb2f442d28d4a32bd5|完整树未找到许可证文件|不接入或再分发源码；不是认定所有使用均违法|
|[microsoft/molucn](https://github.com/microsoft/molucn)|5c9c689c1c9ca1dc12b2c933648bf6e30edf70c3|MIT|可依法复用；仓库已归档，运行兼容性未测|
|[FrancisShizy/SAGGLR](https://github.com/FrancisShizy/SAGGLR)|7bfbe330d0108a34547b2ce7607a60e4a3e8a4d1|完整树未找到许可证文件|仅机制核对，不接入源码|

33份文件SHA256及API树Git blob一致，25份Python仅AST解析通过；未导入作者项目、未下载预训练权重、未测GPU适配。静态通过不能称为复现成功。

## 1. ACES-GNN：归因方向监督与当前协议的边界

令原子归因 `a_v=Σ_k (∂ŷ/∂A_vk) A_vk`，非共有区域归因之和为u，共有区域归因为向量c。原文式6–8：

```
L_u = max(0, -(u_i-u_j)(y_i-y_j))
L_c = ||c_i||² + ||c_j||²
L = MSE + λ Σ_pairs (L_u + L_c)
```

这是无正margin的方向约束，没有要求归因差拟合真实差值幅度。代码对应[XACs/train.py:180–196](https://github.com/Liu-group/XACs/blob/889b6596913161490dbade6b2203183a9f41c3bf/XACs/train.py#L180)、[utils.py:24–33](https://github.com/Liu-group/XACs/blob/889b6596913161490dbade6b2203183a9f41c3bf/XACs/utils/utils.py#L24)；`explain_utils.py`使用`create_graph=True`，归因监督需要保留可微梯度图，不能用普通展示热图代替训练运算。实际二阶求导成本及GraphCliff兼容性未测。

论文§2.6采用ECFP谱聚类/悬崖分层80/10/10、十次重复划分及Hyperopt/GridSearch；最多1000轮、早停150等预算不同于本项目。官方`main.py:22–26`在划分前构造全体cliff字典。`dataset.py:181–208`随后把训练伙伴过滤到train范围，因此**不能说test伙伴直接进入Loss**；但`193–200`先从全局字典确定padding的max_length，可能让其他分区标签影响训练元数据。严格协议需在train内重建字典、区域与padding上限。当前是信息依赖核查，不是对原论文结果失效的裁决。

此外完整树未找到README所述config文件，默认参数/调参入口仍存在；不是“无法运行”，但尚不能一键还原论文每次配置。采用原文划分或可选train+val ensemble会改变评价契约，不能直接与我们的固定行划分排名。

## 2. PrismNet：已有频率/语义交互与单任务权重边界

原文第15页以scaffold/功能团/pharmacophore语义视图预训练，融合Transformer重建；第16页式4–6定义高/低频`(I-P)X`、`(I+P)X`及交互，式10按各任务Loss历史偏差更新后归一化任务权重。

固定源码[models/prismnet.py:425–488](https://github.com/GZU-SAMLab/PrismNet/blob/4698a5bc6c3ae0e31284e84676a375f8ed10941b/models/prismnet.py#L425)采用`D-A`的高阶项、绝对特征差及softmax等运算，并非直接逐行实现上述归一化线性滤波式。需要作者说明或独立运算对齐，不能在未核对时把搬运模块叫作精确论文复现。

`scripts/utils.py:230–245`包含MoleculeACE的30个CHEMBL任务，但配置是`n_output=1`。`finetune.py:294–309`正权重更新后除以权重总和；**单任务时w/w恒为1**，不能把它说成会动态改变该任务内每个分子的权重。此身份已做代数核验；表示学习、标签平滑对比Loss仍可发挥作用，没有证据说整个PrismNet无效。

评价入口的具体差异：`finetune.py:477–479`固定scaffold划分；`313–315`验证loader同时`shuffle=True/drop_last=True`，不足整批的尾部随排列变化；`422`test亦`drop_last=True`遗漏固定尾部。若用来做本项目公平对照，需固定行划分并保留全部评价样本，而非搬用作者成绩。模型还使用115维原子/13维键、附加全局描述符节点及语义预训练权重，额外信息与我们的38维原子输入不同。密集原子对张量随N²增长，8GB可行性及额外预训练成本未知。源码按validation选模；该入口未发现逐epoch按test选模。

## 3. MAPCliff-WMGR：加权图/多特征与实现对应限制

原文第5页式8使用按维度可学习频率的sin/cos映射；模型组合加权分子图及描述符/PubChem/ECFP。固定[ednn_utils.py:306–329](https://github.com/tfwuSUDA/MAPCliff-WMGR/blob/56f6653b31d63dcdc65184fb2f442d28d4a32bd5/code/units/ednn_utils.py#L306)的SineLayer则为`sin(omega_0 * Linear(input))`。两者不应未经运算对齐就称完全等价。

原文§2.5以MoleculeACE原train/test为基础；§2.7从train取10%验证并调参，部分比较来自其他作者已报结果。`end2end_train.py:268–274,318–385,508–515`亦体现train内随机验证、验证调参、选模后test；该入口未观察到test参与选择。输入特征/预算和基线本地重跑身份仍需匹配，不能把文中排名当统一协议下的本地证据。DGL/DeepChem等依赖与当前PyG不同；无许可和公式对应未明时不值得搬整个项目。此处不构成性能否决。

## 4. 三个具体问题：最低反证与淘汰理由

这些是本轮筛查草案，**不是新方法、预注册或已运行实验**。所有新正式/冒烟训练计划数和实际数均为0，不冻结任务/种子/数值门槛，因为G3不通过。解释要求用于判断可证伪性，不为阴性候选新建系统。

### Q1：非共有区域由方向监督改为差值幅度监督

草案：`L=MSE+λ[(s_Ui-s_Uj)-(y_i-y_j)]²`；假设：幅度监督改善合法验证分子对的有符号差值误差，且不牺牲普通预测。最近三项是ACES-GNN、UCN、SAGGLR。

**淘汰：UCN已实现该核心。** [2023正式论文](https://doi.org/10.1186/s13321-023-00733-8)及[作者loss.py:14–26](https://github.com/microsoft/molucn/blob/5c9c689c1c9ca1dc12b2c933648bf6e30edf70c3/molucn/gnn/loss.py#L14)直接对非共有区域标量读出的差做MSE；`model.py:212+`有相应mask读出。因此只把ACES方向改幅度、或把UCN接到GraphCliff，没有具体独立贡献。归因梯度与区域读出不完全相同，但尚无明确新约束来使这种差别成为可检验贡献。

最低对照应为同骨干MSE、ACES、UCN与去区域差值Loss，训练对/MCS规则/预算完全匹配；主要检查验证pair误差及Overall/Cliff代价。解释需检验重要非共有区域相对等大小随机区域的遮罩响应和跨seed稳定性。区域划分、伙伴forward及归因求导是实际额外成本；未测耗时/显存，不填数字。

### Q2：允许共有上下文参与活性差，而非强制共有归因为零

草案：`L=MSE+λ[(w_U Δs_U+w_C Δs_C)-Δy]²+R`；假设：共有上下文参与能在相同区域划分下改善差值预测。最近三项是ACES-GNN、UCN、SAGGLR。

**淘汰：SAGGLR已公开同类结构。** [正式出版页](https://spj.science.org/doi/10.34133/csbj.0012)及[作者loss.py:149–179](https://github.com/FrancisShizy/SAGGLR/blob/7bfbe330d0108a34547b2ce7607a60e4a3e8a4d1/SAGGLR/gnn_framework/loss.py#L149)联合共有/非共有差值并加组稀疏约束。我们的草案没有定位到其未具备的运算。区域预测与梯度归因有差别，不能因抽象不同就认定足够原创。

最低对照为ACES共有置零、UCN非共有、SAGGLR双区域及去每个区域；骨干、特征、分子对和预算匹配。必须注意：只观察总Δy不足以唯一确定共有/非共有贡献，不应声称分解得到化学因果。解释可检验固定两类区域遮罩响应及seed稳定性，不能仅用训练时同一种区域弱标签证明归因正确。成本是区域映射/双读出及正则；许可、忠实度及GPU成本未确认。

### Q3：高低频表示交互配合动态Loss

草案：从`X_H=(I-P)X, X_L=(I+P)X`融合预测，叠加归一化任务Loss权重；假设：频率互补降低固定验证误差。最近三项是GraphCliff、PrismNet、MAPCliff-WMGR。

**淘汰：频率交互已由PrismNet覆盖，当前单任务任务权重还退化为1。** 更换模块名称或把频率模块移入GraphCliff没有新增数学约束；若改成逐样本加权，又须与已审查ACA/不平衡回归机制区分，不能临时追加第四个候选。此判断不是说所有频率方法相同，而是本草案未提供可定位的不同运算。

最低对照为原GraphCliff、匹配输入/预算的频率替换、去交互、固定MSE及最近实现；之前LongPoly替换的No-Go只能约束那次具体替换，不能直接证明PrismNet必败。解释需检验频率运算与遮罩后的模型响应，热图不是“真实化学频率”证据。密集交互和预训练/附加特征成本未测，不能称低成本。

## 5. 五项审查与停止报告

|审查项|已有证据|判断|
|---|---|---|
|贡献差异|三个草案分别有UCN/SAGGLR/PrismNet直接重叠|不通过，是本轮停止的首要依据|
|对照公平性|划分、额外输入、评价drop_last、调参预算等已有具体差异|需薄适配；没有当前统一协议成绩|
|跨任务重复性|本轮0训练；旧结果属于其他固定机制|未验证，不能引用作者/旧成绩冒充新候选效果|
|解释可信度|原解释规范仍有效；新候选忠实度/稳定性未跑|未验证；不新建阴性模型热图系统|
|计算成本|higher-order梯度、N²交互、多依赖有明确实现位置|实际显存/时间未知；没有为不清楚的创新支付测量预算|

**观察事实：** 核查3个主近邻、2个定向补充近邻、3个算法草案，保留0；无新任务/seed/对照指标，0训练、0test、0新解释。已有实验反证见[历史停止报告](../framework_review_20261005/decision.md)，不重复计为本轮结果。

**已支持的解释：** 具体运算已有公开重叠，且作者成绩信息/预算不等同本地公平对照。本轮结论来自贡献审查，不来自预测下降。

**待验证假设：** 高阶梯度成本、区域映射误差、弱标签噪声、共有上下文作用、预训练迁移等可能影响性能；当前都没有对应新对照，不能当作失败原因。

**尚不能确定：** 未知这些方法移入GraphCliff后的涨点、30任务泛化、真实化学分解及忠实性。不从旧性能下降推导过拟合、噪声或模块普遍无效；也不从本轮淘汰推导无法发表。

**停止理由：** 没有可定位的独立方法差异，先训练不能弥补这个关键缺口。按G3“六项全通过”规则，差异项失败就足以不进入G4；其余未知不能被当成通过。暂停本轮选题，保留底座、台账、失败案例与解释局限，不追加模块、第四问题或重新搜索参数。

**什么证据可能改变决定：** 新的具体目标或运算必须说明与UCN/SAGGLR/PrismNet等近邻的精确差别、为何旧试验未覆盖、以及低成本反证和解释检验；先形成可审查定义，才重新申请下一轮筛查。当前不预设该候选存在，不将文献阅读或总体框架占位称为论文贡献。若仍无此证据，当前方法不足以继续支撑二区方法论文目标。

## 6. 解释交付边界、异常与复核

没有最终保留模型，本轮不建设完整解释系统。已有有限失败案例、原子/子结构展示与局限见[解释审查](../framework_review_20261005/explainability.md)。未来任何保留模型仍须生成前固定成功/失败/一般案例，展示结构、真值/预测/误差（pair另有参考和差值），汇报所有选定案例、重要区域对等大小随机区域掩蔽、跨seed稳定性。掩蔽只解释模型行为；注意力/指纹权重不能单独证明机制。PyG节点mask合成兼容检查不能替代ACES可微归因训练兼容性；分类fidelity不直接用于回归。

异常：出版社403/验证码采用本地正式英文PDF补证；Poppler字体警告不影响已检查公式页；最初文本哈希按文件bytes比较提取字符串而失败，确认Windows CRLF后保留原预期哈希，以统一换行文本核验，同时记录真实缓存bytes哈希。控制台打印论文行遇GBK编码失败，未据此推断DOI。没有训练中断或覆盖旧结果。定向加查UCN/SAGGLR属于重叠补证，不扩展候选数。历史记录保留“当时尚未完成”身份，不改成提前计划。

复核入口：`python tools/check_neighbor_sources.py artifacts/neighbor_audit_20261005 --output artifacts/new_source_verification.json`（全新输出，已有输出拒绝覆盖）。33份固定源码Git blob及SHA256、25份静态AST、三份提取文本预期哈希、MIT文件/无许可证完整树、单任务权重恒等式核验通过。私有缓存不随仓库发布，公开来源URL与哈希可用于重建源码部分；全文复核需合法取得相应PDF，不能保证第三方具有本地提取JSON。

本轮只新增报告、来源/核验JSON与来源核验工具，更新当前README/计划/框架/notes/里程碑；不修改模型、Loss、训练配置、原GraphCliff/my_work或原Excel/PDF。提交前另查链接、diff范围、缓存忽略和原文件哈希，push后核对远端SHA。
