# M61：234/seed42风险迁移最小先导协议

**决定：冻结一个四臂工程先导的设计和预算，不执行训练。** 接续[M60](../series_risk_transfer_20261007/report.md)的唯一适配，不新增算法候选。实验回答“在固定组件划分、同一点预测和同一风险RF容量下，风险证据空间重加权是否值得继续”，不是验证跨时间、跨任务或论文级原创性。

## 数据与标签权限

采用[M35](../component_role_feasibility_20261006/report.md)既有234临时组件身份，不重新分配。私有manifest及源码hash见protocol.json；本轮只解析行号/角色/组件/折，不解析分子结构、标签、预测或真实特征。

|角色|行/组件|未来先导唯一用途|
|---|---:|---|
|fit|1403/191|点模型优化；三折OOF风险监督；合法参考|
|monitor|351/178|四个Chemprop模型早停与best checkpoint选择；不作为近邻或风险监督|
|calibration|585/188|整块保留：本先导不读取结构、标签或生成预测，不训练/选带宽/校准|
|evaluation|585/189|锁协议后取无标签结构生成证据、估目标密度；所有四臂分数落盘并锁hash后才开放标签评价|

source三折查询468/468/467，优化与参考935/935/936。full模型优化1403，预测evaluation585。三折点模型各排除本查询整组件；辅助活性RF和近邻严格使用同一折参考。任何外层角色/跨折参考越界、身份/hash变化或结构解析失败立即停止，不丢行重跑。

未来点模型与辅助RF沿旧phase-A的CSV `y`字段，同一尺度连接源误差和目标评价；不把它与M35统计悬崖用的`y [pEC50/pKi]`列混接。本轮不重读这两列。

旧M35已用于开发、源于已见开发结构，evaluation不称独立封存test；本轮只封住这次先导的标签反馈路径。组件不等于真正化学系列，更不等于时间。共享monitor与OOF训练重叠使误差相关，不能把1403点或分子对当独立重复。OOF/full训练规模变化保留为限制，不额外训练修补。

## 固定四臂与复用

所有风险RF使用**同一个未缩放7维输入、同一个OOF绝对误差监督**，200树、max_features=sqrt、min_samples_leaf=5、seed42、n_jobs=-1；其余默认值随现有sklearn1.9.0锁定。密度预处理不改变RF输入。

|臂|密度估权输入|用途|
|---|---|---|
|RF7|不估权，权重1|现有普通风险学习对照|
|IW-GEN5|7维证据的前5维|关闭粗糙度估权路径；RF仍保留粗糙度|
|IW-STRUCT7|同一Morgan2048，经source-only PCA7|标准结构空间重要性重加权基础对照|
|IW-RISK7|GEN5+两项邻域粗糙度|唯一适配|

IW-STRUCT7不是新模型：复用既有Morgan radius2/2048，sklearn PCA(n_components=7, svd_solver='full', whiten=False)，只在source1403个结构上fit；evaluation仅transform。选7为与候选估权维数相同，不做维数搜索。降维可能丢失结构信息，因此它只是一个有界基础对照，胜过它不等于胜过所有标准域适应方法。IW-GEN5是去路径消融，估权维度和带宽随维度变化，不能称密度估计容量完全相同；严格相同的是点模型与风险RF容量。各臂拥有相同原始数据访问预算，不宣称使用的信息完全相同。

证据空间拟合一个source-only StandardScaler；IW-GEN5复用该scaler前5列。结构PCA7输出另fit一个source-only StandardScaler。零方差列遵循库规则scale=1，不按结果删列；NaN/inf或结构PCA秩不足7即停止，不补维。

三个加权臂均复用固定提交SKADA DensityReweightAdapter，显式传KernelDensity(bandwidth='scott', kernel='gaussian', metric='euclidean', atol=0, rtol=0)。source与target密度分别fit，带宽各为n^(-1/(d+4))；源1403、目标585，d=5或7。没有用标签选择带宽，不将Scott规则冒称适合分子数据的已验证最优参数。除均值归一化不截断、不相对化、不加门控。该选择是事前工程规则，若失败不自动切到第二组带宽。

接口只复用adapter算权重，再对原始7维输入调用现有RF.fit(sample_weight=...)；不直接把skada.DensityReweight管线套在缩放/结构输入上，否则会把风险模型输入也改变。密度source与target标签全部屏蔽；query fixture延续旧入口y=0占位，真实源y仅在模型训练和独立OOF误差连接中使用。旧特征源码虽计算额外query-label诊断，输入占位使其不能接触真值，输出只白名单读取所需六项。monitor不进入任何特征。

SKADA/POT本机缺失，未来如安装必须使用隔离环境并保留既有Chemprop环境；本轮不安装。部署前还须验证固定提交adapter与锁定sklearn的兼容性，不能用“官方Ridge测试存在”代替实际通过。旧三任务phase-A入口不可直接执行；只复用其调用方式、固定模型参数、占位标签及排序规则。

## 资源上限与执行顺序

|操作|未来最大数量|行数或配置|
|---|---:|---|
|Chemprop fit|4|935/935/936/1403优化，分别monitor351；seed42；50epochs、patience15、batch64；其余沿旧2.2.3CLI默认并记录完整有效参数|
|Chemprop批量predict|4|468+468+467+585=1988；仅各best checkpoint|
|辅助活性RF fit|4|复用roughness作者200树、sqrt、seed0、其余固定默认；不是风险RF seed42|
|参考/查询特征批次|4|同上1988查询；K10、radius2/2048；参考935/935/936/1403|
|PCA / StandardScaler fit|1 / 2|只source，GEN5复用7维scaler子列|
|KDE fit|6|三个臂各source1403、target585|
|风险RF fit / predict|4 / 4|每臂1403源点、585目标点；不再为calibration拟合或推理|

四点模型共享给全部臂；不乘以四重复训练。4次辅助RF包含在4个特征批次里，不重复计作另一组流程。源/目标Morgan表示在臂间共享；逐分子输出仅留忽略目录。实际时间未知，历史随机折时间不作为本预算估计。

未来执行上限：每次Chemprop训练900秒、推理120秒、含辅助RF的特征批次300秒；密度/预处理/四风险臂与聚合评价合计600秒；整体5400秒。单步上限之和5880秒大于整体，因此整体可先触发停止，不承诺满额完成。最多4个GPU训练作业、串行1设备，GPU训练wall上限3600秒；不是GPU实际利用时间。无自动重试、无seed/task扩展、无搜索。失败/超时先保存作业状态和已完成hash，不自动放宽预算，不把部分完成当成功。

顺序：环境/接口合成检查→身份及标签隔离检查→4个点模型与4份特征缓存→三个密度权重→4个风险RF→锁全部分数及配置hash→一次标签评价。每个fit前递增日志计数。当前尚无执行入口和隔离环境，**这些是准备条件，不是训练已获授权**。

## 评价、解释与停止

主指标仍为50%—100%六点RMSE算术均值，原risk/canonical/source_row排序、ceil接受人数（585行分别293/351/410/468/527/585）。不调用旧metrics整体函数：它会额外读取calibration并计算区间和阈值，本先导没有该合同。可复用其排序/六点逻辑；不报告conformal保证、固定70%校准阈值或原cliff_mol悬崖分组成绩。

各对照公布六点、均值和全部接受交换的误差分解；100%四臂误差必须一致。候选均值必须严格低于三个对照，才具备讨论后续验证的必要条件；这不是显著性或充分Go，也不自动训练第二seed。若相等/退化、数值失败或只有部分臂完成，停止升级贡献，保留失败。

固定解释检查：每个候选-对照交换点计算共同/新增/剔除及精确平方误差贡献；列出最大单点绝对贡献。敏感性只移除该点后在同一有效集合按原比例重算两臂，不重训/重算权重；它是事后诊断，不是去异常部署规则。报告三组权重ESS=(sum w)^2/sum(w^2)、最大归一化份额及零权重数，不事后添加ESS阈值挑结果。非有限、均值非正、有效正权重不足2或adapter发出均匀回退警告则停止；不能静默当普通RF继续。Gaussian KDE处处正不证明真实支持重叠，ESS也不能验证条件误差稳定。

完整分区/互斥、交换人数相同、MSE恒等式、并列排序与100%一致性必须通过。解释仅归于邻域证据估权依赖和有限样本分布适配，不称化学因果、原子归因或已证明的系列迁移机制。

## 来源、核验及下一步

模型/特征契约来自本地run_phase_a.py和已核roughness快照，SKADA固定来源沿[M60记录](../series_risk_transfer_20261007/sources.json)。新工程规则据[KernelDensity](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)、[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)、[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)官方文档；网页当前1.9.1，本机1.9.0，Scott公式另核本机源码，未运行fit。参数选择不冒称来自原分子论文。

本轮新增report/protocol/check/verification四文件，仅追加四计划日志；模型、训练代码、历史报告、旧缓存、原项目和FP/LongPoly worktree不改。0fit/load/predict/真实特征/查询标签/官方test/新划分。下一步只需把此冻结协议接为薄适配和无训练dry-run，验证环境及标签隔离；真实四模型先导须明确授权。方法仍是待验证适配，不承诺二区/三区录用。
