# M28：同维度粗糙度块置乱（拟合前冻结）

2026-10-06，用户授权完整计划的第一个实验。三任务seed42、原缓存与RF参数固定；新增18fit=3真实重放+15置乱，不训Chemprop/不读test。先重放每任务真实RF，cal+evaluation风险预测max误差≤1e-10才继续该任务；重放不计独立重复。二列nbr_disp/sali_mean同块置乱，每角色独立PCG64种子按JSON确定，排序source_row后置乱；不变y/cliff/prediction/其他特征。5重复固定编号0–4，不搜索或换seed。

所有身份、缓存绑定、canonical互斥、有限输入在首fit前检查；执行ledger在每fit开始前计数，失败保留，不自动重跑，失败fits也占18。配置、版本、代码hash、排列hash、source/donor映射、风险输出和权重保存私有忽略目录。运行前提交JSON和代码，不观察新成绩修改协议。主曲线六接受比例，ties继承canonical/source_row；每任务全部5重复+真实结果公开。继续条件至少2任务真实curve_mean_rmse小于5置乱中位数，且三任务平均(median-true)>0。均为预算筛查，非统计显著性或期刊标准。

置乱破坏粗糙度和普通特征的关联，可能产生不自然特征组合；因此只检查模型对正确匹配的依赖，不是化学因果，也不是完备条件独立检验。通过不等于算法创新，失败不等于粗糙度所有用途无效。失败暂停增强候选；有界三问题审查仍需交付。完整计划见../../docs/research/reliability_publication_plan.md。

计划18fit/实际0，0新主干/0官方test；输入来源与hash、软件及配置见permutation_protocol.json。旧runner/协议及GraphCliff/my_work不变。采用experimental-design技能的配对和正确重复层级规范，工具引用继承M27报告。
