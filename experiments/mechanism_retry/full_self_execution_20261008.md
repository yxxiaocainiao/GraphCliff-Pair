# M60执行前协议：full/self基础信号多seed确认

目的：判断保留LongPoly的Pair对照full和self替换的差异是否在两个开发任务、三个训练seed中保持。不是新方法收益试验，不恢复centered/context/anchor，不修改旧No-Go。

- 矩阵固定：CHEMBL234_Ki、CHEMBL244_Ki × seed42/43/44 × full/self，共12fit。训练按任务→seed→full/self依次运行，全部新训练，不混用旧checkpoint。
- 输入仅D:/GraphCliff-main/benchmark_data的上述CSV；沿用read_safe仅解析官方train标签，split_seed42内部90/10划分。234为2632/292、244为2229/247。查询配对到训练集结构最近参考，排除自身及相同规范SMILES，tie按原行号。两臂同一参考及参考活性恢复条件，官方test封闭。
- full保留LongPoly；self沿用已有LayerSwap的self-attention逐层替换，保留ShortGINE、门控、残差及SAG+max/mean读出、反对称差值head。self仍是Pair差值预测条件，只是分子内交互，不是单分子官方GraphCliff结果。
- 不改模型：H256、3层、4heads，batch32、普通差值MSE；AdamW lr1e-4、betas(.9,.95)、weight_decay1e-5、clip1；warmup10、schedule100、min_lr1e-6，最多100epoch、patience15，按每臂最低验证Overall MSE选checkpoint，不按Cliff挑选。
- 同seed两臂从同一官方GraphCliff随机base_state加载共有张量；逐张量核验交集一致，head初始化一致。完整模型容量不同：full6021198、self6810654；不声称匹配容量。训练前重置seed及同seed批次乱序生成器。
- 使用现有native amax读出次梯度规则及strict deterministic algorithms，保持high矩阵精度，不承诺GPU全程逐位。历史同seed分叉仍保留。三个seed体现初始化/训练随机性，在同一个固定划分上，不是三份独立数据集；不加同seed重复，不宣称显著性或置信泛化。
- 预检：两任务×三seed×两臂12个真实batch前后向，损失/梯度有限、参数数及共有初始化一致，不做optimizer更新。通过后冻结源码/输入/配置/协议/预检SHA，从干净提交训练。新增工具复用训练器，不修改历史训练入口。
- 若严格算子异常、非有限值、来源变化或身份检查失败则停止并报告，不能放宽精度/确定性规则后续跑；不开展新的数值根因支线。正常指标好坏均完成12fit，不中途换任务、臂、seed、预算或门槛。未达到100epoch的正常early stopping不算失败。

## 预登记开发筛查规则

两个任务分别计算每臂Overall/Cliff跨三seed算术均值和样本标准差(ddof=1)。每任务须同时满足：两项self平均RMSE严格低于full；同seed比较中至少2/3 seed的Overall和Cliff同时严格改善。两任务均通过才称基础信号筛查通过。无浮点四舍五入后判断、不按单一最佳seed判断、不跨任务平均掩盖失败、不改旧门槛。不通过则停止本轮架构方向；通过也仅进入容量/算子混杂评估，不直接称创新或最终模型。

## 交付与核验

正式产物仅在本worktree artifacts/mechanism_retry_full_self_20261008。独立报告需audit全部12fit，复核历史长度与最优epoch、输入SHA/拆分/配对与训练参考合法性、同seedhead/共有初始化身份及源码SHA；strict加载全部best.pt、固定batch32重放所有验证预测，atol=rtol=1e-5。完整列出12行、均值/标准差、同seed差值、门槛、参数/耗时/显存、实际epoch/optimizer步骤及nearest-reference上下文，不能把参考活性增益说成跨分子交互贡献。

报告写experiments/mechanism_retry/full_self_results_20261008.md/.json；权重与原始逐样本预测仅留忽略目录。独立分支codex/residual-fp-centered-cross-20261007，不写可靠性目录、不合并。
