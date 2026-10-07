# self上下文共同尺度：真实状态诊断与新三臂先导方案

2026-10-08。用户在机制设计说明后明确要求“ok继续”。先完成已有checkpoint诊断；结果支持检验幅度响应动机，因此固定下面的新三臂先导，在方案提交后才训练。旧FP/centered先导No-Go不改，本次不启动原8次seed扩训或FP组合。

## 1. 真实状态是否对应构造问题

对234/244原centered最佳checkpoint的全部292/247个验证查询及其训练参考，在3层、双方向记录MHA的cross/self输出；hooks只观测，不改forward。两个checkpoint全部539条预测重放通过，与保存值最大绝对差小于1e-14。12个分层/方向组共103785次节点出现，含重复参考，不是独立样本数量。

对真实差分方向delta离线使用t=0.01、0.1、1，比较旧/候选变换输出的节点RMS。定义幅度响应指数E=log10(RMS[B(0.1delta)]/RMS[B(0.01delta)])，E=1表示放大10倍，E=0表示不变。零响应另计，不以它们计算log；JSON保留finite_count。固定t的算子扰动不是修改分子，也不是训练新模型。没有用标签选择节点、层或t。

|任务|层|方向|节点出现数|self仿射RMS第1百分位|旧E中位数|候选E中位数|
|---|---:|---|---:|---:|---:|---:|
|234|1|query|8644|0.321524|0.947078|0.999926|
|234|1|reference|8628|0.321882|0.947646|0.999930|
|234|2|query|8644|0.350477|0.208422|0.999080|
|234|2|reference|8628|0.372271|0.208962|0.998748|
|234|3|query|8644|0.594646|0.319452|0.999081|
|234|3|reference|8628|0.541631|0.318804|0.999259|
|244|1|query|8695|0.360806|0.987125|1.000022|
|244|1|reference|8628|0.366296|0.985864|0.999971|
|244|2|query|8695|0.364719|0.151460|0.998896|
|244|2|reference|8628|0.360983|0.148509|0.998866|
|244|3|query|8695|0.349873|0.056712|0.996799|
|244|3|reference|8628|0.352900|0.057317|0.996918|

第一层旧算子接近线性响应；后两层明显压缩这一幅度变化，候选的中位响应仍接近10倍。self分母中eps的最大占比约1.53e-6，所测节点没有eps占比达到1%的情况。这支持“幅度效应在已训练隐藏状态中也存在”和“这些状态中self尺度未退化”，不证明恢复幅度对预测有益，更不证明它造成234的退化。这里没有新候选训练后的状态，训练后尺度仍可能退化。

完整分位数、覆盖、checkpoint/hash和重放见[self_context_scale_diagnosis_20261008.json](self_context_scale_diagnosis_20261008.json)。诊断当时的原型hash对应设计提交2061af7；随后将相同函数移入实际候选模块，构造检查重新通过，没有重写诊断结果冒充新模型结果。

## 2. 唯一候选及实现边界

候选context沿用[设计公式](self_context_scale_proposal_20261008.md)，只改变差分变换的尺度来源及取差位置。实现context_scale.py，原model.py/run.py/训练器及旧配置不改。共同尺度可微，没有detach，没有新参数、Loss或可调门控。每层双方向4次MHA，与旧centered相同；self为2次，不声称等计算。

为避免两个实现漂移，将构造检查的contextual_difference移入实际候选模块，检查脚本直接导入它。三臂均6,810,654参数，全部初始化hash为30d154fb519f3750e5c24dfcd8efea743b75da78573c7e4eeb4d1c96b113a098。H32契约检查覆盖反对称/同输入零/空边与batch隔离/梯度/重载；H256、3层、真实最大训练图245节点batch32前后向有限，候选峰值1551.29 MiB。预检优化步骤0，详见[预检JSON](context_scale_preflight_20261008.json)。

## 3. 执行前固定的新问题和预算

新问题：共同self尺度的候选能否在同初始化、同优化预算下，同时优于self与旧centered的Overall和Cliff？主候选固定context，不能结果后改选某臂。

- 234/244 × self/centered/context × seed42，共6fit；全部从相同随机初始化训练，两个旧臂也同期重跑，不拿旧checkpoint充当本轮训练对照。
- [context_scale_screen.json](context_scale_screen.json)完全沿用H256/3层/4heads/普通MSE/batch32/lr1e-4/100epoch上限/patience15/split_seed42等参数；只有臂集合和状态说明改变。
- 两任务、两指标、相对self和centered，共8项严格小于比较必须全部通过，才建议后续补证。任一失败即不扩大本候选，不调eps、Loss或组合救结果。
- checkpoint仍只按validation Overall MSE选择；参考仍为训练结构Top-1，参考活性只在模型输出后恢复预测。不解析官方test标签，不改变开发划分。
- 单seed、反复使用的开发任务且是旧诊断导出的新方案，不当独立确认或显著性证据。通过也不自动运行新seed/任务、full/cross补充对照或FP组合，更不能反过来称旧12项门槛已通过。

输入只读D:/GraphCliff-main/benchmark_data的两个CSV，hash及覆盖同预检。输出只写D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_context_scale_screen_20261008；目录存在则拒绝覆盖，失败保留现场，不自动减batch或变精度。先提交本方案和实现，再启动；不在运行中改实验源码。

```powershell
Set-Location D:\GraphCliff-Pair-FPPool-LongPoly
& D:\Tools\conda-envs\graphcliff\python.exe -m tools.diagnose_self_context_scale artifacts/mechanism_retry_layer_core_screen_20261007 --json-output artifacts/context_scale_rediagnosis.json
& D:\Tools\conda-envs\graphcliff\python.exe -m unittest discover -s tests -p test_context_scale.py -v
& D:\Tools\conda-envs\graphcliff\python.exe -u -m experiments.mechanism_retry.context_scale --config experiments/mechanism_retry/context_scale_screen.json --csv-root D:\GraphCliff-main\benchmark_data --output artifacts/mechanism_retry_context_scale_screen_20261008
```

## 4. 验收和责任范围

完成6fit后重算三组指标、核对输入/查询参考/覆盖/预算/初始化/来源，独立重放6个最佳checkpoint；报告新候选相对两对照的8项判断、耗时和显存及阴性结果。旧两臂与原先导的重跑差异如实记录，不假设GPU逐位可重复。

本轮只改本方向worktree.md、tools/check_self_context_scale.py，新增context_scale.py、三臂配置与预检、本方案、诊断JSON及工具、tests/test_context_scale.py；结果另行提交本方向报告工具/报告。不写可靠性目录或计划，不上传数据、权重或逐分子预测，不合并分支。
