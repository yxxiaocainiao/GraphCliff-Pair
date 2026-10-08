# self-anchor先导：实现与复核完成，收益门槛未通过

2026-10-08。按计划C—F完成一个层内算子改动及8/8先导fit。主候选anchor在6项预登记严格比较中通过2项，**No-Go，不扩任务/seed、不组合FP、不调门控初始化或归一化参数补救**。这是当前一个开发任务和20epoch预算下的结论，不否定所有self/cross或残差方法。

## 改动与执行边界

层内响应从纯跨分子增量改成

$$B=U+\tanh(a_l)(V-U),\quad U=T(A_{self}),\ V=T(A_{cross}),\ a_l(0)=0.$$

保留self响应，每层增加一个有界有符号修正系数。相对同容量self只增加3参数；相对保留LongPoly的full增加789459参数，不能说相对原模型只加3。零门控前向退化到self，不意味着完整优化轨迹与self相同，新增门控及全局梯度裁剪仍可能改变训练。

[推导与反例](self_anchor_proposal_20261008.md)、[冻结执行协议](self_anchor_execution_20261008.md)、[真实批次预检](self_anchor_preflight_20261008.json)在运行前绑定提交`ab1b26a`。CPU/GPU契约及H256/3层真实批次四臂前向/反向检查通过，预检不做optimizer更新。既有4项机制契约与新增1项测试通过。门控本身不宣称原创，公式不保证预测收益。

只用234官方train开发拆分（2632训练/292验证查询），full/self/gated_delta/anchor四臂×两个独立进程，seed42，固定20epoch/100epoch调度、MSE与原优化参数。checkpoint按每臂20epoch内最小validation Overall MSE选取。所有对照都是Pair信息条件，不是单分子官方test成绩。不同于旧100epoch先导，只比较同期四臂。

## 全部结果

两次重复的指标、最佳epoch、保存预测CSV及checkpoint文件SHA逐臂一致。下表同时列两次，避免只展示一次有利结果。RMSE越低越好。

|重复|臂|Overall RMSE|Cliff RMSE|最佳epoch|
|---|---|---:|---:|---:|
|1|full|0.749560|0.817259|9|
|1|self|0.717925|0.784896|20|
|1|gated_delta|0.759319|0.807654|18|
|1|anchor|0.751436|0.811180|5|
|2|full|0.749560|0.817259|9|
|2|self|0.717925|0.784896|20|
|2|gated_delta|0.759319|0.807654|18|
|2|anchor|0.751436|0.811180|5|

继续门槛是“候选最差重复严格优于每个对照最好重复”，不是比较平均值。

|对照|Overall|Cliff|
|---|---|---|
|full|未通过（恶化0.25%）|通过（改善0.74%）|
|self|未通过（恶化4.67%）|未通过（恶化3.35%）|
|gated_delta|通过（改善1.04%）|未通过（恶化0.44%）|

保留self响应相对gated_delta有Overall局部改善，同时Cliff退化，未形成双指标贡献证据。anchor相对full只有Cliff局部改善，相对self双指标退化，不能把跨分子信息或新增门控说成有效主贡献。

最佳anchor checkpoint的三个有效门控均约为-0.00256、-0.00194、-0.00281，且两次相同。实际已学习到非零修正，不能说分支没有进入训练；系数负值仅描述有符号修正，不是化学机制或cross必然有害的证据。

## 核验、成本和限制

26个suite来源文件、配置/执行协议/预检SHA、干净启动commit、共同初始化及四臂head身份核验通过。额外核验输入CSV SHA和完整配对：两次拆分/配对相同，训练/验证查询互不混合，全部参考来自训练，未解析官方test标签。

8个checkpoint独立重放2336条验证预测，最大差8.881784197001252e-16；重新计算保存预测的Overall/Cliff并核对最佳轮、20epoch预算及参数量。机器可核验[结果JSON](self_anchor_results_20261008.json)由`tools/report_self_anchor_pilot.py`生成。

共160训练epoch、13280个optimizer步骤，fit记录耗时合计1632.69秒（27.21分钟，含训练/验证及checkpoint开销，不含实现与预检）。每fit平均耗时full158.73秒、self179.43秒、gated_delta234.22秒、anchor243.96秒。anchor相对self参数增量小，但本次平均耗时约高36%，不能称零额外计算成本。

两次重复均属于同一seed/机器/开发任务，不能做显著性或跨seed稳健性声明；20epoch也不代表充分收敛。没有官方test或独立新任务确认。M52历史前向分叉仍未定位，本次一致不撤销它。GPU缩批次的数值检查限制见设计文档，未改精度/Dropout、未追加复现诊断。

原始输出在`D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_self_anchor_screen_20261008/`，仅本方向工作目录可写；原始checkpoint/逐样本预测不推送。输入只读`D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv`。原模型、训练器、Loss、历史配置/报告及可靠性目录/计划未修改，不合并。

本轮计划C/D/E/F完成；本候选停止，当前无剩余训练队列。不把这次阴性改写为最终模型，也不自动再设计一个门控/尺度变体。后续若继续，先提交新的问题与可判定证据，再立执行计划。
