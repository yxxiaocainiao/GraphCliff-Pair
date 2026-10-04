# 已完成42次训练的补充诊断

独立重算既定Top-1参考标签基线，并按冻结shuffle/批次/Loss重建12个加权模型的逐轮权重分布。全部42个模型（含30个普通MSE模型）逐轮权重极值与history核对通过。仅使用训练和验证标签；未执行test。

来源身份和完整加权分布见[机器可读记录](validation_diagnostics_42.json)。

## 无训练的最近邻参考标签对照

按固定结构Top-1直接预测训练参考活性。同一划分下各训练seed相同，每任务只报告一次，不计算三种子SD。

|任务|Overall RMSE|Cliff RMSE|Non-cliff RMSE|MAE|查询数|
|---|---:|---:|---:|---:|---:|
|CHEMBL234_Ki|0.7867|0.8841|0.7013|0.5902|292|
|CHEMBL244_Ki|0.9552|1.1480|0.7358|0.6677|247|

## 既定最佳checkpoint所在epoch的训练权重

权重无量纲；batch内均值归一为1。完整逐轮分位数见JSON；下表按训练查询统计，不计算验证权重。CPU重建与原CUDA日志比较允许float32舍入差。

|任务|组|seed|epoch|min|median|q95|max|
|---|---|---:|---:|---:|---:|---:|---:|
|CHEMBL234_Ki|global_dynamic|42|12|0.3895|0.8956|1.7728|2.1049|
|CHEMBL234_Ki|cross_dynamic|42|71|0.3591|0.8995|1.7660|2.1040|
|CHEMBL234_Ki|global_fp_dynamic|42|73|0.3924|0.8949|1.7687|2.1120|
|CHEMBL234_Ki|cross_fp_dynamic|42|6|0.4874|0.9281|1.5818|1.8373|
|CHEMBL234_Ki|global_fp_static|42|7|0.3878|0.8965|1.8023|2.0945|
|CHEMBL234_Ki|cross_fp_static|42|6|0.3656|0.9070|1.7611|2.1353|
|CHEMBL244_Ki|global_dynamic|42|3|0.6755|0.9456|1.3801|1.5206|
|CHEMBL244_Ki|cross_dynamic|42|7|0.4818|0.8995|1.6800|1.8536|
|CHEMBL244_Ki|global_fp_dynamic|42|1|0.8631|0.9766|1.1540|1.1917|
|CHEMBL244_Ki|cross_fp_dynamic|42|2|0.7394|0.9591|1.2808|1.3627|
|CHEMBL244_Ki|global_fp_static|42|20|0.3842|0.8809|1.7957|2.1634|
|CHEMBL244_Ki|cross_fp_static|42|6|0.3675|0.8824|1.7923|2.1131|

此诊断补齐报告信息，不改变训练结果、最佳轮次或既定三种子范围。完整78次结果尚待剩余训练；不能把此阶段表视为最终消融结论。
