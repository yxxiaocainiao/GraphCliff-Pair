# 三机制消融：seed42 阶段结果

两任务共26组模型已独立审计：8组交互对照加18组后续消融。这里只报告一个开发种子，无标准差或显著性结论；seed43/44的36次消融仍须完成。参数及固定协议不据此调整，test尚未评估。

来源：[42次联合审计](validation_through_seed42_audit.json)，[机器可读结果](ablation_seed42_results.json)。Cliff RMSE为带cliff_mol标记分子的RMSE。

## CHEMBL234_Ki

|模型|Overall RMSE|Cliff RMSE|实际轮数|参数数|耗时秒|峰值CUDA MB|
|---|---:|---:|---:|---:|---:|---:|
|direct|0.6156|0.6704|45|6021198|140.5|380.2|
|global|0.7165|0.7475|29|6021198|160.7|539.1|
|pair_mlp|0.6804|0.7505|24|6284710|133.7|543.4|
|cross|0.7024|0.7508|22|6284878|134.3|542.1|
|global_fp|0.6801|0.6785|40|6153553|1824.3|662.3|
|cross_fp|0.7029|0.7497|43|6417233|1931.5|716.5|
|global_dynamic|0.7474|0.7722|27|6021198|163.9|520.2|
|cross_dynamic|0.6836|0.7366|86|6284878|578.6|527.8|
|global_fp_dynamic|0.6965|0.6730|88|6153553|3966.1|674.5|
|cross_fp_dynamic|0.7506|0.7969|21|6417233|943.1|716.5|
|global_fp_static|0.7320|0.7898|22|6153553|1053.9|662.3|
|cross_fp_static|0.7357|0.7739|21|6417233|991.2|716.5|
|pair_mlp_fp|0.7098|0.7211|30|6417065|1488.4|669.6|

条件差为 treatment − reference，负值表示该指标下降。完整列出所有组合及对照，避免只挑有利组合。

|机制或对照|reference|treatment|ΔOverall RMSE|ΔCliff RMSE|
|---|---|---|---:|---:|
|Cross-Attention|global|cross|-0.0141|+0.0032|
|Cross-Attention|global_dynamic|cross_dynamic|-0.0638|-0.0357|
|Cross-Attention|global_fp|cross_fp|+0.0228|+0.0712|
|Cross-Attention|global_fp_dynamic|cross_fp_dynamic|+0.0541|+0.1239|
|FPPool|global|global_fp|-0.0365|-0.0690|
|FPPool|global_dynamic|global_fp_dynamic|-0.0509|-0.0992|
|FPPool|cross|cross_fp|+0.0005|-0.0011|
|FPPool|cross_dynamic|cross_fp_dynamic|+0.0669|+0.0603|
|dynamic Loss|global|global_dynamic|+0.0308|+0.0247|
|dynamic Loss|global_fp|global_fp_dynamic|+0.0164|-0.0055|
|dynamic Loss|cross|cross_dynamic|-0.0188|-0.0142|
|dynamic Loss|cross_fp|cross_fp_dynamic|+0.0477|+0.0472|
|control|global_fp_static|global_fp_dynamic|-0.0355|-0.1168|
|control|cross_fp_static|cross_fp_dynamic|+0.0149|+0.0230|
|control|pair_mlp|cross|+0.0220|+0.0003|
|control|pair_mlp_fp|cross_fp|-0.0069|+0.0286|
|control|direct|cross_fp_dynamic|+0.1350|+0.1265|

## CHEMBL244_Ki

|模型|Overall RMSE|Cliff RMSE|实际轮数|参数数|耗时秒|峰值CUDA MB|
|---|---:|---:|---:|---:|---:|---:|
|direct|0.7312|0.8617|80|6021198|227.6|467.3|
|global|0.8941|1.0217|51|6021198|269.8|610.8|
|pair_mlp|0.8134|0.9503|59|6284710|308.9|625.8|
|cross|0.9034|1.0633|33|6284878|194.3|731.6|
|global_fp|0.9100|1.1013|35|6153553|1339.7|1036.9|
|cross_fp|0.9156|1.1142|21|6417233|794.9|1221.8|
|global_dynamic|0.9435|1.1409|18|6021198|103.0|608.8|
|cross_dynamic|0.9192|1.1242|22|6284878|138.3|709.7|
|global_fp_dynamic|0.9566|1.1776|16|6153553|598.4|1036.9|
|cross_fp_dynamic|0.9363|1.1507|17|6417233|642.6|1221.8|
|global_fp_static|0.9397|1.1299|35|6153553|1307.5|1036.9|
|cross_fp_static|0.9207|1.1315|21|6417233|796.7|1221.8|
|pair_mlp_fp|0.8919|1.0403|63|6417065|2350.0|1048.4|

条件差为 treatment − reference，负值表示该指标下降。完整列出所有组合及对照，避免只挑有利组合。

|机制或对照|reference|treatment|ΔOverall RMSE|ΔCliff RMSE|
|---|---|---|---:|---:|
|Cross-Attention|global|cross|+0.0093|+0.0415|
|Cross-Attention|global_dynamic|cross_dynamic|-0.0243|-0.0167|
|Cross-Attention|global_fp|cross_fp|+0.0057|+0.0129|
|Cross-Attention|global_fp_dynamic|cross_fp_dynamic|-0.0203|-0.0269|
|FPPool|global|global_fp|+0.0158|+0.0795|
|FPPool|global_dynamic|global_fp_dynamic|+0.0131|+0.0366|
|FPPool|cross|cross_fp|+0.0122|+0.0509|
|FPPool|cross_dynamic|cross_fp_dynamic|+0.0171|+0.0265|
|dynamic Loss|global|global_dynamic|+0.0493|+0.1192|
|dynamic Loss|global_fp|global_fp_dynamic|+0.0466|+0.0763|
|dynamic Loss|cross|cross_dynamic|+0.0158|+0.0609|
|dynamic Loss|cross_fp|cross_fp_dynamic|+0.0207|+0.0365|
|control|global_fp_static|global_fp_dynamic|+0.0169|+0.0477|
|control|cross_fp_static|cross_fp_dynamic|+0.0156|+0.0192|
|control|pair_mlp|cross|+0.0900|+0.1129|
|control|pair_mlp_fp|cross_fp|+0.0237|+0.0739|
|control|direct|cross_fp_dynamic|+0.2051|+0.2890|
