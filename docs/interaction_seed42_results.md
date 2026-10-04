# 首轮交互验证：seed42

日期：2026-10-04。8次完整预算训练完成并通过独立审计，尚未评估test。

**当前单种子中，Cross-Attention 未超过原 GraphCliff，且未同时超过全局差与参数匹配 MLP。** 这是开发集的早期阴性信号，不是三种子最终结论；保持冻结配置继续43/44，不调参、不删结果。

|任务|模型|Overall RMSE|Cliff RMSE|MAE|最佳epoch|实际epoch|秒|
|---|---|---:|---:|---:|---:|---:|---:|
|CHEMBL234_Ki|Top-1参考标签|0.7867|0.8841|0.5902|—|—|—|
|CHEMBL234_Ki|原GraphCliff|0.6156|0.6704|0.4629|30|45|140.5|
|CHEMBL234_Ki|全局表示差|0.7165|0.7475|0.5580|14|29|160.7|
|CHEMBL234_Ki|参数匹配pair MLP|0.6804|0.7505|0.5053|9|24|133.7|
|CHEMBL234_Ki|Cross-Attention|0.7024|0.7508|0.5389|7|22|134.3|
|CHEMBL244_Ki|Top-1参考标签|0.9552|1.1480|0.6677|—|—|—|
|CHEMBL244_Ki|原GraphCliff|0.7312|0.8617|0.5175|65|80|227.6|
|CHEMBL244_Ki|全局表示差|0.8941|1.0217|0.6972|36|51|269.8|
|CHEMBL244_Ki|参数匹配pair MLP|0.8134|0.9503|0.5976|44|59|308.9|
|CHEMBL244_Ki|Cross-Attention|0.9034|1.0633|0.6577|18|33|194.3|

## 核对与限制

- 数据哈希、原train/valid行号、完整Top-1配对、train-only参考、标签/cliff顺序、共享编码器及兼容head初值、最低验证Overall MSE和预测重算均通过。
- 234 Ki验证292个分子/128个cliff分子；244 Ki验证247个分子/118个cliff分子。
- 所有指标从同一个最佳checkpoint得到。direct是绝对活性MSE，pair组是差值MSE并加训练参考活性恢复；两种任务形式不能称完全相同。
- direct/global参数6,021,198，pair MLP6,284,710，cross6,284,878；新增参数匹配不代表计算预算匹配。实际图前向、optimizer steps、峰值显存和辅助对指标见JSON。
- 时间为本机实际墙钟时间，包含验证与保存，期间也执行过独立接口核对；不作为严格硬件吞吐排名。
- 当前只一个seed、两个既有开发任务，不作显著性或泛化结论。辅助Morgan对指标不是官方Cliff RMSE，不用分子对数量虚增独立重复。
- FPPool/动态Loss尚无完整预算结果，不能从这张表判定其贡献。

## 来源

训练代码提交：`17c8bd388140200807a7d1133573fadcd746cbbe`；本队列启动时工作树干净。配置：configs/interaction_seed42.json。公开审计与源码/输入哈希见 [interaction_seed42_audit.json](interaction_seed42_audit.json)。原始配对、预测和checkpoint只在本机 artifacts/interaction_seed42_20261004，未公开上传。

下一步：完成seed43/44后合并24次交互结果，再按已冻结配置完成三项机制消融。
