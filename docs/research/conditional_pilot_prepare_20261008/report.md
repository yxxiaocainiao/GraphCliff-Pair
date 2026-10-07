# M64：234/seed42五臂准备入口

已新增[准备入口](../../../experiments/reliability_base/prepare_conditional_pilot.py)，另存[五臂协议](protocol.json)，保留M61四臂历史协议。它只生成作业身份、命令与权重接口绑定，**没有execute模式，当前尚不能直接训练**。

## 输入、输出与实际操作

输入为M61已固定的角色manifest、代码/配置hash，以及已安装Chemprop2.2.3的CLI参数解析器。只读取2924条行号/组件/角色/折元数据，未读取真实结构、活性、特征、预测、权重或官方test。

实际输出在忽略目录`D:/GraphCliff-Pair/artifacts/conditional_pilot_prepare_20261008/`：jobs.json、parsed_cli.json、arm_bindings.json、summary.json、state.json。个体身份和完整命令不公开；公开协议与聚合检查。该输出路径已核忽略规则，入口只允许当前项目artifacts内的新子目录，并拒绝覆盖已有输出。

|作业|未来优化/参考行|查询行|monitor行|
|---|---:|---:|---:|
|fold0|935|468|351|
|fold1|935|468|351|
|fold2|936|467|351|
|full|1403|585|351|

校准585行整块保留。四个点模型共享给RF7、IW-GEN5、IW-STRUCT7、IW-RISK7、IW-COND7五臂，不按臂重复训练。RISK7/COND7绑定M63同一`joint_and_conditional_weights`调用的joint/conditional输出；source-only缩放及原7维风险RF输入合同不变。只验证了绑定存在及输出键，**尚未执行五臂风险训练或预处理**。

## 参数、预算与检查

4条训练命令、4条预测命令全部由现有Chemprop解析器成功解析，未调用CLI执行handler。四条训练参数与历史有效默认值比较，仅data_path/output_dir改变；没有自行换学习率、网络尺寸或搜索参数。解析通过不等于输入文件已存在、GPU可用或训练成功。

新版本数量上限：4Chemprop fit/4批预测1988行、4辅助活性RF/4特征批次、5风险RF/5批风险预测2925行、8KDE、2scaler、1PCA。相对M61只增加1风险RF和2个边缘KDE，仍限定后处理600秒、整体5400秒及GPU训练wall3600秒；实际时长未知，超时停止，不保证在旧时间帽内完成。0自动重试、0任务/seed扩展、0参数搜索。

本轮自检：4个小型合成身份作业通过，4种非法manifest拒绝；真实manifest的组件/角色/折完整性及查询排除通过；权重接口AST绑定通过；现有依赖版本与冻结记录一致；拒绝覆盖与目录越界后，已有输出hash未改变。五臂候选均值低于四个对照仅为继续讨论的必要条件，不是显著性结论或自动进入第二seed。

导入CLI时出现“triton缺失，相关FLOP统计不可用”警告，未安装依赖或将其当训练失败。主准备1次成功，无模型运行失败或重试；目录拒绝是预定检查，不对原输出做清理或重建。

复现自检（0fit、无文件写入）：

```powershell
& D:/Tools/conda-envs/chemprop_baseline/python.exe experiments/reliability_base/prepare_conditional_pilot.py --selfcheck
```

准备命令同入口使用`--output artifacts/<新的子目录>`。已经生成的目录不能重用；失败状态留state.json，不自动补跑。不要把jobs.json中的训练命令当作本轮执行授权。

## 尚未完成与下一步

当前没有生成真实train_val.csv/query.csv/参考fixture，没有匹配的新OOF/evaluation特征、点模型或风险模型；GEN5/结构空间估权和五臂RF训练只有冻结规格，未写成实际执行流程。本轮也没有运行M63合成KDE或真实特征重算。

下一步应完成合法输入物化与有上限的单任务执行接线，再做不训练的检查；真实训练仍需用户明确授权。原历史协议/报告、模型/缓存/图、原项目及另一FPPool/LongPoly worktree不改。新增1准备代码、3研究文件，追加四日志；0真实或合成fit、权重load、模型predict、真实标签特征解析、官方test、安装与调参。创新和性能主张仍待真实证据。
