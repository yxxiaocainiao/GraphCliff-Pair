# M63：条件粗糙度权重薄适配与合成dry-run

已实现[薄适配](../../../experiments/reliability_base/conditional_risk_weights.py)，对应[M62](../atlas_design_bridge_20261008/report.md)的唯一条件版本。直接复用已安装sklearn KernelDensity，不安装SKADA/POT，不重写核密度、不训练点预测器或风险RF。它准备的是**风险RF的源样本权重**，不是完整风险分数或可直接部署的拒答模型。

## 已实现的操作与接口

`joint_and_conditional_weights(source, target)`接收两个DataFrame：固定顺序七列GEN5+EXTRA2，index为各自唯一、相互不交的非负source_row。输入必须是**由source-only scaler处理后**的证据；函数不拟合scaler，不接收任何y列。调用者仍须核源/目标角色、邻域标签和缩放来源，列白名单无法证明上游数据没有泄漏。

函数在两端各fit一个7维Gaussian KDE，用各端自己的7维Scott带宽，再各fit一个前5列、同带宽的边缘KDE，共4次KDE fit。两个联合KDE同时供联合IW对照使用，不重复拟合，因此未来完整五臂仍可保持8次KDE，而非10次。返回联合/条件两组源权重、原log联合/边缘/条件比、带宽、条件权重ESS与最大份额；逐样本数组未来只落忽略目录。

log条件比=log联合比−log通用边缘比，归一化先减最大log权重再exp、最后除均值。该平移不改变理想归一化权重，不是截断或新稳定化模块；它避免不必要的exp上溢，但极小权重仍可能下溢。非有限或少于两个数值正权重会报错，不均匀回退、不调带宽挽救。不保证支持重叠、条件误差稳定或候选有效。

## 理论—消融—解释如何接入代码

|M62部分|对应输出或检查|当前边界|
|---|---|---|
|C1通用/粗糙度变化辨别|log_joint、log_generic、log_conditional三数组的代数恒等式|不是风险因果归因|
|C2条件适配与去核心操作|同一次联合KDE返回joint和conditional，后续可用相同RF输入/容量对照|本轮没有RF fit，未比较任何真实性能|
|C3机制与接受解释|复用旧排序代码前缀，合成核并列、ceil、交换人数、MSE分解与100%一致|手工合成分数仅检验评价接线，不能当条件权重产生的风险预测；训练活性反事实尚未实现|

该文件不提供读取数据、加载权重、运行Chemprop或RF的CLI。后续训练接线必须另核M61/M62合同，不能把源权重直接拿来排序目标分子。也不能将本轮sklearn接口通过说成已验证SKADA完整管线兼容。

## 合成检查、失败与补验

一个可运行[check.py](check.py)用两组小型合成证据（source12行、target8行）：共8次KDE fit，0Chemprop/辅助RF/风险RF。两组检查通过：有粗糙度变化时权重有限且归一化；固定R而改变G时，带宽带来的常数在归一化后消失，条件权重全1。另核两端Scott公式和log分解。

首次完整检查在完成这8次KDE后失败：Pandas3的字符串dtype不被`np.issubdtype`接受，抛TypeError；没有继续到额外拟合。改用Pandas成熟dtype判断，拒绝字符串、复数、布尔特征。之后两次`--validation-only`补验均0fit：第一次确认原错误修复，第二次补拒绝复数/布尔；最终9种错误输入在fit前拒绝，3种非法log权重拒绝。首轮8次KDE的数值检查不重跑，实际总量仍8，保留失败与分段核验记录，**不宣称修复后重新完整跑过全部检查**。

排序检查仅从旧run_phase_a.py的metrics函数AST提取calibration前的原代码，执行合成frame上的六比例；cal/cal_scores/cq全部传None，验证没有读取校准对象。并列先canonical后source_row；交换人数与平方误差贡献、100%结果一致。没有修改或运行原三任务入口，也没有读取真实样本标签。

完整复现（只拟合8次合成KDE）：

```powershell
& D:/Tools/conda-envs/chemprop_baseline/python.exe docs/research/conditional_adapter_dry_run_20261008/check.py
```

只复查输入/数值拒绝与旧排序（0fit）：同一命令末尾加`--validation-only`。默认检查有拟合合成KDE，不能把它称为所有类型的fit均为0。

## 改动与下一步

输入仅合成数组与既有代码/协议元数据；新增1个薄适配代码、1个check、本文及verification，追加四日志。原PDF、原项目、历史报告/协议、模型/训练入口、预测/权重和另一FPPool/LongPoly worktree不改。0真实fit/load/predict/真实标签或特征读取/新划分/test/安装/调参。

下一步是将这个接口接入**单任务五臂准备入口**，冻结新版本预算并做不运行训练的命令/身份检查；完整真实训练仍未授权。不再追加算法模块或第二候选。真实分子性能、充分ensemble对照、最终原子解释与跨seed稳定性仍未完成。
