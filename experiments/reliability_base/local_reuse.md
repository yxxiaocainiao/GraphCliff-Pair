# M22：优先复用my_work已有资产

2026-10-05。输入：`D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines`、同级MoleculeACE和FPPooling目录；输出：当前项目本目录。原目录保持只读。本轮采用已有Chemprop官方CLI作为后续主活性基线，不从零实现D-MPNN，不因为旧测试成绩选择模型。

|资产|本轮确认|用途与边界|
|---|---|---|
|Chemprop D-MPNN|30任务manifest/checkpoint哈希相符，seed42，train/val文件存在；本机官方包2.2.3导入通过|优先活性预测底座；不是创新贡献或当前多种子效果证据|
|MoleculeACE|本地MIT许可证、现成SVM/特征/配置入口|传统对照与数据协议；不复制整个项目|
|GCN/GAT/MLP|各3任务×3种子，共27组配置/manifest/权重/validation文件存在|按需公平对照；本轮只查存在，不重算历史成绩|
|FPPooling|本地目录已定位|历史资产；本轮不重新接入失败路线，不追加模块|
|本项目现成工具|tools/reuse_baselines.py和verify_baseline_reuse.py已存在|复用划分/行号身份检查与验证指标重算，不再新写同类工具|

固定来源：本地来源清单记录Chemprop commit c83f4838cfc05a072f6741ae5b9700a1453ed571、MoleculeACE commit 7e6de0bd2968c56589c580f2a397f01c531ede26；源码缓存版本与实际安装包2.2.3分别记录，不默认二者字节相同。Chemprop和MoleculeACE许可证均核查原文件/hash。基线本地.git不能解析HEAD，不能捏造本地提交；以文件hash及已有upstream来源绑定。

作者/官方代码已在本地，不需要重新下载模型。本机2.2.3的EnsembleEstimator/MVEEstimator导出检查通过；校准类的名称/接口未确认，不强行补一个实现。官方[不确定性教程](https://chemprop.readthedocs.io/en/latest/uncertainty.html)当前为2.3.1，列出不确定性预测头和示例，作为方向参考，使用前须核对2.2.3契约；本轮没有运行UQ/解释系统。固定GitHub树网页获取失败cache miss，不宣称重新验证了远端源码树。

**关键边界**：旧validation用于早停/最佳权重选择，不能直接把该集称独立校准。已有权重可用于工程接口或标记为回顾性的探索；正式风险实验要先固定外层开发划分、内层模型选模、OOF风险标签/邻居标签和独立校准/评价角色。旧test结果只作历史记录，本轮未读取其预测或成绩；任务和继续门槛不得依此选。M22没有重做逐行划分身份，下一步复用现成检查再接入。主干选择依据本地可运行/许可/现有入口，不据性能择优。

计划/实际：30/30 checkpoint哈希、27/27神经资产存在检查；0训练、0新预测、0test。本轮公开的[清单](local_reuse_inventory.json)只含身份、hash及检查状态，不含逐分子数据、预测和权重。

异常如实补记：首次清点脚本误填CHEMBL244_Ki为神经历史任务而断言失败，纠正为读取未改动neural_pilot.json（CHEMBL3979_EC50/CHEMBL4792_Ki/CHEMBL234_Ki）；第二次因基线无可解析HEAD中断，改记未知；第三次只读清点通过。没有训练失败或新训练，未用补跑结果凑资产。Chemprop导入出现可选Triton计数警告，导入正常。

下一步仍是有限基线实验方案：固定一个主干、任务/seed/预算/风险对照、无泄漏训练/校准角色和数值继续/停止条件，再跑正式实验。可解释性、随机掩蔽忠实度、跨seed稳定性及失败案例仍为最终必需。具体创新机制尚未确定，成熟主干可复用不等于已具备论文贡献。
