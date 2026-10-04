# Findings and decisions

2026-10-04：用户明确底座 GraphCliff，三项机制为跨分子 Cross-Attention、FPPool、动态加权 Loss；授权按小里程碑推送，随后明确公开仓库。

原项目存在 README 修改与大量 untracked 研究文件。只读引用，全部留在原处。

当前可用解释器 D:/Tools/conda-envs/graphcliff/python.exe：torch 2.7.1+cu128、PyG 2.6.1、RDKit 2025.03.6；CUDA 可用，RTX 5060 Laptop 8 GB。不升级已有环境。

GraphCliff 官方 MIT，当前 main=d18857db0303346c0be715a56febf71da1768592。本地 HEAD=c18f19c0ca557c8c82bc015a66f19f06543311a2。

FPPool 官方 main=ef2afe82e0490dc31a7cbb6146acea810de393fe，README 声明 MIT，但仓库许可证 API 未识别独立许可证。公开项目先提供固定版本下载脚本和来源说明，不把本地官方源码快照直接重新发布。

SQRL 原文已核对：全局表示差 + MSE + 训练近邻恢复活性已有先例；本项目不宣称参考回归本身新颖，也不把独立适配称官方 SQRL 复现。

M2：批次隔离测试捕获上游无键图边界差异，已用外部包装修复，不修改 vendor。根 LICENSE 同官方 MIT，版权保留。

M4：原 LongPoly 的 dropout=0.1 即使上层 dropout=0 仍保留，因此反对称/同分子零值保证针对 eval 推理；训练时有随机性。共享 head 从原 base 显式加载，训练前重新设置随机种子；attention 使用 math backend。smoke 第一版已被初始化更严格的第二版取代，均不作效果结论。

M5b：独立报告工具已通过16组真实数据输出核对和9项契约测试。辅助验证对定义固定为Morgan radius2/1024 Tanimoto>=0.8，方向y_b-y_a；新增时间晚于训练启动，不称预注册主指标，不修改主门槛。
