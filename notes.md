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

M5c：64个真实SMILES的独立预测核对通过；故意错误的y列不读取，同设备与保存验证预测差异为零。匹配上游数值设置后完成验收，10项测试通过。

M5d：最小完整消融已先于执行冻结，36次缺失factorial组合+12次静态权重+6次FPPool容量对照。此设计履行三项机制完整验收范围；首轮Go仅控制额外任务扩展。

M5e：冻结队列串行执行，避免并行训练竞争GPU；每阶段审计后再继续。入口只读原数据，原队列不重启，已有输出不覆盖。

M5f：首队列8/8审计通过；单seed跨分子交互未超过原GraphCliff，不支持立即宣称收益。继续预定43/44和固定消融，不因阴性改参数。

M5g：提取协议/调度器此前sha256字段实际为normalized LF文本哈希，现改为明确字段并附Windows字节哈希，源码未动；运行manifest本来记录实际文件字节hash，既有训练身份不变。FPPool压缩与完整官方实现输出/梯度等价测试通过。
