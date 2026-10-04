# GraphCliff-Pair

基于 GraphCliff 编码器的参考分子差值回归研究项目。逐项验证跨分子 Cross-Attention、FPPool、动态加权 Loss，保留原短长程编码机制。

**状态：项目初始化；尚无新模型效果结论。**

终极目标与进度见 [task_plan.md](task_plan.md)，来源与决策见 [notes.md](notes.md)。本仓库公开，只发布源码、配置、来源记录和验证摘要。数据与训练权重不随仓库发布。

预测契约：`delta(query, reference) = y_query - y_reference`，`y_hat = y_reference + delta_hat`。参考来自同任务训练集，按结构选择；查询活性不传入模型。

## Sources

- [GraphCliff official code](https://github.com/dmis-lab/GraphCliff)：复用分子特征、编码器及原读出。
- [FPPool official code](https://github.com/shenwxlab/FPPool)：计划复用指纹分层池化，不自行重写算法。
- [SQRL paper](https://arxiv.org/html/2501.09103v1)：参考相对回归的任务定义；不是官方复现。
- [Siamese-Regression-Pairing](https://github.com/AstraZeneca/Siamese-Regression-Pairing)：参考近邻配对流程，是否借用具体源码需逐文件检查。
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：使用库内注意力运算。
- [Scientific Agent Skills](https://arxiv.org/abs/2609.00065)：实验设计及 PyG 接口的程序性指导。
