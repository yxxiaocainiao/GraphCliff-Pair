# 实现身份与实验契约

终极目标见 task_plan.md。本设计是本项目候选，不宣称三模块组合具有新颖性。

## 模型

复用固定官方 GraphCliff 的 AtomEncoder、GraphCliffEncoder、SAGPooling 与 Max/Mean。查询和训练参考共享参数。Cross-Attention 在完整原子表示上双向调用同一个 PyTorch MultiheadAttention，Q 来自当前分子，K/V 来自另一分子；padding 只掩蔽 K/V，每对在独立 batch 位置计算，然后取真实节点。注意力残差 + LayerNorm 后读出。

所有 pair 组使用 `delta = (h(q,r)-h(r,q))/2`。global_diff/cross_attention 的 head 输入读出差；pair_mlp 输入两个读出及其差。head/attention 无 dropout；官方 LongPoly 内部仍保留其默认 dropout=0.1，故交换反对称与同分子零值保证针对 eval 推理，训练随机前向不保证逐次严格相等。global_diff 为 SQRL 式任务适配，使用 Top-1 配对而非论文所有相似度阈值对，且 head 强制反对称，不能称官方复现。

pair_mlp 的新增 head 参数匹配 attention 的 head + interaction，误差 <0.5%；仅与同读出组比较。共享编码器/SAG 初始权重须显式加载和记录哈希。参数匹配不等于计算预算匹配。

含无键图的批次使用单图编码外部适配，避免官方 LongPoly 空边分支随批次改变；vendor 原样。

## FPPool

使用官方 FingerprintPool 固定版本、atoms_repr=True、Morgan radius=2/1024 位。已有本地 adapter 压缩未激活位，按分子调用官方池化，再投影到 2H。原子归属使用原 Morgan 环境映射，先核对 GraphCliff 节点和边顺序。公开仓库不含官方 FPPool 快照，下载脚本核对 SHA256。此版本为 Morgan-only 适配，不称官方多指纹完整模型复现。

## 动态 Loss 候选

目标始终为 query 活性减 reference 活性，恢复为 reference 活性加预测差。首轮默认为普通 MSE。

候选权重：`severity_i = min(abs(delta_y_i)/s, cap)`；`s = max(median(abs(train_pair_delta)), 1e-6)`，只用训练对计算。

`alpha(epoch) = alpha_max * min(epoch/warmup_epochs, 1)`；`w_i = (1 + alpha * severity_i) / batch_mean(1 + alpha * severity)`；`L = mean(w_i * (delta_hat_i-delta_y_i)^2)`。

候选 alpha_max=1、warmup=10、cap=3 均是明确的实验选择，不是 GraphCliff/FPPool/SQRL 官方参数，不是已知最优。权重不接受验证/测试活性，不依赖模型残差，按轮次从普通回归逐渐增加大差值对的关注。归一化防止平均 loss 权重随训练抬升；有限 cap 限制极端标签。batch 内归一化本身仍会改变样本相对权重，batch 大小必须固定。

static 对照始终使用 alpha_max，用于区分“加权”与“动态日程”的贡献。mse 或 alpha_max=0 必须与原 MSE 值/梯度一致。

风险：大差值可能包含实验噪声，也可能是结构不够相似的近邻；该权重不是官方 cliff 判定，也不保证改善 cliff。须比较普通 MSE、静态权重、动态权重，报告权重分布。

## 评价

参考按 Morgan Tanimoto Top-1，仅来自本任务 train，排除 canonical 相同记录，同分并列按原行号。不依据活性、cliff_mol 或预测误差检索。

保持既有 train 内 10% validation 划分，cliff_mol 分层来自 benchmark 标签，需披露；不作为输入、检索条件或 Loss 权重。选模只用 unweighted validation Overall MSE。报告同一 checkpoint 的 Overall/Cliff/Non-cliff RMSE、MAE。查询—训练参考的直接 delta 误差只作诊断，不能冒充验证内部 pairs 指标。

完整比较：direct baseline、Top-1 标签、global_diff、matched pair_mlp、cross_attention；后续完整 2×2×2 attention/FPPool/dynamic 消融，另加 static 权重以检验日程。研究复现单位是任务×seed，分子对共享端点不能当独立重复。

## 首轮实施边界

先做两任务小规模真实数据 smoke，验证能训练、重载、恢复预测；不是效果评估。之后独立冻结完整训练矩阵，再决定是否扩大。默认预算候选为既有 pilot：100 epoch、patience15、batch32、AdamW(lr=1e-4, weight_decay=1e-5, betas=0.9/0.95)、clip1、warmup10/cosine。它不是论文精确复现协议。不得根据 test 修改配置。
