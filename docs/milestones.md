# Milestone log

## M0 — 2026-10-04

建立终极目标、只读原项目边界、分阶段验收、来源记录和上传排除规则。源码/实验尚未实现；后续逐项记录验证结果与 GitHub commit。

## M1 — 2026-10-04

固定官方 GraphCliff 特征、编码器和 MIT；提取既有 split_train_valid。增加严格构图检查、train-only Top-1 配对、自身/canonical 排除及并列按行号处理。两任务与既有划分逐行一致；234: 2632/292，244: 2229/247，共 5400 开发图，无丢弃、跨 train/valid canonical 重复为零。2 项契约测试通过。数据/配对清单只保存在 ignored artifacts。

## M2 — 2026-10-04

新增 global_diff、参数量匹配 pair_mlp、双向 Cross-Attention；复用官方编码器/SAG/MaxMean，统一反对称 head。5 项测试通过，含 ragged padding、配对隔离、交换符号、同分子零值、梯度及重载。发现官方 LongPoly 的空边分支导致无键分子的批次依赖；在外部包装层为含无键图的批次单图编码，保留上游原样。新增参数匹配见 m2_validation.json，不声称计算量一致。官方快照含原始空白字符，保留以保证哈希；新增代码单独检查 whitespace。

## M3 — 2026-10-04

复用本地 FPPool adapter 和 Morgan 归属函数；官方池化用固定哈希外部下载。增加普通/静态/动态差值 MSE，候选公式在 design.md，默认关闭；关闭和 epoch0 时的值及梯度与 MSE 相同。7 项测试通过。第三方源码 external 未跟踪上传。
