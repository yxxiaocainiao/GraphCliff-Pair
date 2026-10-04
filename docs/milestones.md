# Milestone log

## M0 — 2026-10-04

建立终极目标、只读原项目边界、分阶段验收、来源记录和上传排除规则。源码/实验尚未实现；后续逐项记录验证结果与 GitHub commit。

## M1 — 2026-10-04

固定官方 GraphCliff 特征、编码器和 MIT；提取既有 split_train_valid。增加严格构图检查、train-only Top-1 配对、自身/canonical 排除及并列按行号处理。两任务与既有划分逐行一致；234: 2632/292，244: 2229/247，共 5400 开发图，无丢弃、跨 train/valid canonical 重复为零。2 项契约测试通过。数据/配对清单只保存在 ignored artifacts。
