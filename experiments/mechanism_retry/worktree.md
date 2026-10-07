# FPPool / LongPoly 独立工作目录

2026-10-07，用户要求与可靠性对话隔离可写目录，且只提交本方向文件。

## 目录和提交边界

- 本方向：`D:/GraphCliff-Pair-FPPool-LongPoly`，分支 `codex/residual-fp-centered-cross-20261007`，从本方向既有结果提交 `cf4aa83fa4e56c56a8b50cfcb4eba172fd449d9b` 创建linked worktree。
- 可靠性：`D:/GraphCliff-Pair`，保留原分支 `codex/paired-increment-plan-20261007` 及创建时HEAD `dc9a78e104c222d9ab7ef3554767b1af312114e1`，不写入、不切换。
- 本方向相对main的既有三个提交仅涉及FP/LongPoly实现、测试、报告及本方向日志；未从可靠性分支创建新分支或合并其后续工作。
- 后续计划/日志写本文件；实验写本worktree的 `artifacts/`。本次只提交 `AGENTS.md` 和本文件，不提交产物或其他方向文件。

## 隔离步骤与验收

1. 核对两分支和工作区状态，创建独立worktree：已完成。
2. 本地复制12组正式FP结果、两个最终smoke矩阵、正式运行日志及固定官方 `pooling.py`；不复制可靠性产物，不建立可写链接：已完成。
3. 逐文件比对复制件SHA256，检查无reparse point/共享链接，核对源目录HEAD及状态不变：通过。204份产物文件、299,155,775字节与原件一致；固定官方pooling.py另行比对通过；artifacts/external内无reparse point。
4. 在新worktree运行现有契约检查、正式结果审计；核对来源字节与历史manifest兼容：通过。4项现有契约检查通过，正式12组与两个最终smoke各12组审计通过。新checkout默认LF，而历史manifest的4个继承Python文件使用CRLF；仅在Git规范化内容完全一致后恢复历史字节，审计hash一致，Git无源码差异，没有修改算法或提交这些文件。
5. 提交范围核验：仅 `AGENTS.md` 与本文件；推送本方向分支后核对远端SHA。

旧报告中的 `D:/GraphCliff-Pair/artifacts/...` 是历史运行位置，保留用于来源追溯；对应独立副本现在位于本worktree下相同相对路径。历史报告/manifest不因搬迁重写，旧产物原件不删除。字节核验清单保存在忽略目录 `artifacts/worktree_isolation_inventory.json`。

本次不训练、不改模型/配置、不启动LongPoly试验。正式FP主候选不扩训的结论保持不变。
