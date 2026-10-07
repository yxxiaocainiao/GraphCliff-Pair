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

## M47 — 2026-10-07 用户授权继续独立layer研究

上述“不训练”属于隔离操作。用户随后明确要求按计划继续，本轮推进独立LongPoly逐层替换，FP单分子不扩训。

1. 核对两worktree未提交改动归属及GPU：两处均干净，无其他Python训练，可靠性HEAD仍dc9a78e；只在本目录写入。
2. 固定核心主候选centered及分阶段继续条件，原六臂配置不改，先导先取前四臂：234/244×四臂×seed42，共8fit；只有两个任务双指标均胜full/self/cross才执行另4次FP消融。执行前方案layer_execution_20261007.md。
3. 保存输入/覆盖/参数预检，核对现有契约与H256真实批次梯度；先提交执行方案，再启动新目录顺序训练。
4. 核心完成后独立重算指标、checkpoint重放及反对称/身份/来源检查，按规则继续或停止，交付中文报告和机器核验；不在中途换候选或救结果。
5. 只提交本方向文件，核对逐文件diff，推送本分支；不自动合并，不修改对方计划。

当前：4项契约检查及四臂H256/3层实际大图batch32前后向通过；attention三臂初始化hash完全一致，centered峰值1591.10 MiB。输入234为2632train/292valid/128valid-Cliff，244为2229train/247valid/118valid-Cliff。无优化步骤、无正式结果；预检完成，先提交方案后启动8组，组合未启动。
