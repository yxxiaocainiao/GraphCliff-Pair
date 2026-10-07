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

启动前状态：4项契约检查及四臂H256/3层实际大图batch32前后向通过；attention三臂初始化hash完全一致，centered峰值1591.10 MiB。输入234为2632train/292valid/128valid-Cliff，244为2229train/247valid/118valid-Cliff。预检无优化步骤、无正式结果；先提交方案后启动8组，组合未启动。

执行记录：方案提交并推送 `151cda243d792b3ca596a8cda0e9dcd6e600ba61` 后，23:31:53在本worktree启动核心顺序训练，PID30604。输出 `artifacts/mechanism_retry_layer_core_screen_20261007`，同名前缀stdout/stderr日志及process.json保存启动信息。保持模型/配置/训练来源不变；新增结果核验工具 `tools/report_layer_retry.py`，待全部完成后运行。FP消融仍未启动。

23:44核心4/8完成，234四臂全部完成。centered Overall0.741988735/Cliff0.781514929，full0.745143705/0.822568929，self0.733441879/0.783216506。centered未胜self Overall，已不满足预定继续条件；继续完成244核心四臂以保留完整反证，不追加FP组合或改参数。以上为训练器暂存值，独立checkpoint重放及完整审计仍待全部完成。

23:52核心8/8完成，ALL_DONE、completed.json与进程退出确认，stderr为空。`tools/report_layer_retry.py`完整审计和8个checkpoint独立重放退出0：2156条验证预测最大差8.89e-16，首batch交换/同分子契约各256对误差0，18份源码及两份继承依赖核验、初始化一致性、身份/完整覆盖/最低Overall选权重均通过。244 centered为0.927017222/1.118173686，双指标均胜full/self/cross；两任务合计11/12比较通过，唯一失败为234 Overall相对self。按预定门槛No-Go，不执行4次FP消融、不扩seed/任务、不改选self。

累计fit1167.33秒、19.46分钟，不含准备/重放，官方test评估0。运行中补充了初始化零点导数推导及自动微分核验：RMSNorm有效eps使局部导数约1448，零增量仍为0；只说明零值性质不等于稳定性，不当实际失败归因、不据此调参。中文完整报告及JSON为layer_results_20261007.md/.json。

结果提交范围限定本目录README.md、worktree.md、layer_results_20261007.md/.json和tools/report_layer_retry.py。模型/训练源码/原六臂配置/既有测试均未改；不写可靠性目录或计划，不自动合并。提交前逐路径与diff核对，推送本分支并验证远端SHA。

23:54提交前只读复核：可靠性worktree已自行推进到cb41459d6fce7dd211d4c457583da029df72fae4，其docs/milestones.md、docs/research/reliability_publication_plan.md、notes.md、task_plan.md为另一方向未提交改动；本任务未写入、未暂存这些文件，不把对方HEAD推进当隔离失败，不恢复到旧HEAD。

## M48 — 2026-10-08 继续现有预测诊断

用户要求“继续”。默认承接M47的No-Go，继续已有预测的收益/代价分析；已询问是否意指重新讨论扩训，等待澄清。不把简短继续自动解释为撤销停止条件。开始时本worktree干净，HEAD1939dbc；可靠性worktree只读状态检查干净，不写其目录。

1. 核对已有结果与数据来源；只读取本worktree的8组验证预测，不训练、不读官方test。
2. 复用已有审计，对centered相对full/self/cross按既有Overall/Cliff/Noncliff三组计算逐查询平方误差差、改善比例与残差恒等式；核验按组人数加权恢复Overall。
3. 保存明确标为事后描述诊断的中文说明和聚合JSON，不发布逐分子记录，不改主候选、门槛或原结果。判断哪些陈述可写、哪些仍缺证据。
4. 只提交本方向诊断文件和此日志，逐路径核对后推送独立分支，不合并。

expression-skill、planning-with-files、results-analysis/results-report在本地两个skills根目录均未找到；本轮继续使用本文件作持久计划，不因此增加审批步骤。

M48完成：复用审计确认8组和18份源码，保存8份预测hash；18项子集风险恒等式、6项人数加权恢复及另行直接平方误差重算通过，报告6行数值/计数/链接核对通过。centered对self两任务均Cliff改善、Noncliff退化：234加权贡献+0.001167128与−0.013777421，244为+0.025673509与−0.016681573。该共同模式限于self对照，244对full两组均改善，不扩大成普遍机制。

交付layer_tradeoffs_20261008.md/.json与tools/report_layer_tradeoffs.py，加本日志共4文件。事后描述，不作显著性/化学因果声明；新增fit0，官方test0，原No-Go/候选/门槛/配置及结果均不改。用户未进一步澄清时按上述默认路径完成诊断，不自行启动新训练。提交前核对4文件归属，只推送本方向分支，不合并。
