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

## M49 — 老师反馈后的机制设计讨论

用户指出模块替换难以支撑论文方法部分，要求独立思考的小改动。本轮先提出、推导和检查一个候选：以self上下文的共同尺度归一化cross与self输出，再取非线性变换差值，避免直接以微小差值自身作为归一化尺度。不宣称原创确认或有效最终算法。此前建议的8次seed复核未启动，本轮不训练、不改现有模型或历史门槛。

1. 核对当前代码和初始化导数证据；查原始Differential Transformer及相近归一化工作，区分已知操作与待核查差异。
2. 写清参数不增加的算子、零差值性质、固定上下文下的局部Jacobian与条件上界；主动列出自上下文接近零、失去共同偏置抵消、端到端未保证等反例边界。
3. 用独立小脚本对构造输入检查代数/导数/幅度响应，不修改训练模型，不把构造输入检查当实际分子收益证据。
4. 交付本方向设计说明和可运行检查；后续先测真实checkpoint的尺度分布，动机成立才另立最小对照方案，不自动进入训练或组合。

M49完成：self_context_scale_proposal_20261008.md定义共同self尺度的非线性差分、固定self条件的Jacobian及范数界，并列出不能保证端到端稳定/预测改进及共同偏置不再一般抵消等限制。查阅Differential Transformer原文，确认相减与差分后RMSNorm已有先例；相近DINT提示归一化相关工作仍需系统查重，未确认原创。

tools/check_self_context_scale.py构造检查通过：旧初始零点导数1448.154663，新单位self上下文导数0.927670538，均匹配解析式；t从0.01到0.1时旧输出0.730506182→0.731052995，新0.009291768→0.094227552；非零learned affine参数下同输入严格零。报告两行数值与脚本输出逐项核对，优化步骤0，记录在忽略目录artifacts/check_self_context_scale_20261008.json。只暂存本设计、脚本与本日志三文件；实际分子尺度诊断、全模型实现与训练均未启动，不把构造检查当机制或效果证据。

## M50 — 用户授权继续真实checkpoint诊断

先检查234/244两个已训练centered最佳checkpoint，在全部验证查询及训练参考上观察各层MHA输出；用hooks记录而不改变模型forward，并重放验证预测确认观测没有改变输出。每层每侧统计self仿射后RMS、差分RMS、eps占尺度的比例，以及沿真实差分方向t=0.01/0.1/1的旧/候选变换幅度响应。t仅作算子离线扰动，不是新分子或新训练超参数；量化统计含重复参考节点，不当独立样本。

先固定观测范围与量，不根据标签挑分子、层或t，不新增优化或读test；保存checkpoint/来源hash、全部分层统计和中文诊断。若self尺度退化或幅度动机不对应真实状态，如实否定/限定；若支持，则再固定最小三臂方案，旧No-Go不变。本轮所有修改和输出仍限定独立worktree，不写可靠性计划。

诊断完成：2个checkpoint全部539条预测重放通过，12组103785次节点出现覆盖完整；eps占self分母最大约1.53e-6，无>=1%的节点。旧后两层10倍差分的幅度响应中位数仅1.14–2.09倍，候选约9.93–9.98倍；第一层旧响应更接近线性。支持继续测试设计动机，不证明泛化或原差距因果。

在用户“ok继续”的授权下，固定新的6fit先导：234/244×self/旧centered/context×seed42；相对两对照的两任务双指标共8项必须全部严格改善才建议后续补证，不自动扩展。方案self_context_scale_execution_20261008.md、配置context_scale_screen.json；H32全模型契约及H256真实大图batch32梯度预检通过，三臂6810654参数与完整初始化hash一致。函数从prototype移到实际候选模块，旧模型/训练器/配置不改。执行前先提交上述9个本方向文件，再训练；运行中冻结源码，结果另提交。

执行前提交aa1381fb7f3a2da6ab8cd4de4f56706f97adbb9c已推送。00:43:51在本worktree启动6fit顺序训练，PID50252，输出artifacts/mechanism_retry_context_scale_screen_20261008，同名前缀日志/process.json保存启动信息。只新增tools/report_context_scale.py准备结果审计，不修改训练捕获的源码/配置；候选结果尚未完成。

00:55核心3/6完成：234 self Overall/Cliff为0.724281067/0.753278543，旧centered为0.739218832/0.810960484，context为0.731176223/0.830053490。候选仅改善旧centered Overall，Cliff退化且双指标未胜self，已不满足新门槛；继续完成244预定对照，不追加其他变体。以上暂为训练器保存值，等待完整独立重放。

同期旧臂出现重跑差异。两轮历史源码hash、优化参数、torch/PyG/RDKit与完整初始化一致；234 self验证轨迹前16epoch完全一致，第17起分歧，旧centered第1epoch起有差异。当前未定位原因，不把它直接归因GPU；协议本来只承诺best-effort deterministic。结果工具将保存历史预测SHA、重跑偏差与双指标，评价只用本轮同期对照，限制小幅收益解释。

M50完成：6/6训练ALL_DONE，进程退出、stderr空；6个checkpoint独立重放1617条预测最大差8.89e-16，首batch交换/同输入契约各192对误差0，当前19份来源及历史8组18份来源、身份/覆盖/初始化/预算/指标/选权重均核验通过。共1109.97秒即18.50分钟fit计时，官方test0。

context在234为Overall0.731176223/Cliff0.830053490，在244为0.916479053/1.097131332；相对self双指标均退化，仅改善旧centered的Overall，两任务共8项仅2项通过，按预定规则No-Go，不扩训/不调eps/不组合FP。旧centered相对self的Cliff优势也未在同期重跑保持，M48模式只能作为原单次运行描述，不能当稳定机制。重跑来源与记录环境字段一致但原因仍未定位，后续机制收益评价需先处理复现范围，不把它作为泛化保证或直接归因GPU。

中文报告/机器核验为context_scale_results_20261008.md/.json，tools/report_context_scale.py复用原审计并核对历史对照。16行报告表格/8判断/链接/参数与JSON核对通过。结果只提交上述三文件、本目录README.md及本日志，共5文件；运行期间模型/训练源码与配置未改。可靠性目录仅只读核对，本任务未写入其计划；推送本分支，不合并。

## M51 — 同 seed 训练分叉定位（2026-10-08）

用户授权继续迭代。本轮先处理M50暴露的重复差异，冻结架构、Loss与正式筛查门槛；不启动新的正式fit或组合。开始时独立worktree干净，HEAD6f2a244；只写本方向工具、证据与本日志。

1. 复用实际数据读取、初始化、配对、批处理和优化器设置，对234 centered固定首epoch前8个batch做短程追踪，两个独立进程重复。
2. 记录样本顺序、输入、CPU/CUDA随机状态、预测、梯度及更新后权重hash；先检查严格deterministic模式是否拒绝某一步，再与现有warn_only模式对照。
3. 若发现首个差异或明确报错，做最小定向复核；未出现差异则如实限定短程覆盖，不能宣布解决历史分叉。
4. 保留可运行检查及聚合报告；仅提交本任务文件，不改可靠性计划、不自动合并。

首轮strict/warn各两个独立进程共32个优化步骤，前8batch的样本、输入、CPU/CUDA RNG、预测、全部梯度与更新权重逐位一致，strict未报错。为覆盖旧centered首epoch差异，将两次strict诊断延长至完整首epoch83batch；不验证选模、不扩模型、不运行正式100epoch筛查。短程一致不能解释历史全程差异。

M51完成：实际Loss两进程首epoch在第54batch首次出现反向梯度/更新权重差异；前53batch一致，第54输入/预测/Loss/RNG仍一致。模块输入/输出观测确认前向全部一致、head输入输出梯度一致而SAG节点输出梯度不同。初版离线重放漏设训练矩阵精度，错误未发现并列；已撤回该排除性判断，工具改为调用原seed设置并断言捕获预测SHA严格恢复。最终两侧各8个活跃并列max通道，固定输入/上游梯度各100次，原路径两侧各23种梯度SHA，原生amax各1种且前向相同。

新增本方向repro.py可选入口，原生amax在并列时平均分配次梯度，不加参数、不改前向max值，manifest注明协议；退出/异常恢复进程内路由，不支持并发。原型两次及实际helper两次首epoch83batch追踪逐位一致。6个针对性检查通过（原mechanism4、context1、新CPU/GPU并列/恢复契约1），报告工具核验6对追踪、最终5份来源SHA和18段诊断清单，共991个诊断优化步骤，完整性能筛查0、验证选模0、官方test0。首epoch通过不保证完整训练；不据此撤销旧No-Go或声称性能收益。

提交范围为README/worktree、repro.py、结果md/json、tests/test_retry_repro.py和3个tools，共9文件。原模型/训练器/Loss/配置及历史报告/manifest保持不变；不写可靠性目录或计划、不自动合并。下一阶段先验证更长轨迹和另一任务，再讨论有限同期对照，不自动恢复旧队列。

## M52 — 修正协议的长轨迹与第二任务验收（2026-10-08）

用户“继续下一步”授权承接M51。开始时独立worktree干净，HEAD23139e3。默认communication/planning skill仍缺失，继续以本文件持久记录，不写可靠性计划。

1. 冻结新验收范围：234/244 × self/centered × 两次独立进程，seed42、原生amax、strict deterministic、固定20epoch；覆盖历史第17epoch分叉。不是旧seed43/44效果扩训队列，不评估context/FP组合。
2. 复用实际模型、读取/批处理/Loss/优化器/调度器；逐batch将输入、样本顺序、RNG、预测、梯度、更新权重SHA写JSONL，逐epoch记录验证预测、优化器状态和RNG。验证只重放，不计算收益、不选权重、不早停。
3. 执行前提交范围与工具；启动独立进程，启动时固定PYTHONHASHSEED=42和CUBLAS设置，顺序使用本worktree输出，运行期间冻结来源。每对完成即比较，失败则停止后续队列并保留首差异。
4. 全部通过只说明当前机器/软件/两臂两任务20epoch范围重复；不保证100epoch、其他seed或性能收益。核对来源与历史证据、输出聚合报告，只提交本方向文件并推送独立分支，不合并。

执行前4文件提交并推送4c54fbd956a7907b68f447da02f0919257b0275e；比较工具的晚段差异/完整覆盖检查与原生amax梯度/恢复检查通过。08:09:52在本worktree隐藏启动PID12012，启动manifest为干净Git；输出artifacts/mechanism_retry_repro_long_20261008。运行期间仅准备未被捕获的report_layer_repro_long.py，24份来源及配置保持冻结。

M52完成：234/self两次各20epoch/1660batch完成，第6epoch全局第485batch首次出现前向预测差异（trace第496行）；此前全部记录、该批次输入/配对/CPU-CUDA-Python-NumPy RNG一致，forward后RNG也一致。Loss0.34009838104248047与0.3401349186897278，随后梯度和权重分叉。具体前向算子未定位，不能直接归因某GPU内核、dropout或centered。按预定失败即停规则仅2/8次，另6次（含全部244）未启动，不把计划写成完成。

父进程正常退出，completed.status=diverged，stderr为空。报告工具核验24份来源、两份完整轨迹SHA和20epoch覆盖，3400条记录、3320个优化步骤、11680次验证预测哈希；worker累计427.63秒（含读取/hash/验证），峰值各565.65MiB。验证只重放无指标/选权重，官方test0。M51并列max局部修正仍成立，但长程复现未完成，旧机制No-Go保留。

结果交付layer_repro_long_results_20261008.md/.json、tools/report_layer_repro_long.py及README/本日志共5文件；加执行前4文件，本轮累计9文件，均为本任务。原模型/训练器/Loss/历史配置/报告未改；不写可靠性计划，不合并。下一步捕获并定位第485batch的前向算子，暂不恢复其余6次或收益扩展。

## M53 — 第485batch前向算子定位（2026-10-08）

用户授权继续迭代。独立worktree开始干净，HEADd98d17e；expression-skill仍未安装，持久计划继续写本文件。

1. 保留M52工具/配置/模型来源不变，薄适配其worker为两次234/self、6epoch诊断；调度周期仍100，其他设置不变。在第485batch捕获CPU模型/图输入/RNG，观测各nn.Module前向输入输出与RNG。
2. 比较前484batch与历史前缀，检查捕获是否准确衔接旧状态；比较两个观测到的第485batch，定位首个输出不同且输入相同的模块。承认同步/观测可能改变执行路径，不把未复现当解决。
3. 复用真实捕获做独立前向重放，必要时定向检查已定位算子；仅在确认原因后做最小本方向修正及对应检查。未定位则明确保留未知，不凭猜测改精度/Dropout/架构。
4. 只提交本方向工具、证据及日志；不恢复M52剩余6次、不新增收益对照、不写可靠性计划、不合并。

执行前50d756d已提交推送。两次同步观测各6epoch/498步均完成，前484batch与M52历史前缀相同，捕获权重和第485预测准确对应历史repeat2；143个模块观测一致。三次独立同步重放、两次延迟观测重放及50次无模块hook前向均得到同一repeat2输出，尚不能定位历史repeat1分支，未宣称解决。为减少逐模块同步干扰，追加两次同范围6epoch延迟观测（只保留tensor引用，整个forward结束后再hash），仍不改模型/精度/Dropout，不做收益对照；使用新薄适配工具，不修改已固定历史工具。

延迟观测前工具b4ad762提交推送，两次各6epoch完成，仍同一输出。注意延迟工具仍在target forward前将快照复制到CPU；最后追加一组两次同范围诊断，将快照CPU复制也推迟至forward后，保留前状态GPU引用并在优化器更新前保存。若仍未暴露差异，本轮停止扩展并明确未定位；不改模型、精度或Dropout，不恢复效果队列。
