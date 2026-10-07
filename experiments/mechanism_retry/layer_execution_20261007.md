# 独立LongPoly逐层替换：执行前方案

2026-10-07。用户在确认worktree责任与隔离边界后要求“按照计划继续你的研究”。承接先分别检验FP、再检验layer的计划，本轮只推进尚未执行的独立layer先导；已停止的FP单分子主候选不恢复。

## 固定候选和分阶段预算

主候选固定为 `centered`，即逐层共享MHA的cross输出减self输出，并经原分组scale/bias、RMSNorm/SiLU与T(0)修正。保留ShortGINE、门控、残差及SAG读出。公式及反例沿用本目录README，不增加模块、Loss、标签信息或参数搜索。

原 `layer_screen.json` 六臂配置保留不改。为避免组合补救和无信息成本，在看到正式结果前分阶段执行：

1. `layer_core_screen.json`：234/244 × full/self/cross/centered × seed42，8次完整fit上限；只是原配置的前四臂子集，全部模型与优化超参数不改。
2. 主候选在**两个任务、Overall和Cliff两指标，均严格低于同期full、self及cross**，才通过继续条件。不能在结果后改选self/cross或换指标、换任务。单seed通过也仅意味着值得补证，不证明显著性、普遍有效或原创。
3. 若不满足，停止本轮扩展，后四次FP消融不执行；不调参、不换Loss、不用FP来补救。这样可以区分原六臂方案中的机制验证与组合归因，并减少反证后的无用训练。
4. 若满足，再运行同任务/同初始化/同参数的full_fp和centered_fp两臂，最多4次fit。组合仅作已有2×2表的消融，不能替代主候选通过；不自动扩seed/任务。

核心预算：H256、3层、4heads，最多100epoch、patience15、batch32、lr1e-4、普通MSE；固定split_seed42及训练seed42。用validation Overall MSE选最佳checkpoint。不改官方train/valid开发划分，不解析官方test标签。参考只取训练集、按既有结构Top-1排除自身及canonical相同分子；参考活性在输出差值后才用于恢复预测。

## 公平性及已知限制

self/cross/centered均6,810,654参数，初始化规则一致；full为6,021,198参数，并不等容量。每层self/cross有2次attention，centered有4次；参数相同不等于计算相同。规范化、分组scale/bias及LongPoly默认dropout=0.1在三个替换臂保持一致。组合另增加FP参数与成本。

两个任务均为已反复使用的开发任务，单seed、同划分结果不能作为独立确认。相同输入下零交互、置换等变与最终反对称是代数约束，不保证保留有用的共有上下文；centered可能丢掉本应由LongPoly表达的上下文。attention权重不是化学因果解释。本轮不执行原子解释遮罩或构建新划分。

## 输入、验收与责任边界

只读输入 `D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv` 和 `CHEMBL244_Ki.csv`。输入SHA、完整训练/验证覆盖、Cliff数量与参数计数见 `layer_preflight_20261007.json`。运行前核对4项现有契约检查，并验证H256实际批次的forward/backward有限。

输出只写 `D:/GraphCliff-Pair-FPPool-LongPoly/artifacts/mechanism_retry_layer_core_screen_20261007`；目录存在拒绝覆盖。失败保留现场；不自动改batch/精度或修改协议重跑。全部核心fit完成后独立重算指标、对齐查询/参考行、核对来源与初始化、逐个checkpoint重放、检查交换反对称/同分子零差值，报告参数、实际fit耗时/显存/attention调用预算。保存完整阳性和阴性结果，再按上述规则决定是否运行FP消融。

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe -u -m experiments.mechanism_retry.run --config experiments/mechanism_retry/layer_core_screen.json --csv-root D:\GraphCliff-main\benchmark_data --output artifacts/mechanism_retry_layer_core_screen_20261007
```

全过程只使用本worktree和分支。计划/日志只更新 `experiments/mechanism_retry/worktree.md`，实现/报告仅提交本方向文件；不写可靠性对话目录或计划，不合并分支，不上传数据/权重/逐分子预测。先提交本执行前方案，再训练；结果完成后单独提交分析报告。
