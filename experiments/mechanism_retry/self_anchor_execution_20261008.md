# D：self-anchor有限先导执行协议（训练前固定）

2026-10-08。对应[唯一假设与推导](self_anchor_proposal_20261008.md)。这是一项新的anchor假设先导，不回改centered/context/FP旧门槛，不恢复旧队列，不把self当新研究方法。

## 范围与预算

只用CHEMBL234_Ki官方train内部既定split_seed42开发拆分。选234因为原centered在此未胜self，且已有完整输入身份记录；不是选阳性任务，仍属反复使用的开发集，不是独立确认。官方test标签不进入计算。

full/self/gated_delta/anchor四臂，seed42，两个独立进程各运行整套四臂，共8fit，顺序执行。每fit固定最多20epoch、patience20、100epoch调度；验证Overall MSE选20epoch范围内最佳checkpoint，不能按Cliff挑epoch。共最多160个训练epoch。这个短预算不同于旧100epoch先导，只比较本轮同期对照，不跨表排名。

H256、3层、4heads、batch32、lr1e-4、min_lr1e-6、warmup10、AdamW betas(.9,.95)/decay1e-5、clip1、普通MSE，均沿用既有配置；只预先缩短训练上限，不调参。self/anchor/gated_delta共享attention初始化，四臂head初始化须一致；full保留原LongPoly。原max并列规则统一改为native amax，high精度和原Dropout不改。严格确定性开关在每次原set_seed后重新启用。

gated_delta与anchor参数和执行路径相同，仅是否加U不同；不会要求gated_delta的初始化输出等于self。门控有符号，每层一个a，零初始化，实际tanh值在checkpoint中记录。不给某一臂加载训练好的self权重；全部同期从共同底座初始化。

## 显式复现承诺

承诺来源/数据/划分/配对/共同参数初始化/优化预算一致，尽量使用strict算子；**不承诺完整GPU训练逐位相同**。M52失败、M55有限一致均保留。两个进程重复用于描述同seed执行变动，不能估计跨seed方差或做可靠显著性检验；本协议不是对旧实验的追认。

报告各臂两次结果及范围。对每个指标m，继续条件固定为

$$\max_{r\in\{1,2\}}m(\mathrm{anchor},r)<\min_{r\in\{1,2\}}m(c,r),\quad c\in\{\mathrm{full,self,gated\_delta}\}.$$

Overall/Cliff同时满足，共6项严格比较；不用平均值掩盖重复间重叠，不从某次重复挑最好值。通过也只记“此开发任务/短预算/两个重复下的探索阳性”，不是统计显著或最终有效；不通过则停止此候选，不追加seed/任务/FP/调参。扩展与论文方法声明须另立计划，不能自动启动。

## 开跑条件、核验与停止

1. CPU/GPU契约及H256/3层检查通过；真实234首个训练batch四臂各一次forward/backward通过，optimizer步骤0；数据/来源SHA、初始化、参数和无test策略写入self_anchor_preflight_20261008.json。
2. 训练来源/配置/本协议提交且工作区干净后，运行tools/run_self_anchor_pilot.py。suite记录来源和commit，两个manifest均须干净且一致；任何非有限值、来源变化、身份失败或异常，停止并报告技术失败，不改参数续跑。
3. 核验8个已存预测的指标、query/参考/truth身份、epoch预算、初始化和来源；独立重载8个checkpoint重放固定验证batch，与保存预测比较。
4. 本轮报告完整重复范围、6项门槛、训练与推理成本、学习到的门控，不选一个有利旧分数替换同期self。

不新增FP、Loss、蛋白/MCS输入或全模型重构。原模型、训练器、旧配置/报告只读；所有新增文件与产物在本worktree，本方向独立提交，不合并。
