# 两条机制重试：值得有限验证，不原样重跑

日期：2026-10-07。用户确认两个候选分别重试，组合仅作消融。

本分支交付可运行候选、与代码一致的推导及独立核验。2026-10-07用户“试试”授权后，[FPPool正式先导](fp_results_20261007.md)12/12完成，主候选不扩训；随后授权继续的[LongPoly核心先导](layer_results_20261007.md)8/8完成，12项比较通过11项，主候选亦未达到预定扩训门槛。**可证明的代数性质不等于泛化保证，也不等于原创性确认。**

M47更新：在专用worktree按[分阶段执行前方案](layer_execution_20261007.md)及[预检](layer_preflight_20261007.json)完成full/self/cross/centered共8次正式fit。centered在234/244均优于full的双指标，但234 Overall未胜self；因此不扩训，另4次FP组合消融未执行。没有恢复旧队列、调参补救或改选主候选；[完整结果和零点导数理论补充](layer_results_20261007.md)保留局部阳性及限制。

## 1. 历史证据决定这次只改哪里

依据[暂停复核](../../docs/research/module_pause_review_20261006/report.md)：残差FP在287的三seed Cliff均值改善2.24%，2047则恶化2.99%；真实membership相对扰动在两任务方向相反。逐层替换只做了两个任务各一个seed，同时增加约79万参数，并改变规范化/dropout。因此不能说它们普遍无效，也不能凭部分阳性支持原样重跑。

此次选择历史正/负任务作为探索样本，不叫独立确认。两条路线任务不同、信息条件不同，**不能混成一张排行榜**：

|路线|任务|推理信息|本轮问题|
|---|---|---|---|
|残差FP|287、2047|一个分子结构，不需要参考活性|指纹特征能否拟合底座尚未学会的误差？|
|逐层替换|234、244|查询结构、训练参考结构；输出后才加参考活性|跨分子输出中的自身背景能否被显式扣除？|
|组合消融|234、244|同一Pair信息条件|在逐层替换之外加FP是否有独立贡献？|

原GraphCliff源码、旧实验、Loss和历史结论不改。无新增Loss、MCS匹配、蛋白输入或任务选择搜索。

## 2. 候选A：从隐藏表示相加，改成预测残差补偿

设编码器得到节点表示 $H=E_\theta(G)$，原读出和FP读出分别为

$$g=S_\theta(H,G)\in\mathbb R^{2d},\quad p=F_\phi(H,M_G)\in\mathbb R^{2d}.$$

$M_G$是Morgan radius=2、1024位的原子membership。复用固定官方FPPool及现有压缩适配器，未重写其注意力。

旧路线为 $f_{feat}(G)=h_\theta(g+\alpha p)$。它隐含两种读出可在同一坐标中相加的要求，而且经非线性回归头后，$\alpha p$不再是可独立量化的预测修正。

新候选为

$$f_{out}(G)=f_0(G)+c_\phi(G),\qquad f_0=h_\theta(g),\quad c_\phi=w^\top p.$$

只增加一个无偏置线性标量头；$w=0$初始化。它无需假定FP向量与SAG向量有相同语义坐标。FP投影和该线性头可合并为仿射映射，**不将两层线性当成额外深度创新**。零初始化第一步只有$w$接收非零梯度，之后FP分支才开始学习；这是可核验的优化行为，不宣称更优。

先训练同任务、同seed底座，再冻结其encoder、SAG和head，并固定eval模式。冻结候选只拟合FP分支。另保留联合训练对照，分清“输出相加”和“冻结底座”的影响。

### 2.1 命题：冻结条件下的损失严格等价于残差回归

令$e=y-f_0(G)$。则逐样本有

$$[y-f_{out}(G)]^2=[e-c_\phi(G)]^2.$$

这解释了分支的明确任务：拟合底座剩余误差。它不是一个新的Loss，也不是因果分解。若联合更新底座，目标残差会随训练移动；该恒等式仍成立，但“固定残差任务”的解释不再成立。

### 2.2 命题：收益取决于误差相关性，不取决于模块名字

对任意固定数据分布或样本均值，记$R_0=\mathbb E[e^2]$，有

$$R_{out}-R_0=\mathbb E[c^2]-2\mathbb E[ec].$$

所以改进的充要条件是 $2\mathbb E[ec]>\mathbb E[c^2]$。若$c$仅添加与底座误差无关的波动，$\mathbb E[ec]=0$且$\mathbb E[c^2]>0$，误差必然增加。这给出了可直接用保存预测核验的失败条件。

若只在一个固定修正方向$c$上选择缩放$t$，且$\mathbb E[c^2]>0$，则

$$t^*=\frac{\mathbb E[ec]}{\mathbb E[c^2]},\qquad R(t^*)=R_0-\frac{\mathbb E[ec]^2}{\mathbb E[c^2]}.$$

这是二次函数配方的结论，**本实现没有用validation估计$t^*$，也没有添加这项调参步骤**。它说明理论改进要求方向与误差相关；不能把训练集上的相关性直接搬到测试分布。若$c=0$则风险相等，不除以零。

由于$w=0$可行，候选类在同一训练样本上的全局最优经验损失不高于冻结底座；实际非凸优化、早停和泛化均不受这个命题保证。训练器选已完成epoch中最低validation Overall MSE，没有自动回退初始底座；应如实报告候选退化。

### 2.3 六臂对照

`fp_screen.json`：base；feature_joint；output_joint；feature_frozen；output_frozen；output_frozen_null。feature组$\alpha=0$，output组$w=0$，保证eval初始化与各自底座一致。**这不是旧$\alpha=0.1$实验的精确复现**；当前统一使用PyG max pooling，没有复制旧first-node并列梯度规则，不跨阶段直接归因。

frozen两臂从本轮base最佳checkpoint开始，额外训练预算及底座成本必须累加，不能只报可训练参数。null在每个分子内反转membership行，保留各列计数/激活位，并与真实归属组同参数、同初始化。反转可能因对称性没有实际扰动：正式分析须报告改变矩阵的分子比例；这是单一弱null，不是全面随机置乱结论。不要只置乱FP列：FPPool位聚合的对称性可能使列置换无效。

## 3. 候选B：用跨分子相对自身的注意力增量替换LongPoly

保留原层的ShortGINE、三路拆分、sigmoid门控、$v$通道、残差和SAG读出。对第$l$层先按原实现得到

$$[X_2,X_1,V]_q=\operatorname{ShortGINE}_l(W_l\operatorname{LN}(U_q)), $$

参考侧同理。令共享多头注意力为

$$A(X,Y)=\operatorname{MHA}(Q=X,K=Y,V=Y).$$

单个head使用标准$\operatorname{softmax}(XW_Q(YW_K)^\top/\sqrt{d_h})YW_V$，随后拼接和输出投影。无位置编码，按配对batch隔离，有padding mask，attention dropout=0。

定义三个同容量控制算子：

$$D_{self,q}=A(X_{2q},X_{2q}),\quad D_{cross,q}=A(X_{2q},X_{2r}),$$
$$D_{center,q}=A(X_{2q},X_{2r})-A(X_{2q},X_{2q}).$$

参考侧交换$q,r$。注意两个分子的原子数不同，直接相减的是同查询长度的**输出**，不是大小不同的注意力矩阵；不需要虚构原子一一对应。

设$T_l$依次执行原LongPoly的分组scale/bias、RMSNorm和SiLU。替换层使用

$$B_q=\begin{cases}T_l(D_q),&self,cross,\\T_l(D_q)-T_l(0),&centered,\end{cases}$$
$$U_q^{l+1}=U_q^l+\operatorname{Dropout}_{0.1}(B_q)\odot\sigma(X_{1q})+V_q.$$

扣除$T_l(0)$是为了避免训练后的group_bias破坏零增量条件。移除的是Chebyshev系数及图多项式传播，保留其变换和dropout；不是调用已经删除系数的LongPoly.forward。self/cross/centered增加完全相同的MHA参数，初始化相同；full参数更少，不能声称与full等容量。

### 3.1 命题：相同输入的跨分子增量严格为零

若两侧节点表示相同（允许节点重排），共享投影、掩码正确且attention无dropout，则跨注意力与自身注意力输出相同，故$D_{center}=0$。继而$T(0)-T(0)=0$，该层此项注入为零，即使group_bias不为零也成立。

这只表明“没有分子差异时不制造这一类交互增量”。**它同时删除了同分子对原有的LongPoly贡献**，并不恢复原GraphCliff。在相似分子对中，可能连有用的共有上下文也扣掉，这是明确反例方向；因此self与full对照不可省略。

### 3.2 命题：值侧共同常数偏置抵消，但任意背景变化不保证抵消

固定Q/K和注意力权重，每行softmax权重和为1。若两侧所有value向量都加同一$c$，则cross和self输出各增加$c$（输出投影后增加相同$W_Oc$），相减后消失；共享输出偏置也相消。

这个性质针对**固定权重的value侧**。若改变原始节点表示，则Q/K也改变，注意力权重可能重新分配，不能宣称一般“公共化学背景被消除”、更不能宣称只剩非共有官能团。此处也没有证明减小过平滑或获得更正确的化学机制。

### 3.3 置换性质与反对称输出

没有位置编码时，对查询节点置换$P$、参考节点置换$R$，有$A(PX,RY)=PA(X,Y)$，由softmax行/列重排与$R^\top R=I$可得。ShortGINE和共享逐节点变换继承等变性，图读出在无SAG top-k边界并列歧义时不随节点排序改变。SAG并列筛选可能破坏严格全模型置换不变性，不把算子命题扩大到该退化情形。

最终仍沿用已有Pair规则，$d=S(U_q)-S(U_r)$，

$$\widehat\Delta(q,r)=\tfrac12[h(d)-h(-d)].$$

在eval确定性模式下交换两侧即翻转符号；同分子对输出为零。该规则来自现有项目，不是此次新贡献；训练时独立dropout采样不保证两次forward逐位反对称。反对称也不保证传递性或参考无关性。

### 3.4 六臂与组合解释

`layer_screen.json`：full、self、cross、centered、full_fp、centered_fp。前三个attention臂self/cross/centered容量相同，但每层self/cross需2次attention，centered需4次；不能说计算公平性已由参数匹配解决。

组合只使用预测空间FP修正，

$$\widehat\Delta_{fp}=\widehat\Delta+w^\top[F(U_q,M_q)-F(U_r,M_r)],\quad w_0=0.$$

它是**Pair条件下的新消融**，并非单分子冻结FP候选的组合证明。full/full_fp/centered/centered_fp形成该协议内2×2表，可对分子均方误差计算交互对比

$$I=(L_{centered+fp}-L_{centered})-(L_{full+fp}-L_{full}).$$

$I<0$只是当前数据上的额外误差降低，不等于统计显著协同。组合领先而单项不领先时，不能声称两单项分别有效。

## 4. 原创性与论文写法边界

以下为当前核查过的原始来源，不是穷尽文献检索：

- [GraphCliff官方代码](https://github.com/dmis-lab/GraphCliff)：沿用固定vendor，不把底座当新算法。
- [FPPool官方代码](https://github.com/shenwxlab/FPPool)：复用固定commit `ef2afe82e0490dc31a7cbb6146acea810de393fe`；README称MIT，独立LICENSE仍未确认，只外部获取，不随本分支复制其源码。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：标准注意力公式，不当此次创新。
- [Friedman, Greedy function approximation](https://doi.org/10.1214/aos/1013203451)：残差拟合/函数增量有成熟先例。本次是单个冻结GNN的FP标量纠错，不声称提出boosting。
- [Differential Transformer](https://arxiv.org/abs/2410.05258)：已有相减attention机制。本次用同一MHA的cross输出减self输出，不是它的双softmax映射构造，但这点差异本身不足以证明原创。
- [PrismNet原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12955929/)及[ACES-GNN原文](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00012b)：分子分解/频谱和共有/非共有结构解释已有先例；本次不声称首次利用差异或扣除共有背景。原文访问本轮分别遇到验证码/403，具体机制比较继承本地M18讨论，未假装重新读完全文。

可辩护的当前表述：“提出两个可证伪的实现候选，分别研究冻结底座的指纹残差纠错，以及与自身上下文比较的跨分子层间增量。”不能写“理论证明提升活性悬崖预测”“首创差分注意力”或“已经得到最终有效模型”。若发表目标要求明确新方法，仍需进一步核查最近相似算子；不为公式长而增加模块。

## 5. 正式筛查怎样决定是否继续

screen配置沿用原100epoch上限、patience15、lr1e-4、AdamW、batch32、H256、3层、split_seed42和seed42，不称文献最优参数。M45只运行H32/1层/2epoch/8训练查询/4验证查询的smoke；M46按独立执行前方案启动FP正式先导，不能拿M46方案当旧实验的预注册。

优先单独FP，再单独layer；组合只用于归因。原上限为先导两任务×六臂各12次，共24次完整训练；M47执行前进一步将layer拆为8次核心与通过后才运行的4次FP消融，实际累计正式fit为FP12+layer8=20次。冻结臂还需计入同轮底座训练成本；不引用旧checkpoint冒充同轮对照。

固定决策建议：先导中候选若不能在两个任务同时降低Overall与Cliff误差，则不扩大正式矩阵；FP需相对frozen feature和membership null检查增量，layer需相对full/self/cross检查特异性。此规则偏严格，目的是限制本轮投入，不是期刊涨点标准，单seed没有显著性或普遍无效结论。只有先导通过，才评估seed43/44重复及预先指定的未用于开发任务；这些额外运行未自动授权成后台队列。

先记录预测的逐分子误差、$2\overline{ec}-\overline{c^2}$、null实际扰动比例、总/可训练参数、拟合总耗时/峰值显存和attention调用量。FP增量若只来自联合改变底座，不可宣称已验证“固定残差补偿”；centered若不胜self，不可宣称跨分子信息有效。

保留训练参考Top-1和原拆分，不解析官方test标签。所有任务已用于开发，不能称未见确认集。未来解释仍需固定成功/失败/一般案例、真实输入区域对随机区域遮罩及跨seed稳定性；本次没有做这项解释验证，也没有把attention权重当因果解释。

## 6. 可复用入口与交付范围

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe -m unittest discover -s tests -p test_mechanism_retry.py -v
& D:\Tools\conda-envs\graphcliff\python.exe -m experiments.mechanism_retry.run --config experiments/mechanism_retry/fp_smoke.json --csv-root D:\GraphCliff-main\benchmark_data --output artifacts/new_fp_smoke
& D:\Tools\conda-envs\graphcliff\python.exe -m experiments.mechanism_retry.run --config experiments/mechanism_retry/layer_smoke.json --csv-root D:\GraphCliff-main\benchmark_data --output artifacts/new_layer_smoke
```

输入：`D:/GraphCliff-main/benchmark_data`（只读）、现有vendor/FPPool适配及历史复核报告。输出：本目录实现/配置/推导、`tests/test_mechanism_retry.py`、被忽略的`artifacts/mechanism_retry_*`运行产物。训练器、原模型、Loss、原项目和旧结果均未修改。新的SingleBaseline只在新进程内适配空边图，避免上游LongPoly空边特殊分支随混合batch改变；因此不把这次baseline结果当成旧实现的逐位复现。

runner复用既有训练器和其最佳权重重载检查，适配器仅顺序运行，进程内临时路由在finally恢复，不支持同进程并发启动。新输出目录存在则拒绝覆盖。数据、权重和逐分子预测仍由.gitignore排除。M45冒烟验证见[verification.json](verification.json)；M46正式运行与独立核验见[fp_results_20261007.json](fp_results_20261007.json)，两者不能混算成正式效果矩阵。
