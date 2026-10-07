# 方法构思：用自身上下文提供差分交互的共同尺度

2026-10-08，针对用户及老师提出的“模块替换难以支撑方法部分”。本文件是一项待验证的小改动设计，不是已经有效的算法，也不是原创性确认。原FP/centered先导及No-Go结论不改；此前提出的8次seed复核未启动。本轮只写推导及构造输入检查，不修改训练模型、不训练。

## 1. 先确定要解决的具体问题

现有centered先做$D=A(X_q,X_r)-A(X_q,X_q)$，再做$B_{old}=T(D)-T(0)$。其初始化group_scale与RMSNorm权重为1、bias为0，得到

$$B_{old}(D)=\operatorname{SiLU}\!\left(\frac{D}{\sqrt{\operatorname{mean}(D^2)+\varepsilon}}\right).$$

对固定非零方向$h$和正数$t$，当$t^2\operatorname{mean}(h^2)\gg\varepsilon$时，

$$B_{old}(th)\approx\operatorname{SiLU}\!\left(\frac{h}{\sqrt{\operatorname{mean}(h^2)}}\right).$$

也就是沿同一方向改变差分幅度$t$，会被它自己的归一化分母大幅抵消；而在零点附近，Jacobian又为$I/(2\sqrt\varepsilon)$。这给出可检查的算子设计问题：差分用于表达相对自身上下文的变化，但尺度却由差分自身决定。

这不是234差距的已证实原因。真实attention输出未必落在上述条件，学习后的bias/scale会改变行为；也不能把隐藏表示幅度等同于结构改动大小或活性差值。

## 2. 只改变尺度来源与取差位置

对查询侧某个节点，令

$$u=A(X_q,X_q),\qquad v=A(X_q,X_r),\qquad \delta=v-u.$$

两项具有相同查询节点数；不要求跨分子原子一一对应。令$S$为原LongPoly的分组scale展开成的对角矩阵，$b$为其分组bias展开后的向量，$\Gamma$为原RMSNorm逐通道权重。使用self上下文提供共同分母：

$$z_u=Su+b,\quad z_v=Sv+b,\qquad s(u)=\sqrt{\|z_u\|_2^2/d+\varepsilon},$$

$$B_{new}(u,v)=\operatorname{SiLU}\!\left(\frac{\Gamma z_v}{s(u)}\right)
-\operatorname{SiLU}\!\left(\frac{\Gamma z_u}{s(u)}\right).$$

参考侧完全交换$q,r$。沿用ShortGINE、门控、dropout、残差、读出与反对称预测；不增加可训练参数、Loss、蛋白输入或FP。原scale/bias/RMSNorm权重均被使用，但RMSNorm的计算方式确实发生改变，不能称其行为完全保留。共同分母不detach，训练时self上下文仍接收梯度。

每层双方向仍为4次MHA，与原centered相同。变换部分变为两次仿射、一个self RMS尺度与两次SiLU，不声称与旧算子实际耗时完全一致。

设计意图：变化由另一个分子引入，尺度由查询自身上下文提供；不把变化本身再次缩放到近似单位尺度。

## 3. 能证明什么

**相同上下文零注入。** 若$v=u$，两个变换项完全相同，因此$B_{new}(u,u)=0$，即使学习到非零bias也成立。

**固定自身上下文时的局部响应。** 令$w=\Gamma z_u/s(u)$，固定$u$和模型参数，则

$$B_{new}(u,u+\delta)=\phi\!\left(w+\frac{\Gamma S\delta}{s(u)}\right)-\phi(w),\quad\phi=\operatorname{SiLU}.$$

在$\delta=0$处，

$$J_\delta B_{new}(u,u)=\operatorname{diag}(\phi'(w))\frac{\Gamma S}{s(u)}.$$

SiLU满足$|\phi'(x)|\le2$这一保守上界。因此对任意$\delta$，固定$u$时有

$$\|B_{new}(u,u+\delta)\|_2\le
\frac{2\|\Gamma\|_\infty\|S\|_{op}}{s(u)}\|\delta\|_2.$$

这里$\|\Gamma\|_\infty$指其对角元素最大绝对值。若self上下文的仿射后RMS至少为$\rho>0$，分母下界是$\sqrt{\rho^2+\varepsilon}$。与旧算子初始化零点主要由$\sqrt\varepsilon$决定不同，这个条件上界由自身上下文尺度决定。该上界是设计动机，不是新的泛化定理；不保证数值上一定小于旧算子在真实输入处的Jacobian。

**不会只因对差分本身做RMS而消除幅度。** 固定$u$，小$t$下$B_{new}(u,u+th)=tJ_\delta B_{new}h+O(t^2)$。若该方向导数为零，仍可能看不到变化，不能宣称保留所有有用信号。

各侧使用相同规则，因此确定性eval下仍可与现有反对称读出兼容。此次只检查变换函数，没有完成新全模型的交换、置换、padding或训练契约验收，不能将这些预期直接标为实现通过。

## 4. 主动保留的失败条件

- self上下文本身接近零时，$s(u)$仍可能由eps支配，不能宣称完全解决小分母问题。
- 上述界固定了$u$。真实输入变化会同时改变self/cross及其尺度，端到端梯度、优化稳定性和活性预测误差均未被保证。
- 共同value偏置会改变self分母和SiLU工作点；旧cross-self线性差中的共同常数抵消性质，不再一般成立。
- 大小差异、SAG筛选、dropout与参考选择问题没有因此消失；数值设计不等于识别共有/非共有化学结构。
- 活性悬崖允许相似结构出现大活性变化，本设计没有假设“结构相似必然活性相近”。隐藏算子的局部响应界也不等于活性函数的化学平滑性结论。

## 5. 已做的最小检查与尚未做的实验

`tools/check_self_context_scale.py`只复用当前变换参数，在构造的attention输出上检查。初始$u=\mathbf1$、float32时：旧零点导数1448.154663，新算子固定单位上下文时导数0.927670538，与解析式相符。

|构造变化幅度t|旧算子首坐标|候选首坐标|
|---:|---:|---:|
|0.01|0.730506182|0.009291768|
|0.10|0.731052995|0.094227552|

旧响应近乎相同，新响应随幅度明显变化；这是构造反例的性质检查，不是分子有效性证据。将group_bias设为0.4、group_scale设为2、RMSNorm权重设为1.3后，相同输入仍严格输出0。优化步骤0，没有新训练模型或分子预测结果。

```powershell
Set-Location D:\GraphCliff-Pair-FPPool-LongPoly
& D:\Tools\conda-envs\graphcliff\python.exe -m tools.check_self_context_scale
```

建议下一步先用已有checkpoint记录各层self上下文、差分及仿射尺度分布，并检查真实状态中的幅度响应；若self尺度也经常退化或假设无实际对应，先否定此动机。诊断支持后，再另立self/旧centered/候选三臂的最小同初始化实验方案，旧结果保持原标签；这不是直接扩训旧候选。上述诊断与训练尚未执行，不自动排队。

## 6. 与已有方法的边界及论文组织

[Differential Transformer原文](https://arxiv.org/html/2410.05258v2)式1–3使用两张softmax注意力映射的相减，并对差分head输出做RMSNorm和固定缩放。差分注意力与归一化不是新的通用概念。本候选使用同一MHA的cross/self输出，并用self上下文作为两路变换的共同尺度，区别是具体算子与参照选择；这点区别不足以宣布首创。

本轮检索还发现[DINT Transformer](https://arxiv.org/html/2501.17486v1)讨论差分注意力相关数值问题，提示相近工作需要进一步系统比对。本次没有完成穷尽检索，也未确认此算子是否已在别的领域使用；不能写“首次提出”。

如果后续证据支持，方法部分可按“差分自身归一化的幅度退化 → self上下文共同尺度 → 算子定义 → 零注入与固定上下文响应界 → 最小归因对照”组织。这比列举替换模块更有明确的问题与解决路径，但仍不保证达到目标期刊的贡献要求。

本轮只改本方向worktree.md，新增本设计说明及检查脚本；不改模型、训练器、配置、原报告或可靠性目录，不合并分支。
