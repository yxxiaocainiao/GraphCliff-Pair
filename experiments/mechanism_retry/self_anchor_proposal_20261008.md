# C：保留self响应的有界跨分子修正

2026-10-08。唯一待检验假设：跨分子增量可以在保留self长分支响应的前提下提供额外预测信息。旧centered/context把差分作为整个长分支响应；同期self较强，提示值得检查“是否丢弃了有用自身响应”。这不是已证明的失败归因。

## 算子与推导

对每层、每侧，沿用同一个共享MHA和原LongPoly变换T（group affine、RMSNorm、SiLU）：

$$U_q=T(A(X_q,X_q)),\quad V_q=T(A(X_q,X_r)),\quad \Delta_q=V_q-U_q,$$
$$B_q=U_q+\tanh(a_l)\Delta_q,\qquad a_l(0)=0.$$

参考侧对称计算，共享同层标量a_l。B进入原dropout、sigmoid门控和残差，ShortGINE/v/读出/反对称预测头不变。与context不同，不用self共同分母；与centered不同，保留U，不对差分自身归一化。每层仅加一个标量，H256/3层实际参数6,810,657，比self多3。

保留LongPoly的full实际6,021,198参数，候选比full多789,459；“+3”只相对既有同容量self注意力对照成立，不能表述为相对原GraphCliff仅加3参数。

有限值条件下a=0有B=U，代数上退化到同容量self注意力对照，**不是退化到原LongPoly/原GraphCliff**。若cross=self，则Delta=0，任何a都不改变self响应。

对固定U/V，任意向量范数有

$$\|B-U\|=|\tanh a|\,\|\Delta\|\le\min(1,|a|)\|\Delta\|,$$
$$\frac{\partial B}{\partial a}=(1-\tanh^2a)\Delta,\qquad \left.\frac{\partial B}{\partial a}\right|_{a=0}=\Delta.$$

所以零初始化不会像a²门控那样必然截断标量的一阶梯度，但Delta=0或损失梯度与Delta正交时仍可能没有学习信号。a为有符号系数，不能称概率或凸组合；负值会沿cross的反方向修正。

该界仅对变换后的固定响应和门控参数成立，不是对原始分子扰动的全局Lipschitz界，不保证化学局部性、优化收敛或预测风险下降。T本身仍含RMSNorm，不宣称其敏感性消失。有限值假设必需，0乘NaN不会恢复基线。

反例：若cross响应含与预测目标无关的大噪声，学习到非零a仍可恶化泛化；若self已足够，最佳修正可以为0；若U很小而Delta很大，门控有界仍不保证修正相对self很小。这些情况不能用公式排除。

## 最小对照与可证伪结果

四臂：保留LongPoly的full、同容量self、gated_delta（仅tanh(a)Delta）、主候选anchor（U+tanh(a)Delta）。后两者共用完全相同实现与参数，只切换是否保留U；用于检查anchor本身，而不是将相对full提升全归给跨分子信息。self/anchor/gated_delta共用attention初始化，head初始化须四臂一致。

候选与gated_delta每层各调用两次现有interaction（self/cross），约为self的attention调用数两倍；+3参数不是计算成本可忽略的证明。不得另加FP/Loss/匹配模块。

若候选在预登记范围内不能同时胜full/self/gated_delta的Overall与Cliff，就停止本候选；若只胜full、不胜self，不能声称跨分子增量有效；若不胜gated_delta，不能声称保留U有预测贡献。

## 检查与新颖性边界

`tests/test_retry_self_anchor.py`通过CPU H32/2层及GPU H32/2层、H256/3层：零门控eval/受控training RNG的self退化、非零门控梯度、eval交换反对称/同对零输出、同形状批次隔离、重载与参数数目。是实现/构造检查，不是训练收益。

最初测试未按正式strict设置时GPU端点差约2.98e-8；按现有high/strict/native amax运行后端点检查通过。缩小整个GPU批次时观察到约3.33e-4输出差，不能据此定位原因；GPU隔离检查因此采用同形状替换无关pair，CPU另保留缩批次检查。未改模型精度/Dropout，不把此观察扩成新的复现支线。

零初始化残差门控已有先例，例如[ReZero，UAI 2021](https://proceedings.mlr.press/v161/bachlechner21a.html)。本方案不宣称门控、残差或self/cross差分本身原创；完整相关工作查重未完成。当前贡献定位仅是一个可证伪的GraphCliff替换假设，论文方法贡献必须等待预测证据与进一步文献核对。
