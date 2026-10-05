# GitHub优先复用：三个官方候选核查（2026-10-05）

结论：优先复用已有项目，只写薄适配、数据协议和论文贡献必需的差异。现有GraphCliff、my_work基线和FPPool继续复用。本轮找到并核对三个官方项目，两个MIT项目的最小算法文件及许可证已按提交缓存，纯合成接口检查完成；没有启动性能实验，也没有建立新的论文贡献。

| 官方项目 | 可复用入口 | 当前结论 |
|---|---|---|
| [DIR / LDS / FDS](https://github.com/YyzHarry/imbalanced-regression) | imdb-wiki-dir/loss.py、utils.py、fds.py；weighted_mse_loss、get_lds_kernel_window、FDS | MIT；适合作为标签不平衡对照。Loss已做CPU/GPU等价和梯度核对；分箱与FDS需薄适配，尚未运行完整算法 |
| [BalancedMSE](https://github.com/jiawei-ren/BalancedMSE) | synthetic_benchmark/loss.py：BMCLossMD / bmc_loss_md | MIT；优先作为不同于普通权重MSE的对照。原GPU路径设备失配；仅补两处device参数后CPU/GPU与梯度通过，尚未接入训练 |
| [IA-MoE](https://github.com/syan1992/Interval-Aware-Mixture-of-Expert) | models/module.py：IAMoE；models/gating_loss.py；utils/util.py：calmean | 分子回归相关性最高；当前递归文件清单未见许可证，先作技术/创新对照，不复制源码到公开项目、不声称已可运行 |

## 固定版本和获取边界

- DIR：a6fdc45d45c04e6f5c40f43925bc66e580911084。
- BalancedMSE：c6f2f33eb05774819b71b93061de2bb7bf9bfaca。
- IA-MoE：07935f736403fc2a6560c994f3a80d99f2dc8593。
- 递归仓库文件列表、14个所查文件的URL及UTF8文本SHA256：github_reuse_audit_20261005.json；该哈希明确为解码后UTF8文本，不冒称下载响应原字节哈希。
- 原字节缓存仅6个MIT文件：DIR的LICENSE/loss/utils/fds，以及BalancedMSE的LICENSE/loss。位于D:/GraphCliff-Pair/external/reuse_candidates，Git忽略；精确原文件SHA256见reuse_candidates_downloads_20261005.json。
- 没有克隆完整历史、图片数据、预训练权重或IA-MoE源码；没有安装或升级依赖。IA-MoE源码只读核对，未执行。

## 不能直接照搬的细节

**DIR：**原数据集权重函数面向非负整数年龄桶；当前y=−log10(nM)包含负数，不能直接int(y)照搬分箱。桶边界/分箱宽度必须由训练信息与预先声明的配置确定，不从test估计。FDS原实现有显式.cuda()，以及按epoch更新统计的流程；不能只替换一个Loss就宣称完成FDS复现。训练特征统计、标签平滑与旧动态pair-delta权重是不同方法。

**BalancedMSE：**多维函数的torch.eye和torch.arange默认CPU，当前GPU执行实际报设备失配。本轮仅在内存中给这两处加device=pred.device，原Loss公式与缓存文件不变。其noise_sigma为可训练参数；若选用可学习sigma，须明确加入训练optimizer并记录学习率，不能只替换Loss后把它漏掉。合成检查init_noise_sigma=1.0仅用于接口检查，不是为分子实验选定的参数。BMC有批内交互，需核对batch_size、末批及成本；没有证明会提高当前分子指标。

**IA-MoE：**calmean会读遍所传dataset.y计算均值、标准差与四个分位区间，调用方必须仅传train。实际区间在标准化标签尺度上构造，README却用targets_denorm示例调用gate Loss，接入前必须统一尺度。module.py末尾的squeeze可能在单样本批次去掉batch维；需明确[B,D]契约。requirements.txt仅列torch/numpy，而util还导入RDKit、PyG、SciPy、sklearn，不能据该文件声称环境完整。所查仓库主要提供模块与使用片段，尚未找到完整论文任务训练/评估入口，不将其当作已完成可复现实验。

这些是静态代码核对或合成接口发现，不是对官方方法效果的否定。许可未确认时优先选用已确认许可的替代入口，不为此阻塞整个研究。

## 已实际验证与未执行项

- 检查6个下载原文件的SHA256；缓存源码均未修改。
- 官方DIR weighted_mse_loss在CPU/GPU上权重全1等价普通MSE，预测梯度有限。
- 官方BalancedMSE BMCLossMD在CPU前向、预测梯度及sigma梯度有限；原GPU失败原因记录完整。
- 两处设备薄适配后的BMC在CPU/GPU前向值一致，梯度有限。检查脚本只加载两个MIT Loss模块和一个原函数，不执行完整外部训练项目。
- 尚未运行LDS/FDS完整流程、IA-MoE、GraphCliff结合后的分子训练或效果评估。本轮不读取原test标签/旧test预测。

核对输出：reuse_candidates_interface_20261005.json。复算入口：在GraphCliff环境执行`python -B tools/check_reuse_candidates.py`，需本地上述固定缓存。缓存缺失时按下载清单逐文件获取，保留LICENSE并校验SHA256；不按分支最新版本替换。

## 围绕发刊的选择

这三个项目是可借用的算法/强对照，不是三项新贡献。直接GraphCliff＋LDS/BMC/IA-MoE不能仅因组合或换数据集声称原创，也不能据代码可运行保证二区录用。

当前优先级：先用已有基线验证活性区间/邻居覆盖控制后的cliff额外误差问题；若得到跨任务一致信息，再选择最低成本、与假设对应的已审计算法作小规模对照。LDS/加权MSE是低成本参考，BMC为另一种现成Loss对照，IA-MoE为相关结构先例。一次只选择一个有依据的候选，保留原GraphCliff对照和固定止损条件，不同时堆三种机制。

上述候选取舍是当前研究判断，不是穷尽性算法检索。必要时继续查原论文参数、完整实验入口和近期工作；不自行猜参数。最终以贡献和可信证据是否能支撑投稿决定下一步。

本轮新增核查清单、文件哈希、合成检查脚本和本地MIT缓存，更新README/task_plan/notes/milestones。原GraphCliff-main、my_work基线、当前模型/Loss/训练配置、旧数据与结果不改；旧训练队列继续暂停。
