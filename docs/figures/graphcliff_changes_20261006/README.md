# GraphCliff改动位置图：老师汇报版

蓝色沿用，橙色新增/替换，灰色删除线被替换，灰虚线未定方案。图示是历史实现，不代表已经获得有效算法创新。

1. 原框架：ShortGINE、LongPoly、门控、读出及层输入残差：[PNG](01_original.png) / [SVG](01_original.svg) / [PDF](01_original.pdf)

2. FPPool：直接替换Pool和残差增加是两个方案：[PNG](02_fppool.png) / [SVG](02_fppool.svg) / [PDF](02_fppool.pdf)

3. 编码后Cross：保留双通道，在Filter后、Pool前交互：[PNG](03_post_cross.png) / [SVG](03_post_cross.svg) / [PDF](03_post_cross.pdf)

4. 逐层Cross：替换LongPoly，保留SHORT、门控、残差和读出：[PNG](04_branch_cross.png) / [SVG](04_branch_cross.svg) / [PDF](04_branch_cross.pdf)

5. Loss：动态差值、组件辅助、ACA迁移是三类独立实验：[PNG](05_losses.png) / [SVG](05_losses.svg) / [PDF](05_losses.pdf)

6. 可靠性：固定Chemprop；组件均衡停止，新核心未确定：[PNG](06_reliability.png) / [SVG](06_reliability.svg) / [PDF](06_reliability.pdf)

[六页PDF](all_diagrams.pdf) / [预览全部图](gallery.html) / [历史结果和方法来源](../../reports/teacher_brief_20261006/report.md)。SVG保留文字，编辑需中文字体。省略维度、部分归一化与详细评估流程，以代码和报告为准。

10项源码/报告SHA绑定见sources.json，其中原项目残差FPPool只读。用户附图只参考布局、存哈希，未复制发布；无新增论文结果或虚构化学案例。

复现（仓库根目录）：

```powershell
& D:/Tools/conda-envs/chemprop_baseline/python.exe tools/draw_graphcliff_changes.py
```

使用已有Matplotlib、Windows微软雅黑和DejaVu Sans回退，无新安装；异机需提供绑定原项目文件、调整字体路径，脚本非跨平台框架。

三级审核见verification.json。六图经目视；PNG尺寸、SVG XML及PDF页对象计数通过。初次缺下标字形失败，添加已有字体回退修复；4次渲染调用，1失败3成功（2次修订重绘）。修正结构指纹归属、训练标签箭头、AtomEncoder说明、候选/消融对照关系及标签重叠。PDF核验环境未安装fitz/pypdf，采用标准库检查Matplotlib生成文件中的Page对象数，未安装新包；不是完整PDF解析。默认sandbox git读取失败，授权仓库环境恢复，无文件损坏。

0训练/推理；模型、配置、缓存、权重、原图及GraphCliff/my_work不变。M40停止结论不改，这些图不能证明发表可行性。
