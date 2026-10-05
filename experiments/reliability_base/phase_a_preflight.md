# M24：阶段A入口与无拟合验收

2026-10-05。已实现[薄编排入口](run_phase_a.py)，复用官方Chemprop CLI、作者原样特征代码、作者cq纯函数和已安装sklearn RF。没有改编码器、Loss、作者公式或重写训练循环；UNIQUE整流水线未运行，按协议采用已声明的RF等价适配。

12组输入/CLI默认解析、来源/分区/折/canonical隔离检查通过；作者cq完成13个边界/秩检查，离散风险曲线有确定性无拟合检查。模型标签只进入fold fit＋monitor入口；query CSV仅SMILES。作者特征query采用y=0/cliff=0占位，参考是该折实际fit标签；查询真实误差不作为白名单特征。作者RF辅助输出rf_err不作为Chemprop风险目标。

旧Chemprop234权重以绑定hash核验，只做2个fit角色结构的无标签CLI接口预测，返回形状/顺序/有限值通过；不进入正式模型或风险实验，不提供成绩证据。正式A全部新初始化，OOF风险RF只fit OOF绝对误差，calibration/evaluation不训练风险模型。准备文件hash在run前再次核验；有execution.json拒绝重跑，不覆盖现场。

首次准备因UTF8来源清单被GBK解码而中断；第二次因官方CLI的Rich日志GBK编码中断；第三次修复共享子进程编码后通过。三个目录均留忽略区，0训练，接口预测job实际2次（1失败＋1成功）。这些是代码修复，未改参数/任务/数值筛查；Triton/性能建议记录但未安装新依赖或改精度。

已校验全部有效CLI默认并保存私有effective_cli_defaults；官方训练还将保存自己的配置。主预算按[固定协议](fixed_protocol.md)：阶段A只seed42、12主干＋12作者RF＋6风险RF；每主干≤15分钟、阶段≤180分钟。训练源码版本写execution.json；失败/超时立即留下状态，不自行换参数/补跑。GPU峰值显存尚未采样，不能在结果里编数字。

```powershell
python experiments/reliability_base/run_phase_a.py prepare --output artifacts/new_phase_a --data-root D:/GraphCliff-main/benchmark_data --partitions artifacts/reliability_protocol_20261005_v2 --author artifacts/roughness_sources_20261005/qsar-landscape-roughness
python experiments/reliability_base/run_phase_a.py run --output artifacts/new_phase_a --data-root D:/GraphCliff-main/benchmark_data --partitions artifacts/reliability_protocol_20261005_v2 --author artifacts/roughness_sources_20261005/qsar-landscape-roughness
```

用既有chemprop_baseline环境，不安装依赖。原数据/分区/作者缓存/my_work只读，源码复制及训练/预测/逐分子校准表留新忽略目录。公开[验收摘要](phase_a_preflight.json)无分子行号、预测或权重。协议最终manifest与源码预期hash必须一致，不同数据/协议不能假称原实验重放。

下一步按此代码执行A并公开全部聚合结果。A不通过停止增强候选，交付预定6个最大/中位误差私有结构案例与公开汇总/解释局限，B不运行。A通过才有资格按固定规则补seed确认，仍不把成熟机制组合称新算法。
