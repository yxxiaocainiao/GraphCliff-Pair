# M27：有限补证准备与零拟合简单对照

2026-10-06。**G0投入筛查通过：现有增强误差RF在三个任务上都优于作者式等权组合。可以继续准备小幅增量的补证，但新算法贡献仍未建立；本轮0fit、0checkpoint预测、0官方test，不恢复旧B/三板斧。** 方案在计算前提交并推送14b1d87，已有seed42评价此前看过，因此此次明确属事后探索，不是新确认数据。

## 已固定的框架与当前结果

SMILES → 官方Chemprop D-MPNN活性预测 → fit内折外绝对残差 → 普通风险RF / 加入合法训练邻域粗糙度的风险RF → 固定接受比例评价 → 后续校准/结构归因。四个原风险臂共享点预测；本轮额外验证唯一简单组合 `z_OOF(rf_var)+z_OOF(sali_mean)`，均值/标准差只取fit-OOF，固定等权，不用evaluation标签选权重。作者已经有方差与粗糙度组合，我们只作归纳化适配，不能称首次粗糙度拒答。

|任务|普通RF|简单组合|增强RF|增强对普通改善|增强对组合改善|
|---|---:|---:|---:|---:|---:|
|CHEMBL234_Ki|0.728290|0.732673|0.713141|+2.08%|+2.67%|
|CHEMBL244_Ki|0.875086|0.867937|0.853321|+2.49%|+1.68%|
|CHEMBL4792_Ki|0.759667|0.733798|0.731103|+3.76%|+0.37%|

六个接受比例.5/.6/.7/.8/.9/1的RMSE算术均值，越低越好；不是全体点预测RMSE涨幅，也不是连续AURC积分。增强对组合三个任务均值相对改善1.57%，4792仅0.37%，该任务优势尤其脆弱；234简单组合不及普通RF，不能用弱对照替代原主比较。G0规则至少2任务正向、任务相对改善平均>0且无任务退化>5%，三项通过；不意味着统计显著、原创或二区录用，旧M25 No-Go不改。

## 方法差异与下一步投入

[粗糙度作者项目](https://github.com/krishnatheaverage/qsar-landscape-roughness)已有组合与条件校准，[UNIQUE作者项目](https://github.com/Novartis/UNIQUE)已有监督误差模型。我们的实际差异是D-MPNN折外误差监督加合法邻域特征，不是新Loss/新注意力。现在支持“优于两类固定替代方案”的有限证据，尚不能证明特定粗糙度机制：特征维度增加及随机分裂改变也可能影响结果。待验证假设是粗糙度与结构/残差之间的匹配关系起作用，而非仅增加两维特征。下一步优先同维度置乱粗糙度对照，使用已有OOF缓存及相同RF配置，先固定随机方式/预算/指标，再少量RF拟合；不新训主干来回避机制缺口。

跨seed草案已在[补证协议](incremental_evidence_protocol.md)登记，当前未冻结执行；43/44只改变训练随机性，沿用同一分区，不称跨划分独立确认。强ensemble/MVE对照、特征特异性、跨系列/时间与完整归因仍缺，不能单靠此表形成完整方法论文。不存在已证明的新算法名称或机制。若下一步同维度对照削弱贡献，再停止具体候选，不添加新模块。

新发现的成本复用点：作者build_features.py使用 `np.random.seed(0)` 和 `RandomForestRegressor(...random_state=0)`，与Chemprop seed无关；固定角色/折/顺序、参考标签及作者版本时，12份作者特征缓存可复用到43/44，只有prediction列需要新Chemprop输出替换。原计划每seed都重做12份特征RF，非必要。若缓存身份/配置/哈希及不含新prediction依赖全部验收通过，新增两seed只需24 Chemprop＋12风险RF＝36fit（非60fit）；加上先导30仍在原90总预算内。当前只是源码已确认的复用路径，未实际跑43/44，不把缓存复用写成已完成验收。原M23预算及历史配置不改，新正式方案必须明确更新计数。

最终可解释性仍是必需交付：分开活性原子/子结构归因和风险依据；固定改善/失败/一般案例，top20%对20个同规模随机mask的预测变化，跨seed稳定性。Chemprop2.2.3接口先检查，未通过就披露；粗糙度/风险关联不是化学因果。仅G0不建设完整系统，已有六个有限结构案例保留。

## 来源、核验、异常与边界

[聚合数值及输入SHA256](simple_control_results.json)，[复用脚本](check_simple_risk_control.py)，私有缓存输入 `artifacts/reliability_phase_a_20261005_v3`，新增私有输出 `artifacts/reliability_simple_control_20261006_v2`。输入仅OOF/cal/evaluation及已绑定配置/作者源码；继承M25身份manifest，原数据/模型/源目录只读。源码基准14b1d87；旧训练基准见M25，不能将这次源码说成其训练版本。

复用旧metrics/cq函数，并且只发布组合的排序曲线，不发布将有符号z分数当区间尺度的附加输出。重算四旧臂共72个曲线数值，与保存指标误差<1e-12；再独立用stdlib CSV/statistics重建三个任务18个新曲线数值，误差<1e-12；更改查询y/cliff不改变组合分数。OOF/cal/eval source_row与原manifest逐项一致、canonical互斥、来源hash前后不变。组合标准化只从OOF计算，OOF参考规模偏移仍未排除。

实际1次成功计算；前1次在结果写入阶段因相对/绝对路径混用而中断，未输出成功结果，留下空私有目录；修复路径规范化后写新v2，不覆盖旧现场，两个尝试均0fit。另有两处猜测路径未找到，已通过rg定位真实runner及不带seed层的CSV；不是训练失败。在线固定conformal.py页面返回Cache miss，转用既有固定缓存及hash，不擅自换main版本。未安装依赖，旧方案/runner/模型/Loss不改。

```powershell
python experiments/reliability_base/check_simple_risk_control.py --input artifacts/reliability_phase_a_20261005_v3 --output artifacts/new_simple_control
```

输出已存在则拒绝覆盖。提交前核对公开范围、表格、来源hash、链接和diff，推送后核对本地远端提交及公开属性。新协议计算前已提交；结果单独提交，不把事后对照说成M23原计划。

设计支持：使用experimental-design技能处理对应任务/相同预测比较及正确重复层级，未安装其DOE依赖；expression-skill/planning-with-files在指定位置未找到，沿用已有task_plan/notes记录。工具引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。本轮核对arXiv当前作者/年份，不作为分子方法效果证据。
