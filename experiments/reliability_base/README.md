# M21：可靠预测方向的可复用底座与接口验收

2026-10-05。**底座已选、作者代码最小运行已通过。** 不是新算法或效果证明。用户已明确：以发刊为目标，尽量站在成熟公开项目上改进，不限定GraphCliff。历史阴性结论保留；此前迁移/组合未提供充分贡献，不等于任何组合或针对性改进都不可研究。

## 选择与责任

|组件|采用/角色|核验状态|
|---|---|---|
|[作者粗糙度项目](https://github.com/krishnatheaverage/qsar-landscape-roughness/tree/91522b8b81a52d5c95ae216d01a00cbbf0c66209)|RF工程参考、邻域特征、校准基线；未改作者算法|MIT原LICENSE/4文件hash；最小运行通过|
|[UNIQUE](https://github.com/Novartis/UNIQUE/tree/c6d65b9c63102bc18e68c25c98d76e24650c3e4a)|误差模型与UQ评价参考，后续按需薄适配|BSD-3-Clause已核；未安装或运行，不称端到端接通|
|SelectiveNet官方及第三方PyTorch实现|方法对照来源；暂不复制源码|2完整固定树未找到许可证文件，见provenance；不认定方法不可使用|
|成熟分子编码器|后续固定一个公开GNN底座，保留其原任务/特征/训练设置|GraphCliff可选，不再必须；本轮未接新GNN|

本轮实际新增只有编排/身份/边界核验薄脚本；指纹、RF、粗糙度与校准公式均复用作者实现。没有重写编码器、另建训练平台或模型工厂。作者源码/许可证按原样复制到忽略运行目录，并保留原文件hash；公开仓库只保存适配、元数据和聚合核验。

## 固定预算与实际结果

从已有CHEMBL234_Ki seed42 manifest取前128训练行、前32验证行，均来自官方train；32验证行在作者临时CSV中标`test`仅满足其入口，不是官方test。原CSV与manifest只读、hash不变，canonical train/validation重叠为0。任务选择是最早已有完整准备资产的工程样本，不据本轮成绩择优。

3次作者RF200树冒烟：原始、只把查询标签加1000、只将训练标签seed0置换。每次输出32行，顺序/合法特征有限值核验通过，耗时约1.80/1.53/1.48秒（环境/规模限定，总约4.83秒，不是正式成本）。RF是低成本工程参考，不是论文主模型。

|接口检查|结果与含义|
|---|---|
|查询标签变化|nbr_disp/SALI/holder/距离/密度/大小/rf_var保持不变（atol1e-10）|
|回顾性列|dirichlet/lipschitz随查询标签变化，明确不得作为部署风险输入|
|训练标签变化|sali_mean发生变化，证明这些“无查询活性”特征仍依赖训练邻居标签|
|纯校准接口|复用作者cq/qbins，16校准/16评价验证行、2bin，尺寸/有限值通过|
|输入身份|原文件hash未变；官方test行使用数0；没有缓存混用/覆盖|

不报告这些32样本上的涨点、名义覆盖或投稿证据；16/16与2bin只是低成本冒烟设置，未冒充作者50重复/5bin。训练标签置换是依赖检查，不是固定模型下的性能零分布。源脚本的rf_err/cliff_mol等输出仅描述性，未进入风险特征白名单。正式OOF、独立校准、风险模型选模与最终test仍须另行固定完整方案。

原始分子/标签/行号/缓存/stdout留`artifacts/reliability_base_smoke_20261005`；公开[smoke_summary.json](smoke_summary.json)与[provenance.json](provenance.json)记录版本、hash、计划/实际和偏离，未公开逐分子预测。

## 复现入口

使用已有含numpy/pandas/scikit-learn/RDKit的环境；本机为`D:/Tools/conda-envs/chemprop_baseline/python.exe`，Python3.11.15、sklearn1.9.0、RDKit2026.03.4。没有安装新依赖，也不把该环境称为作者锁定环境。固定源码URL和hash来自[来源清单](../../docs/research/roughness_method_review_20261005/provenance.json)。合法取得作者文件后执行（全新输出）：

```powershell
python experiments/reliability_base/run_smoke.py --source-root artifacts/roughness_sources_20261005/qsar-landscape-roughness --csv D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv --manifest artifacts/prepare/CHEMBL234_Ki/pairs.json --output artifacts/new_reliability_smoke
```

源码缓存不随仓库发布。只需按来源清单下载LICENSE、src/config.py、src/build_features.py、src/conformal.py，保持相对路径；脚本会核对预期SHA256。示例取得源码（拒绝覆盖不一致文件）：

```python
import hashlib, json, urllib.request
from pathlib import Path
meta = json.loads(Path('docs/research/roughness_method_review_20261005/provenance.json').read_text(encoding='utf-8'))
target = Path('artifacts/roughness_sources_20261005/qsar-landscape-roughness')
needed = {'LICENSE', 'src/config.py', 'src/build_features.py', 'src/conformal.py'}
for item in meta['sources']:
    if item['repo'] != 'krishnatheaverage/qsar-landscape-roughness' or item['path'] not in needed:
        continue
    data = urllib.request.urlopen(item['url'], timeout=30).read()
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    file = target / item['path']
    assert not file.exists() or file.read_bytes() == data
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
```

私有manifest也不分发，第三方需自行生成并核对样本身份；不同划分不称精确重放。`--output`拒绝已有目录，防止结果覆盖。工具执行三个契约断言并留下日志；发生失败保留现场，不自动换参数重跑。

## 发表目标与下一里程碑

研究任务是单靶标活性回归及预测风险/接受决策。先建立成熟活性基线与风险基线，随后只提出一个针对具体不足的改进。GraphCliff只是可选对照；RF通过接口不等于主算法已确定。复用、组合并不自动否决贡献，但最终仍须说明具体方法差异、简单替代、公平消融与跨任务重复，不能以“公开模块拼装”直接宣称原创。

下一里程碑是固定一个成熟分子模型和合法OOF/风险校准角色，先完成有限基线验证方案（任务、seed、预算、指标、数值继续/停止条件），再执行正式试验。不恢复旧队列，不以冒烟成绩挑任务/继续门槛。不预设改进有效或保证二区；已实现的是可复用底座，发表目标仍未完成。

最终解释仍必须提供结构、真值/预测/误差、固定成功/失败/一般案例、重要对随机同规模掩蔽和跨seed稳定性；拒答另展示风险及合法邻域证据。注意力/指纹/粗糙度关联不独立证明化学因果。见[解释规范](../../docs/research/framework_review_20261005/explainability.md)。

异常：GraphCliff环境无sklearn，改用已安装chemprop环境；两SelectiveNet仓库缺许可证未接入；GitHub CLI沙箱读取配置失败后授权重试。无训练失败或重跑；实际3次RF冒烟/0正式效果训练/0test。原项目/my_work、模型/Loss/旧配置与数据不改。


## M22补充：本地模型资产

M22完成本地复用核验：30 Chemprop权重hash相符，27神经对照资产存在；优先复用官方Chemprop2.2.3及本项目已有身份/验证工具。0训练/0预测/0test，旧validation不能作独立校准。来源、版本和两次清点中断见[本地复用报告](local_reuse.md)。下一步固定有限基线协议。


## M23：固定协议

M23：有限基线协议已冻结：234/244/4792仅official train，fit/monitor/calibration/evaluation角色分开并按canonical排重；先seed42三任务，筛查通过才43/44，最多90次fit（含辅助RF），当前0训练/0预测/0test。数值止损、解释、异常和预检修订见[固定协议](fixed_protocol.md)。下一步仅薄CLI编排与契约验收，再执行A；不是新算法效果。


## M25：阶段A结论

M25阶段A完成并独立核验：三任务seed42风险曲线改善2.08%/2.49%/3.76%，均值2.78%；未达训练前继续门槛，停止具体粗糙度增强候选，B不执行。这是小幅正向信号，不是粗糙度普遍无效。30/30fit、0官方test；固定六个失败/一般结构案例及解释局限已保留。方法贡献/跨seed/忠实度仍缺，发表目标未完成。见[阶段A结果与停止报告](phase_a_report.md)。
