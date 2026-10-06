# M34：CARA实际字段核验与用途边界

## 决定

CARA v1具备assay、靶点与端点类型元数据，可继续核验assay外验证的适用性；本次七张Task表均无日期/年份字段，**不能直接用作时间验证**。算法核心仍0，不新增训练或评价。CARA和MoleculeACE均涉及ChEMBL，不能仅因仓库不同称独立外部数据；具体分子/assay重合尚未查明。

## 固定输入、版本与执行

输入为[Zenodo不可变v1记录](https://zenodo.org/records/11063965)中的CARA.zip，数据CC-BY-4.0；沿用M33官方代码Apache-2.0核验。[协议](protocol.json)先在80eb76d推送，读取脚本9118f92、端点字段修复d5a5ee8均在对应正式读取前推送。只下载一个包，不下载baseline.zip，不解压文件或执行档案代码。

实测106,000,271字节，MD5 `b6bd8b9ff0ce6752fbc8bf0a45d71b6b`与官方记录一致，SHA256 `8d6ab35e17bfd27454b8960375d1f62ca3b21f0e40dde2ca34483105432ebb33`；下载91.907秒，小于300秒上限。ZIP共33个目录/文件条目。逐文件读取Task表时ZIP CRC由zipfile检查；未声称对全部其余成员做CRC核验。

计划/实际：1/1次下载、1/2次schema扫描，另1次独立元数据复算及1次最终可复现脚本重放（共3次全表扫描）；0fit/0权重推理/0性能评价/0新划分/0参数搜索，GPU耗时0。CPU耗时未单独测量，不以wall替代CPU。首次扫描漏识别`Value Type`，修复后再读，原输出保留。该故障仅涉及字段识别，不是性能阴性或额外拟合。行解析会解码CSV整行，但仅统计合法元数据列；未提取、分析或展示pChEMBL数值、分子结构或模型误差。

## 实测表头与数据规模

| 表 | 行数 |
|---|---:|
| ChEMBL30_seq.csv | 140816 |
| LO_All.tsv | 1187136 |
| LO_GPCR.tsv | 321904 |
| LO_Kinase.tsv | 200800 |
| VS_All.tsv | 1237256 |
| VS_GPCR.tsv | 70179 |
| VS_Kinase.tsv | 84605 |

这是各表行数，不是独立样本数；All及子集存在范围重叠，禁止相加作为独立数据规模。六张LO/VS表共享字段：空名称索引列、Task ID、Assay ChEMBL ID、Molecule ChEMBL ID、Target ChEMBL ID、Smiles、Value Type、pChEMBL Value、Task Type、Target Cluster 0.3/0.4/0.5/0.6、Target Type。序列表为Task ID、Assay ChEMBL ID、Target ChEMBL ID、Target Sequence、protein_class_id。

Task ID/Assay ChEMBL ID/Target ChEMBL ID在各表均无空缺，Value Type在六张活动表无空缺。日期、年份、文献ID、原始单位与relation字段均不在这些表头中。不能从“pChEMBL Value”列名直接推导所有记录具备本项目要求的等号测定及统一实验条件。Task Type不是日期，Target Cluster不是化学系列。

## 三个靶点的明确Ki元数据

仅LO_All，按Target ChEMBL ID精确匹配、Value Type==Ki统计：

| 靶点 | 全端点记录 | Ki记录 | Ki不同assay数 |
|---|---:|---:|---:|
| CHEMBL234 | 5775 | 5112 | 335 |
| CHEMBL244 | 6862 | 3823 | 209 |
| CHEMBL4792 | 3733 | 1917 | 49 |

这些是活动行/assay元数据计数，不是去重分子数、独立组件数、合法未来样本数或悬崖对数。相同靶点和Ki仅通过粗匹配，仍缺结构重合、重复测定、条件及来源核验，不能直接与MoleculeACE混并或评价。LO_All含IC50/Ki/EC50/Kd等七类端点，不能跨端点混为同一标签。

## 核验与复现

脚本自检覆盖标签/结构列排除、日期/assay/端点/靶点识别与Value Type回归。另一路独立按文本行/表头复算七表全部行数及表头；显式DictReader元数据选列复算三靶点端点分布，与首次扫描靶点总数一致；档案hash一致。事后将三靶点端点交叉计数加入同一脚本并重放一次；全部表头/行数/元数据及独立Ki/assay计数一致，明确属于可复现性整理而非新增预注册分析。完整聚合、源码hash、原始核验文件hash及运行数见[results.json](results.json)。

```powershell
& 'D:\Tools\conda-envs\chemprop_baseline\python.exe' experiments/reliability_base/audit_cara_schema.py --selfcheck
& 'D:\Tools\conda-envs\chemprop_baseline\python.exe' experiments/reliability_base/audit_cara_schema.py --archive artifacts/cara_schema_audit_20261006/CARA.zip --output artifacts/cara_schema_audit_20261006/replay.json
```

公开仅协议、薄脚本、聚合与hash；档案、初次/修复扫描和核验现场在忽略目录`artifacts/cara_schema_audit_20261006`。原GraphCliff、my_work、PDF和旧协议保持只读。

## 事实、解释、未知及停止理由

**观察事实：** 七张Task表无逐记录日期，assay/target/Value Type存在，三靶点Ki记录非空；本轮无模型成绩。

**已支持解释：** CARA默认NewAssay不能直接实现时间排序；仅靠该包无法完成计划中的时间验证。assay外数据准备有可用标识，但尚未证明独立性。

**待验证假设：** 可能通过固定ChEMBL版本的assay/document关联补来源年份；即使获得文献年份，也只是来源时间代理，不自动等于测定时间。

**未知原因：** 档案为何省略日期、本项目能否获得足够独立系列/未来Ki记录、粗糙度能否迁移以及算法增量都未确认。缺日期不表示所有时间路线不可行；元数据丰富不表示方法有效。

**停止范围：** 停止“直接拿CARA档案作时间验证”的做法，不停止全部研究；不追加模块或扩大训练。改变这一判断需合法、可追溯的逐活动/assay时间来源与映射覆盖证据，另冻范围后再核。

## 下一步

优先用已核验三任务联合组件，冻结角色与最小独立组件/样本/悬崖要求后做一次有限可行性检查；CARA仅作为候选assay外来源，先核重合与质量，不评价模型。随后最多保留一个可定位的方法增量及最小反证。没有核心不进入9次风险先导或30任务。时间支线须另有可靠来源才启动；暂不下载全量ChEMBL或换源找阳性。二区优先、三区最差情况备选，工作量不构成录用保证，旧18fit/90fit/100GPUh预算不变。
