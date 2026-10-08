# M65：合法输入已物化，五臂执行流程已接线，尚未训练

日期：2026-10-08。沿用[M64冻结协议](../conditional_pilot_prepare_20261008/protocol.json)，只处理CHEMBL234_Ki、seed42。研究问题和条件粗糙度候选不变，没有新增候选、分区或参数搜索。

## 已完成与数据边界

新增[run_conditional_pilot.py](../../../experiments/reliability_base/run_conditional_pilot.py)，复用既有`jobs_for`、`run_command`、来源hash函数、作者粗糙度入口和M63条件权重函数。没有修改历史协议、旧训练入口或M63权重实现。

最终忽略目录`artifacts/conditional_pilot_inputs_20261008_v2/`实际生成12份作业CSV：4份train_val、4份仅含smiles的query、4份作者特征fixture。另存私有身份、源fit标签、作业命令、协议快照、作者MIT许可/config/build_features副本及文件hash。逐分子文件均不公开。首份准备目录保留，不覆盖，原因见核验记录。

|项目|数量及处理|
|---|---|
|允许的结构|2339：fit1403、monitor351、evaluation585|
|真实标签|1754：仅fit1403与monitor351；使用原CSV的`y`列|
|OOF查询|468、468、467；各参考935、935、936|
|full查询|evaluation585；参考fit1403|
|monitor|每个点模型351，仅供点模型早停|
|calibration|585整块保留，未解析结构或标签|
|evaluation标签|未读取；query无标签，所有fixture查询y固定为0|
|官方test|未解析记录；只对源CSV整体作字节hash|

原CSV读取使用`usecols`和预先确定的`skiprows`，先排除禁止记录，再解析允许列。结构只读取smiles/split，标签另一次只读取fit/monitor的y；不使用整行DictReader。源文件仍只读。核对源CSV、角色manifest及全部M64来源hash；三个作者文件另与既有来源记录核对。

## 五臂接线

五臂均向相同规格的风险RF输入**原始7维特征**，监督目标均为源分子的绝对OOF误差，区别只在源样本权重。不会把源权重当作目标风险直接排序。

|臂|源权重计算|
|---|---|
|RF7|全1|
|IW-GEN5|source-only缩放后的前5维，独立source/target Scott KDE|
|IW-STRUCT7|Morgan radius2/2048，source-only PCA7 full/no-whiten，再source-only缩放和KDE|
|IW-RISK7|7维联合密度比|
|IW-COND7|7维联合log比减前5维边缘log比；边缘沿对应联合带宽|

联合与条件臂复用M63同一次调用。总量仍是8KDE、2scaler、1PCA、5风险RF；所有预处理仅fit源数据。PCA源秩不足7、非法身份/误差、非有限值及密度数值失败均停止，不换维度、不均匀回退。

已写`execute`函数，但CLI仅开放prepare/check，本轮未调用execute。未来必须有明确真实训练授权才调用；函数拒绝重复执行，不自动重试。4次点训练、4次点预测和4次作者特征作业共享于全部五臂。每次训练900秒、预测120秒、特征300秒；后处理在独立子进程中限600秒，整体限5400秒。4次训练各自900秒给出累计训练wall上限3600秒。每步取单步限额与整体剩余额度较小者；错误/超时/中断记入execution日志并抛出，不继续下一作业。

后处理只提取作者缓存的6个合法列，点预测另补为第7列。作者fixture中的查询y为0，作者生成的rf_err/dirichlet/lipschitz等字段不进入估权或风险RF。全五臂分数、输入清单、脚本和协议hash写入score_lock后，才允许读取585条evaluation的y并计算六比例RMSE。排序/并列/ceil规则沿旧实现；100%结果以浮点容差核对。

## 核验与证据强度

可运行检查：[check.py](check.py)；聚合记录：[verification.json](verification.json)。

```powershell
& D:/Tools/conda-envs/chemprop_baseline/python.exe docs/research/conditional_pilot_inputs_20261008/check.py
& D:/Tools/conda-envs/chemprop_baseline/python.exe experiments/reliability_base/run_conditional_pilot.py check --output artifacts/conditional_pilot_inputs_20261008_v2
```

前三次递增自检均通过，全部0真实/合成估计器fit、0实际predict、0真实模型加载。第一次检查毒标签/五臂mock/13阶段mock/超时与重试拒绝；第二次增加synthetic score-lock与旧metrics AST六点曲线比较；第三次增加条件/联合权重绑定区分、源秩停止与整体预算停止。后两次是新增检查覆盖，不是失败补跑；M63 KDE测试未重做。score-lock检查只对20个合成结构计算Morgan位向量，无真实特征重算。

首份真实输入prepare及其内置check、随后独立check通过。最终审查发现未来execute若传入相对路径，后处理子进程会从日志目录错误解析路径；增加入口resolve，后续用相对路径的mock验证修正。因脚本hash已改变，首份目录保留并降为旧准备，不重写它的hash或CSV；另在v2目录准备并核验最终输入。两次prepare均0训练，不是失败后静默补跑。训练/查询/fixture身份顺序、占位标签、参考标签一致性、组件角色/折隔离、canonical隔离及文件hash均通过；已有输出与工作区外路径拒绝写入。最终目录内没有execution、checkpoint、真实features/cache或predictions文件。

**事实：** 输入与接线完成，且无训练。**支持的解释：** 可以进入一次有预算的真实先导，不必再增加准备阶段或模块。**未验证：** 实际GPU运行、作者真实特征生成、真实密度稳定性、五臂训练时长与收益；mock通过不能证明这些事项。

补充检查记录：首次相对路径自检在测试夹具的`os.path.relpath`处失败（临时目录在C盘、工作区在D盘，Windows不支持跨盘相对路径），未到真实执行。改为在临时目录父目录内切换cwd，再用相对路径调用mock入口；最终自检通过。故自检共5次：4通过、1测试夹具失败，均0实际fit/predict；原失败保留在verification，不记成5次全部通过。修正后的最终v2独立输入检查和两类输出拒绝检查均通过，拒绝后私有文件hash不变。

这仍是已经用于开发的临时组件分区，没有时间外推证据，也不是最终独立测试。条件权重的理论对象仍是混合分布H，不保证目标Q风险降低。没有新增算法效果、化学解释或期刊录用结论。

## 下一步

后续明确授权启动时，仅执行已冻结的234×seed42五臂真实先导，按四对照分别判断条件候选是否继续；不扩任务、种子或预算。结果可能是保留，也可能是停止。当前交付不自动触发训练。
