# FPPool正式先导执行方案

执行前保存，2026-10-07。授权：用户在收到“先做固定FPPool正式筛查”的建议后回复“试试”。仅启动本文件的FP先导，不恢复旧队列，不启动逐层替换、组合或额外seed。

输入：`D:/GraphCliff-main/benchmark_data`只读CSV；配置`fp_screen.json`原样执行。来源/划分/Cliff覆盖/null实际扰动覆盖见`fp_preflight_20261007.json`。split_seed42，训练seed42，H256/3层，最多100epoch、patience15、batch32；AdamW及全部参数沿用已提交配置，不在看到结果后调参。每个任务base、feature_joint、output_joint、feature_frozen、output_frozen、output_frozen_null，共12次训练上限。

所有臂使用同轮底座共享初始化；冻结臂使用同任务base最佳checkpoint，训练期间固定底座eval及requires_grad=False。选模只用validation Overall MSE；官方test标签不解析，test不运行。单分子任务不使用参考标签预测；既有通用训练器仍生成配对记录，但这里只作为样本迭代/身份，不把它解释为参考回归效果。

主候选固定为output_frozen，不从六臂中事后改选最佳者。首先要求它相对base在两个任务的Overall RMSE、Cliff RMSE均严格降低。机制特异性要求同候选在两个任务两指标均低于feature_frozen及output_frozen_null，才有继续扩大验证的依据。其余joint臂用于解释输出空间与冻结因素，不替代主候选通过。任一要求不满足则不扩seed/任务，不用layer或组合补救。该严格规则控制本轮投入，不是期刊涨点要求；单seed不作普遍有效/无效或显著性声明。

两个任务均为反复使用的开发任务；baseline选模与冻结纠错复用同一validation，会产生额外选模适应。当前不能当独立确认或正式test证据。固定行反转null虽改变membership，也不代表所有化学信息已消除。

输出：`artifacts/mechanism_retry_fp_screen_20261007`，目录存在则拒绝覆盖。运行日志和PID另外保存到忽略目录；失败保留现场、不覆盖历史输出。完成全部12组后用既有audit独立核对预测/指标、来源hash、冻结tensor和残差风险恒等式，报告主/控制指标、总/可训练参数、含底座的分阶段耗时/峰值显存、null覆盖和停止决定。不因先导某一任务不通过而改门槛或换任务。

命令：

```powershell
& D:\Tools\conda-envs\graphcliff\python.exe -u -m experiments.mechanism_retry.run --config experiments/mechanism_retry/fp_screen.json --csv-root D:\GraphCliff-main\benchmark_data --output artifacts/mechanism_retry_fp_screen_20261007
& D:\Tools\conda-envs\graphcliff\python.exe -m experiments.mechanism_retry.audit artifacts/mechanism_retry_fp_screen_20261007
```

本轮不改模型/训练器/固定配置，完成后写本目录的`fp_results_20261007.md`及对应核验JSON，更新项目状态并提交当前独立分支。数据、权重、逐分子预测和日志不公开上传。
