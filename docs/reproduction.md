# 复现固定实验

本说明对应仓库中的固定配置；实际进度见task_plan.md。已有24次交互验证完成，54次消融正在运行，最终test未执行。以下测试命令是后续步骤，不是已完成证据。

## 环境与输入

在项目checkout根目录执行。使用Python3.10，先安装适合设备的PyTorch2.7，再执行README的editable安装、FPPool下载和单元测试。实际本机版本见environment_verified.json；CPU CI只证明接口测试通过，不证明完整GPU实验已复现。GPU确定性采用上游warn_only策略，不保证跨硬件逐位一致。

数据CSV应来自固定GraphCliff数据版本，包含smiles/y/split/cliff_mol。本机原始数据目录为D:/GraphCliff-main/benchmark_data，只读；其他机器自行指定目录。两任务文件字节SHA256如下，改编码、行序或换数据版本后不得称本次精确复现：

|CSV|SHA256|
|---|---|
|CHEMBL234_Ki.csv|447d38f659b86c7177f022b16f028db6828ab09c80051e74dad420168970f657|
|CHEMBL244_Ki.csv|b296c5956c9b3659fa039f057ebcb617919acd759c9255ed72bee917d99b4ef2|

可用PowerShell的`Get-FileHash -Algorithm SHA256`核对。先用README的smoke命令检查环境；smoke不能代替正式预算或作方法排名。数据和权重不随公开仓库分发。

## 五个正式队列

从干净checkout开始，保持训练源码和来源清单不变。各阶段必须使用全新输出目录；入口拒绝覆盖，无自动恢复训练功能。以下命令串行运行，训练/审计失败即停止。本机已经运行的队列无需重启；tools/run_queue.ps1绑定本机路径和已有进程，不是通用的新机器入口。

```powershell
$pairCsvRoot = 'D:/GraphCliff-main/benchmark_data'
$pairStages = @('interaction_seed42','interaction_seed43_44','ablation_seed42','ablation_seed43','ablation_seed44')
foreach ($pairStage in $pairStages) {
    python -m graphcliff_pair.train --config "configs/$pairStage.json" --csv-root $pairCsvRoot --output "artifacts/repro_$pairStage"
    if ($LASTEXITCODE -ne 0) { throw "训练失败: $pairStage" }
    python tools/audit_runs.py --runs "artifacts/repro_$pairStage" --csv-root $pairCsvRoot --output "artifacts/repro_${pairStage}_audit.json"
    if ($LASTEXITCODE -ne 0) { throw "审计失败: $pairStage" }
}
$pairRuns = $pairStages | ForEach-Object { "artifacts/repro_$_" }
python tools/summarize_results.py --runs @pairRuns --csv-root $pairCsvRoot --phase full --output-prefix artifacts/repro_validation_results
if ($LASTEXITCODE -ne 0) { throw '完整验证汇总失败' }
```

数量应为8+16+18+18+18=78个训练模型。每个checkpoint由最低无权重validation Overall MSE选择，最大100epoch、patience15；不得以Cliff最好轮次或test重选。报告必须保留全部三种子和条件比较，不能挑最优组。交互扩展门槛只控制额外任务；本次实际结果为No-Go，必要三机制消融仍完成。

## 完整验证后冻结，再执行一次test

只有上述78次全部完成且审计/报告核对通过才能执行。freeze_validation会在生成冻结文件前重放全部保存权重的validation；也可先用replay_validation.py生成证据，再传入--replay-json。严格重放必须使用原训练设备类型：本次GPU训练须用CUDA；CPU与GPU的matmul数值差异不以放宽容差处理。结构重复或无效分子检查失败时保留错误，不改划分来继续，不按test调整模型。

```powershell
python tools/freeze_validation.py --runs @pairRuns --csv-root $pairCsvRoot --report-json artifacts/repro_validation_results.json --output artifacts/repro_validation_freeze.json
if ($LASTEXITCODE -ne 0) { throw '验证冻结失败，禁止test' }
python tools/evaluate_test.py --freeze artifacts/repro_validation_freeze.json --csv-root $pairCsvRoot --output artifacts/repro_final_test
if ($LASTEXITCODE -ne 0) { throw '最终test未完成，检查原输出状态' }
python tools/summarize_test.py --freeze artifacts/repro_validation_freeze.json --test-output artifacts/repro_final_test --csv-root $pairCsvRoot --output-prefix artifacts/repro_test_results
if ($LASTEXITCODE -ne 0) { throw '最终test独立核对失败' }
```

同一冻结内容绑定唯一输出目录，登记在artifacts/test_evaluation_registry内；复制冻结JSON或换输出目录不能用于重复评估。JSON状态和预测输出原子发布，纯预测阶段未登记的孤立文件会重新生成。测试先完成全部固定模型的只输入SMILES预测，再读取test标签；训练参考活性仅从既定train行加载。只有未读取标签的中断预测阶段可以显式使用--resume-predictions；已开始标签评估或已完成的测试拒绝重复执行。报告核对不重新预测或选模。

## 保留的复现证据

保留各阶段manifest.json（环境、配置、Git提交、源哈希、dirty状态）、任务pairs.json（行号/参考/数据哈希）、模型initialization.json、history.json、best.pt、validation_predictions.csv、summary.json、completed.json，以及独立audit和最终freeze/test报告。终止输出不能冒充完成。公开发布配置、汇总和来源证据；原始数据、预测与checkpoint留在忽略的artifacts内。

已完成的本机交互数值见interaction_three_seed_results.md及对应audit.json。新的复跑结果需独立核对；此说明中的正式命令/最终测试流程不能视作已在第二台机器端到端验证。
