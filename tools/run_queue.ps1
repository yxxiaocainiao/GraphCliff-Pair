param([Parameter(Mandatory=$true)][int]$WaitPid)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = 'D:\Tools\conda-envs\graphcliff\python.exe'
$statePath = Join-Path $projectRoot 'artifacts/queue_state.json'
Set-Location -LiteralPath $projectRoot

function Write-QueueState($phase, $current, $childId, $message) {
    @{status=$phase; current=$current; child_pid=$childId; message=$message; updated_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $statePath
}

try {
    Write-QueueState 'waiting' 'interaction_seed42_20261004' $WaitPid '等待已启动的具体进程；不会重启原队列'
    $observedProcess = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($null -ne $observedProcess) { $observedProcess.WaitForExit() }
    if (-not (Test-Path -LiteralPath 'artifacts/interaction_seed42_20261004/completed.json')) {
        throw '原进程已终止但首队列未完成，停止后续，不重启或覆盖'
    }
    & $pythonPath tools/audit_runs.py --runs artifacts/interaction_seed42_20261004 --csv-root D:/GraphCliff-main/benchmark_data --output artifacts/interaction_seed42_audit.json
    if ($LASTEXITCODE -ne 0) { throw '首队列独立审计失败' }
    $stages = @(
        @{config='configs/interaction_seed43_44.json'; name='interaction_seed43_44_20261004'},
        @{config='configs/ablation_seed42.json'; name='ablation_seed42_20261004'},
        @{config='configs/ablation_seed43.json'; name='ablation_seed43_20261004'},
        @{config='configs/ablation_seed44.json'; name='ablation_seed44_20261004'}
    )
    foreach ($stage in $stages) {
        $outputPath = 'artifacts/' + $stage.name
        if (Test-Path -LiteralPath $outputPath) { throw "输出已存在，拒绝重用: $outputPath" }
        $trainArgs = @('-m','graphcliff_pair.train','--config',$stage.config,'--csv-root','D:/GraphCliff-main/benchmark_data','--output',$outputPath)
        $child = Start-Process -FilePath $pythonPath -ArgumentList $trainArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot ($outputPath + '.out.log')) -RedirectStandardError (Join-Path $projectRoot ($outputPath + '.err.log')) -PassThru
        Write-QueueState 'running' $stage.name $child.Id '串行运行冻结配置，结束后审计再进入下一阶段'
        $child.WaitForExit()
        if ($child.ExitCode -ne 0) { throw "训练失败: $($stage.name), exit=$($child.ExitCode)" }
        & $pythonPath tools/audit_runs.py --runs $outputPath --csv-root D:/GraphCliff-main/benchmark_data --output ($outputPath + '_audit.json')
        if ($LASTEXITCODE -ne 0) { throw "独立审计失败: $($stage.name)" }
    }
    Write-QueueState 'completed' 'all_frozen_validation_queues' 0 '全部24次交互与54次消融训练完成并分别审计；尚未评估test'
} catch {
    Write-QueueState 'failed' 'inspect_logs' 0 $_.Exception.Message
    Write-Error $_.Exception.Message
    exit 1
}
