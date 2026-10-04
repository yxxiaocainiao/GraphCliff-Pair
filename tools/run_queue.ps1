param([int]$WaitPid, [switch]$ResumeAfterSeed42, [switch]$RecoverSeed43)
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
    if ($RecoverSeed43 -and $ResumeAfterSeed42) { throw '恢复入口互斥，只选择一个' }
    if ($RecoverSeed43) {
        $activeTraining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'graphcliff_pair\.train|recover_validation\.py' }
        if ($activeTraining) { throw '已有项目训练或恢复进程，拒绝重复启动' }
        $recoveredOutput = 'artifacts/ablation_seed43_recovered_20261005'
        if (Test-Path -LiteralPath $recoveredOutput) { throw '恢复目录已存在，拒绝覆盖' }
        $recoveryArgs = @('tools/recover_validation.py','--source','artifacts/ablation_seed43_20261004','--config','configs/ablation_seed43.json','--csv-root','D:/GraphCliff-main/benchmark_data','--output',$recoveredOutput)
        $child = Start-Process -FilePath $pythonPath -ArgumentList $recoveryArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot ($recoveredOutput + '.out.log')) -RedirectStandardError (Join-Path $projectRoot ($recoveredOutput + '.err.log')) -PassThru
        Write-QueueState 'running' 'ablation_seed43_recovered_20261005' $child.Id '原样复用14个完整模型，其余4个按冻结配置从头训练；保留失败现场'
        $child.WaitForExit()
        if ($child.ExitCode -ne 0) { throw "恢复训练失败，保留新输出: exit=$($child.ExitCode)" }
        & $pythonPath tools/audit_runs.py --runs $recoveredOutput --csv-root D:/GraphCliff-main/benchmark_data --output ($recoveredOutput + '_audit.json')
        if ($LASTEXITCODE -ne 0) { throw '恢复阶段独立审计失败' }
    } elseif ($ResumeAfterSeed42) {
        $activeTraining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'graphcliff_pair\.train' }
        if ($activeTraining) { throw '已有项目训练进程，拒绝启动重复队列' }
        $completedStages = @('artifacts/interaction_seed42_20261004','artifacts/interaction_seed43_44_20261004','artifacts/ablation_seed42_20261004')
        Write-QueueState 'auditing' 'completed_before_resume' 0 '先联合审计已完成42次训练，不重启或覆盖'
        & $pythonPath tools/audit_runs.py --runs @completedStages --csv-root D:/GraphCliff-main/benchmark_data --output artifacts/resumed_completed_validation_audit.json
        if ($LASTEXITCODE -ne 0) { throw '恢复前联合审计失败' }
    } else {
        if ($WaitPid -le 0) { throw '首次观察队列需要明确WaitPid；恢复使用ResumeAfterSeed42' }
        Write-QueueState 'waiting' 'interaction_seed42_20261004' $WaitPid '等待已启动的具体进程；不会重启原队列'
        $observedProcess = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
        if ($null -ne $observedProcess) { $observedProcess.WaitForExit() }
        if (-not (Test-Path -LiteralPath 'artifacts/interaction_seed42_20261004/completed.json')) {
            throw '原进程已终止但首队列未完成，停止后续，不重启或覆盖'
        }
        & $pythonPath tools/audit_runs.py --runs artifacts/interaction_seed42_20261004 --csv-root D:/GraphCliff-main/benchmark_data --output artifacts/interaction_seed42_audit.json
        if ($LASTEXITCODE -ne 0) { throw '首队列独立审计失败' }
    }
    $stages = @(
        @{config='configs/interaction_seed43_44.json'; name='interaction_seed43_44_20261004'},
        @{config='configs/ablation_seed42.json'; name='ablation_seed42_20261004'},
        @{config='configs/ablation_seed43.json'; name='ablation_seed43_20261004'},
        @{config='configs/ablation_seed44.json'; name='ablation_seed44_20261004'}
    )
    if ($ResumeAfterSeed42) { $stages = $stages[2..3] }
    if ($RecoverSeed43) { $stages = @(@{config='configs/ablation_seed44.json'; name='ablation_seed44_20261005'}) }
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
