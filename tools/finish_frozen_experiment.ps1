param([Parameter(Mandatory=$true)][int]$ObserverPid)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = 'D:\Tools\conda-envs\graphcliff\python.exe'
$statePath = Join-Path $projectRoot 'artifacts/finalization_state.json'
Set-Location -LiteralPath $projectRoot

function Write-FinalState($phase, $stage, $childId, $message) {
    @{status=$phase; current=$stage; child_pid=$childId; message=$message; updated_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $statePath
}

function Invoke-ExperimentStage([string]$Stage, [string[]]$ProgramArgs) {
    $logPrefix = Join-Path $projectRoot ('artifacts/finalization_' + $Stage)
    $child = Start-Process -FilePath $pythonPath -ArgumentList $ProgramArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput ($logPrefix + '.out.log') -RedirectStandardError ($logPrefix + '.err.log') -PassThru
    Write-FinalState 'running' $Stage $child.Id '按冻结顺序执行；失败即停止，不覆盖已有结果'
    $child.WaitForExit()
    if ($child.ExitCode -ne 0) { throw "阶段失败: $Stage, exit=$($child.ExitCode)，请查看对应日志" }
}

$pipelineLock = $null
try {
    $pipelineLock = [System.IO.File]::Open((Join-Path $projectRoot 'artifacts/finalization.lock'), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
} catch {
    throw '无法取得收尾执行锁；已有收尾进程或锁文件不可访问，拒绝重复启动'
}

try {
    if ($ObserverPid -le 0) { throw '必须指定已确认的训练队列观察进程' }
    $observed = Get-CimInstance Win32_Process -Filter "ProcessId = $ObserverPid"
    if ($observed) {
        if ($observed.CommandLine -notlike ('*' + (Join-Path $projectRoot 'tools/run_queue.ps1').Replace('/','\') + '*')) {
            throw '指定PID不是本项目训练队列，拒绝等待无关进程'
        }
        Write-FinalState 'waiting' 'all_training_and_stage_audits' $ObserverPid '等待已启动的恢复和seed44队列，不启动额外训练'
        $process = Get-Process -Id $ObserverPid -ErrorAction SilentlyContinue
        if ($null -ne $process) { $process.WaitForExit() }
    }
    $queueState = Get-Content -LiteralPath 'artifacts/queue_state.json' -Raw | ConvertFrom-Json
    if ($queueState.status -ne 'completed' -or $queueState.current -ne 'all_frozen_validation_queues') {
        throw '训练队列尚未完整完成或阶段审计失败，禁止进入收尾'
    }
    $runs = @('artifacts/interaction_seed42_20261004', 'artifacts/interaction_seed43_44_20261004',
              'artifacts/ablation_seed42_20261004', 'artifacts/ablation_seed43_recovered_20261005',
              'artifacts/ablation_seed44_20261005')
    $expectedCounts = @(8,16,18,18,18)
    for ($index=0; $index -lt $runs.Count; $index++) {
        $completion = Get-Content -LiteralPath (Join-Path $runs[$index] 'completed.json') -Raw | ConvertFrom-Json
        if ($completion.runs -ne $expectedCounts[$index] -or $completion.test_evaluated -ne $false) {
            throw "阶段完成标记不符合固定矩阵: $($runs[$index])"
        }
    }
    $csvRoot = 'D:/GraphCliff-main/benchmark_data'
    $freezePath = 'artifacts/final_validation_freeze_20261005.json'
    $testOutput = 'artifacts/final_test_20261005'
    Invoke-ExperimentStage -Stage 'validation_report' -ProgramArgs (@('tools/summarize_results.py','--runs') + $runs + @('--csv-root',$csvRoot,'--phase','full','--output-prefix','docs/full_validation_results'))
    Invoke-ExperimentStage -Stage 'replay_and_freeze' -ProgramArgs (@('tools/freeze_validation.py','--runs') + $runs + @('--csv-root',$csvRoot,'--report-json','docs/full_validation_results.json','--output',$freezePath,'--device','cuda'))
    Invoke-ExperimentStage -Stage 'final_test' -ProgramArgs @('tools/evaluate_test.py','--freeze',$freezePath,'--csv-root',$csvRoot,'--output',$testOutput)
    Invoke-ExperimentStage -Stage 'test_report' -ProgramArgs @('tools/summarize_test.py','--freeze',$freezePath,'--test-output',$testOutput,'--csv-root',$csvRoot,'--output-prefix','docs/final_test_results')
    Write-FinalState 'completed' 'reports_ready_for_final_review' 0 '78模型报告、权重重放、冻结与一次test完成；待最终结论核对和发布'
} catch {
    Write-FinalState 'failed' 'inspect_finalization_logs' 0 $_.Exception.Message
    Write-Error $_.Exception.Message
    exit 1
} finally {
    if ($null -ne $pipelineLock) { $pipelineLock.Dispose() }
}
