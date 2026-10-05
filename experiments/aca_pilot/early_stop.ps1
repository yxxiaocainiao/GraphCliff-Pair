# SPDX-License-Identifier: GPL-3.0-only
# Apply the already fixed task-level veto after three paired seeds; preserve artifacts.
$ErrorActionPreference = 'Stop'
$runRoot = 'D:\GraphCliff-Pair\artifacts\aca_pilot_20261005'
$allRecords = @(Get-Content -LiteralPath "$runRoot\summary.json" -Raw | ConvertFrom-Json)
$scope = @($allRecords | Where-Object { $_.dataset -eq 'CHEMBL234_Ki' })
if ($scope.Count -ne 6) { Write-Output 'WAIT_FOR_THREE_PAIRED_SEEDS'; exit 0 }
foreach ($seed in @(42,43,44)) {
    foreach ($arm in @('mse','aca')) {
        if (@($scope | Where-Object { $_.seed -eq $seed -and $_.arm -eq $arm }).Count -ne 1) { throw 'Incomplete/duplicate scoped matrix' }
    }
}
$base = @($scope | Where-Object { $_.arm -eq 'mse' })
$aca = @($scope | Where-Object { $_.arm -eq 'aca' })
$cliffChange = ($aca | Measure-Object -Property cliff_rmse -Average).Average / ($base | Measure-Object -Property cliff_rmse -Average).Average - 1
$overallChange = ($aca | Measure-Object -Property overall_rmse -Average).Average / ($base | Measure-Object -Property overall_rmse -Average).Average - 1
if ($cliffChange -le 0.03 -and $overallChange -le 0.01) { Write-Output 'TASK_VETO_NOT_TRIGGERED'; exit 0 }
if (Test-Path -LiteralPath "$runRoot\completed.json") { throw 'Full matrix already completed; use full audit instead' }
$training = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*experiments.aca_pilot.run*' -and $_.CommandLine -like '*aca_pilot_20261005*' })
if ($training.Count -ne 1) { throw 'Training process identity is not unique' }
$trainingProcessId = $training[0].ProcessId
Stop-Process -Id $trainingProcessId -ErrorAction Stop
$partial = @(Get-ChildItem -LiteralPath $runRoot -Filter history.json -Recurse | Where-Object { -not (Test-Path -LiteralPath (Join-Path $_.Directory.FullName 'summary.json')) } | ForEach-Object { $_.Directory.FullName })
$closure = @{status='paused_early';reason='user requested immediate pause on poor outcomes; existing any-task Cliff>3% or Overall>1% deterioration veto established after three paired seeds';planned_runs=18;completed_runs=$allRecords.Count;audited_dataset='CHEMBL234_Ki';cliff_relative_change=$cliffChange;overall_relative_change=$overallChange;stopped_process_id=$trainingProcessId;interrupted_folders=$partial;test_evaluated=$false;utc_time=[DateTime]::UtcNow.ToString('o');all_artifacts_preserved=$true}
$closure | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$runRoot\stopped.json" -Encoding utf8
Write-Output ($closure | ConvertTo-Json -Depth 10)
