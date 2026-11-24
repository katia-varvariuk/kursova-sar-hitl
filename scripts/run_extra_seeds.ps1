# Додаткові сіди 3 і 4 для питання 2 (налаштування як у run_overnight.ps1).
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "runs\overnight"
New-Item -ItemType Directory -Force $logs | Out-Null
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$c2aWeights = "runs\baseline_c2a\yolo11n_e30\weights\best.pt"

function Invoke-Step($name, [string[]]$argv) {
    $t0 = Get-Date
    "[{0:HH:mm:ss}] START {1}" -f $t0, $name | Tee-Object -FilePath "$logs\progress_extra.txt" -Append
    & $py @argv *> "$logs\$name.log"
    $code = $LASTEXITCODE
    "[{0:HH:mm:ss}] END   {1} exit={2} ({3:N0} min)" -f (Get-Date), $name, $code, ((Get-Date) - $t0).TotalMinutes |
        Tee-Object -FilePath "$logs\progress_extra.txt" -Append
}

Invoke-Step "q2_al_sard_seeds34" @("src\al_loop.py", "--strategy", "random", "uncertainty", "--seeds", "3", "4",
    "--pool-json", "data\processed\sard\pool.json", "--pool-images", "data\processed\sard\images",
    "--test-json", "data\processed\sard\test.json", "--test-images", "data\processed\sard\images",
    "--init-weights", $c2aWeights, "--n0", "50", "--batch-size", "50", "--rounds", "6", "--epochs", "15",
    "--eval-init", "--tag", "q2_sard", "--experiment", "q2_al_sard")
# аналіз: питання 1 — 3 сіди, питання 2 — 5 сідів
Invoke-Step "analyze_5seeds" @("src\analyze.py", "--q1-tag", "main", "--q2-tag", "q2_sard",
    "--seeds", "0", "1", "2", "--q2-seeds", "0", "1", "2", "3", "4")
"[{0:HH:mm:ss}] ALL DONE" -f (Get-Date) | Tee-Object -FilePath "$logs\progress_extra.txt" -Append
