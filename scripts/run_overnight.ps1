# powershell -ExecutionPolicy Bypass -File scripts\run_overnight.ps1
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
    "[{0:HH:mm:ss}] START {1}" -f $t0, $name | Tee-Object -FilePath "$logs\progress.txt" -Append
    & $py @argv *> "$logs\$name.log"
    $code = $LASTEXITCODE
    "[{0:HH:mm:ss}] END   {1} exit={2} ({3:N0} min)" -f (Get-Date), $name, $code, ((Get-Date) - $t0).TotalMinutes |
        Tee-Object -FilePath "$logs\progress.txt" -Append
}

# Питання 1: аудит розмітки C2A
Invoke-Step "q1_main" @("src\exp_q1_audit.py", "--seeds", "0", "1", "2", "--epochs", "10", "--tag", "main")
# Маршрутизація на C2A
Invoke-Step "routing_c2a" @("src\exp_routing.py", "--dataset", "c2a", "--q1-tag", "main", "--seeds", "0", "1", "2")
# Питання 2: АМН на SARD зі стартом від C2A-моделі
Invoke-Step "q2_al_sard" @("src\al_loop.py", "--strategy", "random", "uncertainty", "--seeds", "0", "1", "2",
    "--pool-json", "data\processed\sard\pool.json", "--pool-images", "data\processed\sard\images",
    "--test-json", "data\processed\sard\test.json", "--test-images", "data\processed\sard\images",
    "--init-weights", $c2aWeights, "--n0", "50", "--batch-size", "50", "--rounds", "6", "--epochs", "15",
    "--eval-init", "--tag", "q2_sard", "--experiment", "q2_al_sard")
# Маршрутизація на SARD test: модель після АМН і модель лише C2A
$alWeights = "data\work\al\q2_sard_uncertainty_seed0\runs\r6\weights\last.pt"
Invoke-Step "routing_sard_al" @("src\exp_routing.py", "--dataset", "sard", "--weights", $alWeights, "--seeds", "0", "1", "2", "--tag", "al_r6")
Invoke-Step "routing_sard_c2a" @("src\exp_routing.py", "--dataset", "sard", "--weights", $c2aWeights, "--seeds", "0", "1", "2", "--tag", "c2a_only")
"[{0:HH:mm:ss}] ALL DONE" -f (Get-Date) | Tee-Object -FilePath "$logs\progress.txt" -Append
