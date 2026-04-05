param(
    [string]$Mode = "all"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

function Run-Lint {
    $lintTargets = @()

    foreach ($dir in @("api", "ml", "dags")) {
        if (Test-Path -Path $dir -PathType Container) {
            $lintTargets += $dir
        }
    }

    if ($lintTargets.Count -eq 0) {
        Write-Host "No lint targets found (api/, ml/, dags/). Skipping flake8/black."
        return
    }

    Write-Host "Running flake8 on: $($lintTargets -join ' ')"
    Invoke-CheckedCommand -Command "flake8" -Arguments ($lintTargets + @("--max-line-length=120"))

    Write-Host "Running black check on: $($lintTargets -join ' ')"
    Invoke-CheckedCommand -Command "black" -Arguments (@("--check") + $lintTargets)
}

function Run-Tests {
    if (-not (Test-Path -Path "tests" -PathType Container)) {
        Write-Host "No tests directory found. Skipping pytest."
        return
    }

    $covArgs = @()
    if (Test-Path -Path "api" -PathType Container) {
        $covArgs += "--cov=api"
    }
    if (Test-Path -Path "ml" -PathType Container) {
        $covArgs += "--cov=ml"
    }

    Write-Host "Running pytest..."
    Invoke-CheckedCommand -Command "pytest" -Arguments (@("tests/", "-v") + $covArgs + @("--cov-report=xml"))
}

function Run-Protolint {
    if (-not (Test-Path -Path "schemas" -PathType Container)) {
        Write-Host "No schemas directory found. Skipping protolint."
        return
    }

    if (-not (Get-Command protolint -ErrorAction SilentlyContinue)) {
        Write-Host "protolint is not installed. Skipping proto lint."
        return
    }

    Write-Host "Running protolint..."
    Invoke-CheckedCommand -Command "protolint" -Arguments @("lint", "schemas/")
}

try {
    switch ($Mode) {
        "lint" {
            Run-Lint
            Run-Protolint
        }
        "test" {
            Run-Tests
        }
        "all" {
            Run-Lint
            Run-Tests
            Run-Protolint
        }
        default {
            Write-Error "Unknown mode '$Mode'. Use: lint, test, or all."
            exit 1
        }
    }
}
catch {
    Write-Error $_
    exit 1
}
