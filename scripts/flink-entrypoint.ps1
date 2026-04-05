Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$flinkHome = if ($env:FLINK_HOME) { $env:FLINK_HOME } else { "/opt/flink" }
$flinkBin = "{0}/bin/flink" -f $flinkHome
$defaultJob = if ($env:DEFAULT_JOB) { $env:DEFAULT_JOB } else { "{0}/jobs/main.py" -f $flinkHome }

$venvBin = "/opt/flink/pyflink-venv/bin"
$pathSeparator = [System.IO.Path]::PathSeparator
if (-not [string]::IsNullOrWhiteSpace($env:PATH)) {
    if (-not (($env:PATH -split [Regex]::Escape($pathSeparator)) -contains $venvBin)) {
        $env:PATH = "{0}{1}{2}" -f $venvBin, $pathSeparator, $env:PATH
    }
}
else {
    $env:PATH = $venvBin
}

function Show-Help {
    @"
Usage:
  /docker-entrypoint.ps1 help
  /docker-entrypoint.ps1 jobmanager
  /docker-entrypoint.ps1 taskmanager
  /docker-entrypoint.ps1 run [job_file.py] [additional flink args]

Examples:
  /docker-entrypoint.ps1 run /opt/flink/jobs/pipeline.py --python /opt/flink/jobs/pipeline.py
"@ | Write-Host
}

$mode = if ($args.Count -gt 0) { $args[0] } else { "help" }

switch ($mode) {
    "help" {
        Show-Help
        exit 0
    }
    "jobmanager" {
        & ("{0}/bin/jobmanager.sh" -f $flinkHome) "start-foreground"
        exit $LASTEXITCODE
    }
    "taskmanager" {
        & ("{0}/bin/taskmanager.sh" -f $flinkHome) "start-foreground"
        exit $LASTEXITCODE
    }
    "run" {
        $jobFile = if ($args.Count -gt 1 -and -not [string]::IsNullOrWhiteSpace($args[1])) { $args[1] } else { $defaultJob }
        if (-not (Test-Path -Path $jobFile -PathType Leaf)) {
            Write-Error "PyFlink job file not found: $jobFile"
            exit 1
        }

        $additionalArgs = @()
        if ($args.Count -gt 2) {
            $additionalArgs = $args[2..($args.Count - 1)]
        }

        & $flinkBin "run" "-py" $jobFile @additionalArgs
        exit $LASTEXITCODE
    }
    default {
        if ($args.Count -eq 0) {
            Show-Help
            exit 1
        }

        $command = $args[0]
        $commandArgs = @()
        if ($args.Count -gt 1) {
            $commandArgs = $args[1..($args.Count - 1)]
        }

        & $command @commandArgs
        exit $LASTEXITCODE
    }
}
