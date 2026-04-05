param(
    [string]$Service = "",
    [string]$ImageTag = "latest",
    [string]$RefTag = ""
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

$pushImage = if ($env:PUSH_IMAGE) { $env:PUSH_IMAGE } else { "false" }
$registry = if ($env:REGISTRY) { $env:REGISTRY } else { "ghcr.io/local/indicagri-stream" }

if ([string]::IsNullOrWhiteSpace($Service)) {
    Write-Error "Usage: .\\ci\\scripts\\build-docker.ps1 -Service <api|flink|airflow> [-ImageTag latest] [-RefTag branch-name]"
    exit 1
}

$dockerfile = Join-Path "ci/docker" ("Dockerfile.{0}" -f $Service)
if (-not (Test-Path -Path $dockerfile -PathType Leaf)) {
    Write-Error "Dockerfile not found: $dockerfile"
    exit 1
}

switch ($Service) {
    "api" {
        if (-not (Test-Path -Path "api" -PathType Container)) {
            Write-Host "api/ directory not found. Skipping API image build."
            exit 0
        }
    }
    "flink" {
        if (-not (Test-Path -Path "flink" -PathType Container)) {
            Write-Host "flink/ directory not found. Skipping Flink image build."
            exit 0
        }
        if (-not (Test-Path -Path "scripts/flink-entrypoint.sh" -PathType Leaf)) {
            Write-Host "scripts/flink-entrypoint.sh not found. Skipping Flink image build."
            exit 0
        }
    }
    "airflow" {
        if (-not (Test-Path -Path "dags" -PathType Container)) {
            Write-Host "dags/ directory not found. Skipping Airflow image build."
            exit 0
        }
    }
    default {
        Write-Error "Unknown service '$Service'. Expected one of: api, flink, airflow."
        exit 1
    }
}

$imageName = "{0}/{1}:{2}" -f $registry, $Service, $ImageTag
Write-Host "Building $imageName from $dockerfile..."
Invoke-CheckedCommand -Command "docker" -Arguments @("build", "-f", $dockerfile, "-t", $imageName, ".")

if ($pushImage -eq "true") {
    Write-Host "Pushing $imageName..."
    Invoke-CheckedCommand -Command "docker" -Arguments @("push", $imageName)

    if (-not [string]::IsNullOrWhiteSpace($RefTag)) {
        $safeRefTag = $RefTag -replace "/", "-"
        $refImageName = "{0}/{1}:{2}" -f $registry, $Service, $safeRefTag

        Invoke-CheckedCommand -Command "docker" -Arguments @("tag", $imageName, $refImageName)
        Invoke-CheckedCommand -Command "docker" -Arguments @("push", $refImageName)
    }
}
else {
    Write-Host "PUSH_IMAGE=false; build complete, push skipped."
}
