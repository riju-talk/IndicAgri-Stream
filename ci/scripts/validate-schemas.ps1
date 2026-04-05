param(
    [string]$BaseBranch = "main"
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

function Test-GitBranchExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Branch
    )

    & git rev-parse --verify $Branch *> $null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-Path -Path "schemas" -PathType Container)) {
    Write-Host "No schemas directory found. Skipping schema validation."
    exit 0
}

if (-not (Get-Command buf -ErrorAction SilentlyContinue)) {
    Write-Error "buf is required for schema validation."
    exit 1
}

try {
    Write-Host "Running buf lint..."
    Invoke-CheckedCommand -Command "buf" -Arguments @("lint", "schemas/")

    if (-not (Test-GitBranchExists -Branch $BaseBranch)) {
        & git fetch origin ("{0}:{0}" -f $BaseBranch) --depth=1 *> $null
    }

    if (Test-GitBranchExists -Branch $BaseBranch) {
        Write-Host "Running buf breaking check against $BaseBranch..."
        Invoke-CheckedCommand -Command "buf" -Arguments @("breaking", "schemas/", "--against", ".git#branch=$BaseBranch")
    }
    else {
        Write-Host "Base branch '$BaseBranch' not available locally. Skipping buf breaking check."
    }
}
catch {
    Write-Error $_
    exit 1
}
