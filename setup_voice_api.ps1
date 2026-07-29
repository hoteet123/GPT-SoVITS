param(
    [ValidateSet("CPU", "CU126", "CU128")]
    [string]$Device = "CPU",

    [ValidateSet("HF", "HF-Mirror", "ModelScope")]
    [string]$Source = "HF"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Installing GPT-SoVITS dependencies and pretrained models..."
$PowerShellCommand = Get-Command pwsh -ErrorAction SilentlyContinue
if (-not $PowerShellCommand) {
    $PowerShellCommand = Get-Command powershell -ErrorAction Stop
}

& $PowerShellCommand.Source -ExecutionPolicy Bypass -File install.ps1 -Device $Device -Source $Source

Write-Host ""
Write-Host "Setup complete. Start the API with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_voice_api.ps1"
