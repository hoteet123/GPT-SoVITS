param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 9881,
    [string]$Config = "GPT_SoVITS/configs/tts_infer.yaml"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python -m voice_api.server --host $HostName --port $Port --config $Config
