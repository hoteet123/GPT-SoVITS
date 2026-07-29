param(
    [ValidateSet("HF", "HF-Mirror", "ModelScope")]
    [string]$Source = "HF",

    [switch]$DownloadUVR5
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Get-Urls {
    param([string]$Source)

    switch ($Source) {
        "HF" {
            return @{
                Pretrained = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
                G2PW       = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
                UVR5       = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
                NLTK       = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
                OpenJTalk  = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
            }
        }
        "HF-Mirror" {
            return @{
                Pretrained = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
                G2PW       = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
                UVR5       = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
                NLTK       = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
                OpenJTalk  = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
            }
        }
        "ModelScope" {
            return @{
                Pretrained = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
                G2PW       = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
                UVR5       = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip"
                NLTK       = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
                OpenJTalk  = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
            }
        }
    }
}

function Receive-File {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$OutFile
    )

    Write-Host "Downloading $OutFile"
    $Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($Curl) {
        & $Curl.Source -L --fail --retry 5 --retry-delay 5 -C - -o $OutFile $Uri
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed while downloading $OutFile"
        }
        return
    }

    Invoke-WebRequest -Uri $Uri -OutFile $OutFile
}

function Expand-Zip {
    param(
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    Write-Host "Extracting $Archive"
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    Remove-Item -LiteralPath $Archive -Force
}

$Urls = Get-Urls $Source

if (-not (Test-Path "GPT_SoVITS/pretrained_models/sv")) {
    Receive-File -Uri $Urls.Pretrained -OutFile "pretrained_models.zip"
    Expand-Zip -Archive "pretrained_models.zip" -Destination "GPT_SoVITS"
} else {
    Write-Host "Pretrained models already exist. Skipping."
}

if (-not (Test-Path "GPT_SoVITS/text/G2PWModel")) {
    Receive-File -Uri $Urls.G2PW -OutFile "G2PWModel.zip"
    Expand-Zip -Archive "G2PWModel.zip" -Destination "GPT_SoVITS/text"
} else {
    Write-Host "G2PWModel already exists. Skipping."
}

if ($DownloadUVR5) {
    if (-not (Test-Path "tools/uvr5/uvr5_weights")) {
        Receive-File -Uri $Urls.UVR5 -OutFile "uvr5_weights.zip"
        Expand-Zip -Archive "uvr5_weights.zip" -Destination "tools/uvr5"
    } else {
        Write-Host "UVR5 weights already exist. Skipping."
    }
}

$PythonPrefix = (python -c "import sys; print(sys.prefix)").Trim()
Receive-File -Uri $Urls.NLTK -OutFile "nltk_data.zip"
Expand-Zip -Archive "nltk_data.zip" -Destination $PythonPrefix

Receive-File -Uri $Urls.OpenJTalk -OutFile "open_jtalk_dic_utf_8-1.11.tar.gz"
$OpenJTalkTarget = (python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))").Trim()
tar -xzf "open_jtalk_dic_utf_8-1.11.tar.gz" -C $OpenJTalkTarget
Remove-Item -LiteralPath "open_jtalk_dic_utf_8-1.11.tar.gz" -Force

Write-Host "Model download complete."
