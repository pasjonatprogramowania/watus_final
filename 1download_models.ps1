$workDir = "./watus_project/models"

if (-not (Test-Path -Path $workDir -PathType Container)) {
    New-Item -Path $workDir -ItemType Directory | Out-Null
}

Set-Location -Path $workDir

$piperUrl = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
$piperZip = "piper_windows_amd64.zip"
$piperNewDir = "piper"

Invoke-WebRequest -Uri $piperUrl -OutFile $piperZip
Expand-Archive -Path $piperZip -DestinationPath . -Force
Set-Location -Path $piperNewDir

$modelOnnxUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx"
$modelJsonUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx.json"
$phonemizeUrl = "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_windows_amd64.zip"
$phonemizeZip = "piper-phonemize_windows_amd64.zip"

Invoke-WebRequest -Uri $modelOnnxUrl -OutFile "pl_PL-darkman-medium.onnx"
Invoke-WebRequest -Uri $modelJsonUrl -OutFile "pl_PL-darkman-medium.onnx.json"
Invoke-WebRequest -Uri $phonemizeUrl -OutFile $phonemizeZip
Expand-Archive -Path $phonemizeZip -DestinationPath . -Force

Set-Location ..
Remove-Item -Path $piperZip -Force
Remove-Item -Path "./$piperNewDir/$phonemizeZip" -Force
Set-Location ..

$modelPt = "https://huggingface.co/deepghs/yolos/blob/main/rtdetr-l/model.pt"
Invoke-WebRequest -Uri $modelPt -OutFile "yolov8n.pt"

Set-Location ..