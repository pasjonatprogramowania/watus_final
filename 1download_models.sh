#!/bin/bash

workDir="./watus_project/models"

if [ ! -d "$workDir" ]; then
    mkdir -p "$workDir"
fi

cd "$workDir"

piperUrl="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
piperArchive="piper_linux_x86_64.tar.gz"
piperNewDir="piper"

wget "$piperUrl" -O "$piperArchive"
tar -xzf "$piperArchive"
cd "$piperNewDir" || exit

modelOnnxUrl="https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx"
modelJsonUrl="https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx.json"
phonemizeUrl="https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_linux_x86_64.tar.gz"
phonemizeArchive="piper-phonemize_linux_x86_64.tar.gz"

wget "$modelOnnxUrl" -O "pl_PL-darkman-medium.onnx"
wget "$modelJsonUrl" -O "pl_PL-darkman-medium.onnx.json"
wget "$phonemizeUrl" -O "$phonemizeArchive"
tar -xzf "$phonemizeArchive"

cd ..
rm -f "$piperArchive"
rm -f "./$piperNewDir/$phonemizeArchive"
cd ..

modelPt="https://huggingface.co/deepghs/yolos/blob/main/rtdetr-l/model.pt"
wget "$modelPt" -O "yolov8n.pt"
cd ..
