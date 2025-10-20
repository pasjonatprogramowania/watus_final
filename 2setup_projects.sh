#!/bin/bash
# włączasz: sudo apt install -y python3-venv python3-pip && ./2setup_projects.sh
# Główny katalog
cp .env.example .env
python3 -m venv ./api_venv
source ./api_venv/bin/activate
pip install -r requirements.txt
deactivate

# Katalog watus_project
cd ./watus_project

cp .env.example .env
python3 -m venv ./watus_venv
source ./watus_venv/bin/activate
pip install -r requirements.txt
deactivate

cd ..