Copy-Item .env.example .env
python -m venv .\.api_venv
.\.api_venv\Scripts\Activate
pip install -r requirements.txt
deactivate

cd .\watus_project

Copy-Item .env.example .env
python -m venv .\.watus_venv
.\.watus_venv\Scripts\Activate
pip install -r requirements.txt
deactivate
cd ..
