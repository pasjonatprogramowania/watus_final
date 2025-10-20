# PowerShell script dla Docker Setup Watus AI

Write-Host "=== Docker Setup dla Watus AI ===" -ForegroundColor Green

Write-Host "Sprawdzanie czy modele są pobrane..." -ForegroundColor Yellow
if (-Not (Test-Path "watus_project\models")) {
    Write-Host "Brak katalogu models. Uruchom najpierw: .\1download_models.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "Sprawdzanie plików .env..." -ForegroundColor Yellow
# Kopiowanie .env (jak w 2setup_projects.sh)
if (-Not (Test-Path ".env")) {
    Write-Host "Kopiowanie .env.example -> .env" -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
}

if (-Not (Test-Path "watus_project\.env")) {
    Write-Host "Kopiowanie watus_project\.env.example -> watus_project\.env" -ForegroundColor Cyan
    Copy-Item "watus_project\.env.example" "watus_project\.env"
}

Write-Host "Zatrzymywanie istniejących kontenerów..." -ForegroundColor Yellow
docker-compose down --remove-orphans

Write-Host "Budowanie obrazów Docker..." -ForegroundColor Yellow
docker-compose build --no-cache

Write-Host "Uruchamianie wszystkich serwisów..." -ForegroundColor Green
docker-compose up -d

Write-Host ""
Write-Host "=== Wszystkie serwisy uruchomione ===" -ForegroundColor Green
Write-Host "- API (uvicorn):      http://localhost:8000" -ForegroundColor Cyan
Write-Host "- Reporter:           Działa w tle" -ForegroundColor Cyan
Write-Host "- Camera Runner:      Działa w tle" -ForegroundColor Cyan
Write-Host "- Watus:              Działa w tle" -ForegroundColor Cyan
Write-Host ""
Write-Host "Komendy do monitorowania:" -ForegroundColor Yellow
Write-Host "  docker-compose logs -f                    # Wszystkie logi" -ForegroundColor White
Write-Host "  docker-compose logs -f api               # Logi API" -ForegroundColor White
Write-Host "  docker-compose logs -f reporter          # Logi Reporter" -ForegroundColor White
Write-Host "  docker-compose logs -f camera_runner     # Logi Camera Runner" -ForegroundColor White
Write-Host "  docker-compose logs -f watus            # Logi Watus" -ForegroundColor White
Write-Host "  docker-compose ps                        # Status serwisów" -ForegroundColor White
Write-Host ""
Write-Host "Zatrzymanie wszystkich serwisów:" -ForegroundColor Yellow
Write-Host "  docker-compose down" -ForegroundColor White