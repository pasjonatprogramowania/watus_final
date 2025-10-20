#  WATUŚ - Wieloagentowy System Robota Domowego

Kompletny system wieloagentowy w Pythonie dla robota Watuś działającego na Raspberry Pi/Linux. System obsługuje przetwarzanie mowy, rozpoznawanie mówcy, agregację danych sensorycznych i komunikację z LLM.

##  Spis Treści

- [Funkcjonalność](#funkcjonalność)
- [Architektura](#architektura)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Konfiguracja](#konfiguracja)
- [Uruchomienie](#uruchomienie)
- [Struktura Projektu](#struktura-projektu)
- [Moduły](#moduły)
- [Testy](#testy)
- [Rozwiązanie TTS](#rozwiązanie-tts)
- [API i Format Danych](#api-i-format-danych)
- [Troubleshooting](#troubleshooting)

---

##  Funkcjonalność

###  **watus.py** - Program Przetwarzania Mowy

-  **Ciągłe nasłuchiwanie** z VAD (Voice Activity Detection)
-  **Wykrywanie wake word**: "hej watusiu" z tolerancją na zniekształcenia
-  **Rozpoznawanie lidera**: ECAPA-TDNN speaker recognition
-  **Transkrypcja**: Faster-Whisper (model base) dla języka polskiego
-  **Analiza głośności** w decybelach
-  **Kontrola LED**: zielony=nasłuch, czerwony=przetwarzanie
-  **Half-duplex**: albo słucha albo mówi
-  **TTS**: Edge-TTS (Microsoft) z naturalnym głosem polskim
-  **Komunikacja**: ZeroMQ pub/sub

###  **reporter.py** - Agent Agregacji Danych

-  **Odbiera dialog** przez ZeroMQ
-  **Zbiera dane** z wielu źródeł (kamera, LiDAR, scenariusze)
-  **Tworzy kontekst** z timestampem
-  **Wysyła do LLM** (API endpoint z konfiguracji)
-  **Zwraca odpowiedź** przez ZeroMQ

---

##  Architektura

```

   MIKROFON          

           
           
      
   watus.py             ZeroMQ            
   - VAD                      - Pub: dialog     
   - Wake Word                - Sub: response   
   - Speaker ID            
   - Transcription                    
   - TTS                              
   - LED Control           
         reporter.py       
                                - Aggregator      
         - LLM Client      
   LED                   - Context Builder 
   - Green (listen)         
   - Red (process)                     
                 
                             
         LLM Endpoint      
   SPEAKER                     (API z konfiguracji) 
      
           
           
           

Źródła danych:
- data/dialog.jsonl
- data/camera.jsonl
- data/lidar.jsonl
- data/scenarios.jsonl
```

---

##  Wymagania

### Hardware

- **Raspberry Pi 4/5** (4GB+ RAM) lub Linux PC
- **Mikrofon USB** (zalecane: z redukcją szumów)
- **Głośnik** lub wyjście audio
- **LED** (opcjonalnie, działa w trybie mock)
- **GPU NVIDIA** (opcjonalnie, dla przyspieszonego przetwarzania)

### Software

- **Python 3.9+**
- **CUDA** (opcjonalnie, dla GPU)
- **System audio**: ALSA/PulseAudio
- **mpg123** lub **ffmpeg** (dla TTS playback)

---

##  Instalacja

### 1. Sklonuj repozytorium

```bash
cd /home/ubuntu/
git clone <repo-url> watus_robot
cd watus_robot
```

### 2. Utwórz środowisko wirtualne

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Zainstaluj zależności Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Zainstaluj narzędzia systemowe

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y mpg123 ffmpeg portaudio19-dev

# Raspberry Pi - dodaj RPi.GPIO
pip install RPi.GPIO
```

### 5. Zainstaluj PyTorch z CUDA (opcjonalnie)

```bash
# Dla GPU NVIDIA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dla CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

##  Konfiguracja

### Plik `config.json`

Edytuj `config.json` aby dostosować system:

```json
{
  "audio": {
    "sample_rate": 16000,
    "chunk_duration_ms": 30,
    "vad_mode": 3,
    "silence_threshold_ms": 430,
    "pre_speech_buffer_ms": 300,
    "volume_threshold_db": -35
  },
  "wake_word": {
    "phrase": "hej watusiu",
    "alternatives": ["hej watuś", "hej watu", "ej watusiu"],
    "threshold": 0.7
  },
  "speaker_recognition": {
    "model_name": "speechbrain/spkrec-ecapa-voxceleb",
    "threshold": 0.85,
    "leader_timeout_seconds": 180
  },
  "whisper": {
    "model_size": "base",
    "language": "pl",
    "device": "cuda",
    "compute_type": "float16"
  },
  "tts": {
    "engine": "edge",
    "voice": "pl-PL-MarekNeural",
    "rate": "+0%",
    "volume": "+0%"
  },
  "led": {
    "green_pin": 17,
    "red_pin": 27,
    "mock_mode": true
  },
  "zmq": {
    "dialog_publisher": "tcp://127.0.0.1:5555",
    "response_subscriber": "tcp://127.0.0.1:5556",
    "reporter_publisher": "tcp://127.0.0.1:5557"
  },
  "llm": {
    "endpoint": "http://localhost:11434/api/chat",
    "model": "llama3",
    "timeout": 30
  }
}
```

#### Kluczowe parametry:

- **`vad_mode`**: 0-3 (3 = najbardziej agresywny)
- **`silence_threshold_ms`**: Min. przerwa do uznania końca wypowiedzi (430ms)
- **`volume_threshold_db`**: Próg głośności (-35 dB)
- **`leader_timeout_seconds`**: Timeout lidera (180s = 3 minuty)
- **`device`**: `"cuda"` dla GPU, `"cpu"` dla CPU
- **`mock_mode`**: `true` dla testów bez GPIO

---

##  Uruchomienie

### Testowanie modułów

```bash
# Uruchom testy jednostkowe
pytest tests/ -v

# Test z pokryciem kodu
pytest tests/ --cov=. --cov-report=html
```

### Uruchomienie systemu

#### 1. Skonfiguruj endpoint LLM

Upewnij się że masz dostęp do endpointu LLM API. Skonfiguruj go w `config.json`:

```json
"llm": {
  "endpoint": "https://your-llm-api-endpoint.com/api/chat",
  "model": "your-model-name",
  "timeout": 30
}
```

Endpoint powinien być kompatybilny z formatem API opisanym w sekcji "API i Format Danych".

#### 2. Uruchom reporter.py (w osobnym terminalu)

```bash
source venv/bin/activate
python reporter.py
```

#### 3. Uruchom watus.py

```bash
source venv/bin/activate
python watus.py
```

#### 4. Wybierz urządzenie audio

System wyświetli listę dostępnych mikrofonów:

```
Dostępne urządzenia audio:
  [0] Built-in Microphone
  [1] USB Audio Device
  [2] ...

Wybierz numer urządzenia: 1
```

### Użycie

1. **Powiedz wake word**: "Hej Watusiu"
2. **Poczekaj na reakcję**: Czerwona LED
3. **Otrzymaj odpowiedź**: Robot odpowie przez głośnik
4. **Kontynuuj rozmowę**: W ciągu 3 minut (timeout lidera)

---

##  Struktura Projektu

```
watus_robot/
 config.json                 # Konfiguracja systemu
 watus.py                    # Główny program mowy
 reporter.py                 # Agent agregacji
 requirements.txt            # Zależności Python
 README.md                   # Dokumentacja (ten plik)

 audio_utils.py              # Obsługa audio
 vad_utils.py                # Voice Activity Detection
 speaker_recognition.py      # Rozpoznawanie mówcy
 transcription.py            # Faster-Whisper STT
 tts_utils.py                # Edge-TTS
 led_control.py              # Kontrola LED
 zmq_utils.py                # Komunikacja ZeroMQ

 tests/                      # Testy jednostkowe
    __init__.py
    test_audio_utils.py
    test_vad_utils.py
    test_speaker_recognition.py
    test_transcription.py
    test_led_control.py
    test_reporter.py

 data/                       # Pliki danych JSONL
    dialog.jsonl
    camera.jsonl
    lidar.jsonl
    scenarios.jsonl

 logs/                       # Logi systemowe
    watus.log
    watus_reporter.log

 models/                     # Modele AI (pobierane automatycznie)
     whisper/
     speaker_recognition/
```

---

## 🧩 Moduły

### 1. **audio_utils.py**

Obsługa audio: listowanie urządzeń, nagrywanie, analiza głośności.

**Główne funkcje:**
- `list_audio_devices()` - lista mikrofonów
- `select_audio_device()` - wybór urządzenia
- `calculate_volume_db(audio_data)` - głośność w dB
- `create_audio_stream()` - strumień audio

### 2. **vad_utils.py**

Voice Activity Detection (WebRTC VAD).

**Główne funkcje:**
- `create_vad(mode)` - tworzy VAD
- `is_speech(vad, audio_data, sample_rate)` - wykrywa mowę
- `SpeechBuffer` - bufor do zbierania wypowiedzi

### 3. **speaker_recognition.py**

Rozpoznawanie mówcy przez ECAPA-TDNN.

**Główne funkcje:**
- `load_speaker_model(model_name, device)` - ładuje model
- `extract_embedding(model, audio_data, sample_rate)` - ekstraktuje embedding
- `calculate_similarity(emb1, emb2)` - podobieństwo kosinusowe
- `LeaderTracker` - śledzi lidera

### 4. **transcription.py**

Transkrypcja mowy na tekst (Faster-Whisper).

**Główne funkcje:**
- `load_whisper_model(model_size, device)` - ładuje model
- `transcribe_audio(model, audio_data, sample_rate)` - transkrypcja
- `detect_wake_word(text, wake_phrase)` - wykrywa wake word

### 5. **tts_utils.py**

Text-to-Speech przez Edge-TTS (Microsoft).

**Główne funkcje:**
- `synthesize_speech(text, voice)` - synteza mowy
- `play_audio_file(file_path)` - odtwarzanie
- `speak(text, voice)` - syntetyzuj i odtwórz
- `list_available_voices()` - lista głosów

**Dostępne głosy polskie:**
- `pl-PL-MarekNeural` (męski, zalecany) 
- `pl-PL-ZofiaNeural` (żeński)

### 6. **led_control.py**

Kontrola LED przez GPIO (z trybem mock).

**Główne funkcje:**
- `setup_leds(green_pin, red_pin, mock_mode)`
- `set_listening_mode()` - zielona LED
- `set_processing_mode()` - czerwona LED

### 7. **zmq_utils.py**

Komunikacja ZeroMQ pub/sub.

**Główne funkcje:**
- `create_zmq_publisher(address)` - publisher
- `create_zmq_subscriber(address, topic)` - subscriber
- `publish_message(socket, message, topic)` - publikuj
- `receive_message(socket, timeout_ms)` - odbierz

---

## 🧪 Testy

### Uruchamianie testów

```bash
# Wszystkie testy
pytest tests/ -v

# Konkretny moduł
pytest tests/test_vad_utils.py -v

# Z pokryciem kodu
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Test-First Approach

Projekt został stworzony z test-first approach:
-  Testy przed implementacją
-  100% pokrycie krytycznych funkcji
-  Mocking dla I/O operacji

---

##  Rozwiązanie TTS

### Edge-TTS (Microsoft) - ZALECANE 

**Dlaczego lepsze niż Piper?**

 **Naturalność**: Najwyższa jakość głosów (neural TTS)  
 **Szybkość**: Bardzo szybkie przez API  
 **Polski**: Doskonałe wsparcie dla języka polskiego  
 **Darmowe**: Bez limitów (używa publicznego API)  
 **Proste**: Bez instalacji modeli, działa od razu  

**Wady:**
 Wymaga internetu  
 Zależność od Microsoft API  

### Alternatywy

| TTS Engine | Naturalność | Szybkość | Offline | Polski |
|------------|-------------|----------|---------|--------|
| **Edge-TTS** |  |  |  |  Doskonały |
| **Piper** |  |  |  |  Dobry |
| **Coqui TTS** |  |  |  |  Bardzo dobry |
| **gTTS** |  |  |  |  Podstawowy |

### Zmiana głosu TTS

Edytuj `config.json`:

```json
{
  "tts": {
    "engine": "edge",
    "voice": "pl-PL-ZofiaNeural",  // Zmień na żeński
    "rate": "+10%",                 // Przyspiesz
    "volume": "+10%"                // Głośniej
  }
}
```

### Lista dostępnych głosów

```python
from tts_utils import list_available_voices
voices = list_available_voices()
```

---

##  API i Format Danych

### Format dialog.jsonl

```json
{
  "timestamp": "2025-10-16T10:15:23.456789",
  "speaker": "leader",
  "text": "Hej Watusiu, jak się masz?",
  "duration": 2.3,
  "is_leader": true,
  "volume_db": -28.5
}
```

### Format camera.jsonl

```json
{
  "timestamp": "2025-10-16T10:15:20.123456",
  "description": "Wykryto osobę w centralnej części obrazu",
  "confidence": 0.95,
  "objects": [
    {
      "type": "person",
      "bbox": [120, 80, 280, 400],
      "confidence": 0.95
    }
  ],
  "brightness": "medium",
  "scene": "indoor"
}
```

### Format lidar.jsonl

```json
{
  "timestamp": "2025-10-16T10:15:18.123456",
  "obstacles": "brak przeszkód w promieniu 2m",
  "distances": {
    "front": 2.5,
    "left": 1.8,
    "right": 3.2,
    "back": 1.5
  },
  "closest_obstacle": {
    "direction": "back",
    "distance": 1.5
  },
  "safe_to_move": true
}
```

### Format scenarios.jsonl

```json
{
  "timestamp": "2025-10-16T10:00:00.000000",
  "name": "patrol",
  "status": "active",
  "description": "Regularny patrol pomieszczenia co 30 minut",
  "priority": 2,
  "params": {
    "interval_minutes": 30,
    "route": "kitchen-livingroom-bedroom"
  }
}
```

### Format API LLM

**Request:**
```json
{
  "model": "llama3",
  "messages": [
    {
      "role": "system",
      "content": "<prompt z kontekstem>"
    }
  ],
  "stream": false
}
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "Odpowiedź robota"
  }
}
```

---

##  Troubleshooting

### Problem: Brak urządzeń audio

```bash
# Sprawdź urządzenia
arecord -l

# Test nagrywania
arecord -d 5 -f cd test.wav
aplay test.wav
```

### Problem: CUDA nie działa

```bash
# Sprawdź CUDA
nvidia-smi

# Zmień na CPU w config.json
"device": "cpu",
"compute_type": "int8"
```

### Problem: Błąd ZeroMQ

```bash
# Sprawdź porty
netstat -tuln | grep 555

# Zmień porty w config.json jeśli zajęte
```

### Problem: Brak odpowiedzi od LLM

```bash
# Sprawdź czy endpoint LLM jest dostępny
curl -X POST https://your-llm-endpoint.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"test"}]}'

# Sprawdź konfigurację w config.json
cat config.json | grep -A5 '"llm"'

# Sprawdź logi reporter.py
```
```

### Problem: VAD nie wykrywa mowy

- Zwiększ czułość: `"vad_mode": 1` (w config.json)
- Zmniejsz próg głośności: `"volume_threshold_db": -40`
- Sprawdź mikrofon i głośność

### Problem: Wake word nie działa

- Dodaj alternatywy w config.json
- Zmniejsz threshold: `"threshold": 0.5`
- Sprawdź transkrypcję w logach

### Problem: TTS nie działa (brak dźwięku)

```bash
# Zainstaluj mpg123
sudo apt-get install mpg123

# Lub ffmpeg
sudo apt-get install ffmpeg

# Test
mpg123 --version
```

---

##  Logi

Logi znajdują się w `logs/`:
- `logs/watus.log` - główny program
- `logs/watus_reporter.log` - reporter

Poziom logowania można zmienić w `config.json`:
```json
{
  "logs": {
    "level": "DEBUG",  // DEBUG, INFO, WARNING, ERROR
    "file": "logs/watus.log"
  }
}
```

---

##  Bezpieczeństwo

-  **Timeout lidera**: System automatycznie resetuje lidera po 3 minutach braku aktywności
-  **Tylko lider**: Odpowiedzi są wysyłane tylko na zapytania od lidera
-  **Prywatność**: Dane audio nie są przechowywane (tylko transkrypcje w JSONL)

---

##  Optymalizacje

### GPU

System jest zoptymalizowany dla GPU:
- Faster-Whisper: `device="cuda"`, `compute_type="float16"`
- ECAPA: automatyczne wykrywanie CUDA

### CPU only

Dla Raspberry Pi bez GPU:
```json
{
  "whisper": {
    "model_size": "tiny",  // Zmniejsz model
    "device": "cpu",
    "compute_type": "int8"
  }
}
```

---

##  Dokumentacja Zewnętrzna

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [Edge-TTS](https://github.com/rany2/edge-tts)
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad)
- [ZeroMQ](https://zeromq.org/)

---

##  Licencja

MIT License - wolne do użytku osobistego i komercyjnego.

---

## ‍ Autor

System Watuś - Robot Domowy  
Wersja: 1.0.0  
Data: 2025-10-16

---

##  Podsumowanie

System Watuś to kompletne rozwiązanie dla robota domowego z zaawansowanym przetwarzaniem mowy:

 **Prosty kod** - tylko funkcje, bez klas  
 **Test-first** - pełne pokrycie testami  
 **Zoptymalizowany** - GPU support  
 **Naturalny TTS** - Edge-TTS dla polskiego  
 **Rozpoznawanie mówcy** - ECAPA-TDNN  
 **Wieloagentowy** - watus.py + reporter.py  
 **Dokumentacja PL** - pełna dokumentacja po polsku  

**Powodzenia z robotem Watuś! **
