#!/usr/bin/env python3
"""
WATUS.PY - Główny program robota Watuś do przetwarzania mowy.

Funkcjonalność:
- Nasłuchiwanie ciągłe z VAD
- Wykrywanie wywołania "hej watusiu"
- Rozpoznawanie lidera przez ECAPA
- Transkrypcja przez Faster-Whisper
- Kontrola LED (zielony=nasłuch, czerwony=przetwarzanie)
- Half-duplex (albo słucha albo mówi)
- Komunikacja przez ZeroMQ
"""

import json
import logging
import time
import signal
import sys
from typing import Any
from datetime import datetime

# Importy lokalnych modułów
import audio_utils
import vad_utils
import speaker_recognition
import led_control
import zmq_utils
import transcription
import tts_utils


# ============================================================================
# KONFIGURACJA I INICJALIZACJA
# ============================================================================

def load_config(config_path: str = "config.json") -> dict[str, Any]:
    """
    Ładuje konfigurację z pliku JSON.
    
    Args:
        config_path: Ścieżka do pliku konfiguracji
    
    Returns:
        Słownik konfiguracji
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Załadowano konfigurację z {config_path}")
        return config
    except Exception as e:
        logging.error(f"Błąd ładowania konfiguracji: {e}")
        sys.exit(1)


def setup_logging(log_level: str, log_file: str) -> None:
    """
    Konfiguruje system logowania.
    
    Args:
        log_level: Poziom logowania (DEBUG, INFO, WARNING, ERROR)
        log_file: Ścieżka do pliku logu
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def save_dialog_entry(dialog_file: str, entry: dict[str, Any]) -> None:
    """
    Zapisuje wpis do pliku dialog.jsonl.
    
    Args:
        dialog_file: Ścieżka do pliku
        entry: Wpis do zapisania
    """
    try:
        with open(dialog_file, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
        logging.debug(f"Zapisano wpis dialogu: {entry['speaker']}")
    except Exception as e:
        logging.error(f"Błąd zapisu dialogu: {e}")


# ============================================================================
# GŁÓWNA PĘTLA PRZETWARZANIA
# ============================================================================

def process_speech(audio_data: bytes, config: dict[str, Any], 
                  whisper_model, speaker_model, leader_tracker) -> dict[str, Any] | None:
    """
    Przetwarza nagraną wypowiedź.
    
    Args:
        audio_data: Surowe dane audio
        config: Konfiguracja
        whisper_model: Model Whisper
        speaker_model: Model ECAPA
        leader_tracker: Tracker lidera
    
    Returns:
        Słownik z danymi wypowiedzi lub None
    """
    sample_rate = config['audio']['sample_rate']
    
    # 1. Oblicz głośność
    volume_db = audio_utils.calculate_volume_db(audio_data)
    logging.info(f"Głośność: {volume_db:.1f} dB")
    
    # Sprawdź próg głośności
    if volume_db < config['audio']['volume_threshold_db']:
        logging.info("Zbyt cicha wypowiedź, pomijam")
        return None
    
    # 2. Transkrypcja
    text, duration = transcription.transcribe_audio(
        whisper_model, audio_data, sample_rate, 
        config['whisper']['language']
    )
    
    if not text:
        logging.warning("Brak tekstu z transkrypcji")
        return None
    
    # 3. Wykryj wake word
    is_wake_word = transcription.detect_wake_word(
        text,
        config['wake_word']['phrase'],
        config['wake_word']['alternatives'],
        config['wake_word']['threshold']
    )
    
    if not is_wake_word:
        logging.info("Brak wake word, pomijam")
        return None
    
    # 4. Rozpoznawanie mówcy
    embedding = speaker_recognition.extract_embedding(
        speaker_model, audio_data, sample_rate
    )
    
    if embedding.size == 0:
        logging.error("Nie udało się wyekstraktować embeddingu")
        return None
    
    # 5. Sprawdź czy to lider
    is_leader = False
    speaker = "unknown"
    
    if not leader_tracker.has_leader():
        # Ustaw nowego lidera
        leader_tracker.set_leader(embedding)
        is_leader = True
        speaker = "leader"
        logging.info("Nowy lider rozpoznany")
    else:
        # Sprawdź czy to lider
        if leader_tracker.is_leader(embedding):
            is_leader = True
            speaker = "leader"
            logging.info("Lider rozpoznany")
        else:
            logging.info("Nie-lider wykryty")
    
    # 6. Przygotuj wpis
    entry = {
        "timestamp": datetime.now().isoformat(),
        "speaker": speaker,
        "text": text,
        "duration": round(duration, 2),
        "is_leader": is_leader,
        "volume_db": round(volume_db, 1)
    }
    
    return entry


def handle_response(response_text: str, config: dict[str, Any]) -> None:
    """
    Obsługuje odpowiedź od LLM i odgrywa ją przez TTS.
    
    Args:
        response_text: Tekst odpowiedzi
        config: Konfiguracja
    """
    if not response_text:
        return
    
    logging.info(f"Odpowiedź: {response_text}")
    
    # Odtwórz przez TTS
    tts_config = config['tts']
    tts_utils.speak(
        response_text,
        voice=tts_config['voice'],
        rate=tts_config['rate'],
        volume=tts_config['volume']
    )


def main_loop(config: dict[str, Any], audio_device: int, 
             whisper_model, speaker_model, leader_tracker,
             zmq_publisher, zmq_subscriber) -> None:
    """
    Główna pętla nasłuchiwania i przetwarzania.
    
    Args:
        config: Konfiguracja
        audio_device: Indeks urządzenia audio
        whisper_model: Model Whisper
        speaker_model: Model ECAPA
        leader_tracker: Tracker lidera
        zmq_publisher: Socket ZeroMQ do publikacji
        zmq_subscriber: Socket ZeroMQ do subskrypcji
    """
    # Parametry audio
    sample_rate = config['audio']['sample_rate']
    chunk_duration_ms = config['audio']['chunk_duration_ms']
    chunk_size = int(sample_rate * chunk_duration_ms / 1000)
    
    # Parametry VAD
    vad = vad_utils.create_vad(config['audio']['vad_mode'])
    
    silence_frames = vad_utils.calculate_silence_frames(
        config['audio']['silence_threshold_ms'],
        chunk_duration_ms
    )
    
    pre_buffer_size = vad_utils.calculate_pre_buffer_size(
        config['audio']['pre_speech_buffer_ms'],
        chunk_duration_ms
    )
    
    speech_buffer = vad_utils.SpeechBuffer(pre_buffer_size, silence_frames)
    
    # Parametry LED
    green_pin = config['led']['green_pin']
    red_pin = config['led']['red_pin']
    mock_mode = config['led']['mock_mode']
    
    # Otwórz strumień audio
    audio, stream = audio_utils.create_audio_stream(
        audio_device, sample_rate, chunk_size
    )
    
    logging.info("=" * 60)
    logging.info("WATUŚ - System nasłuchiwania uruchomiony")
    logging.info("Nasłuchiwanie słowa wybudzającego 'hej watusiu'...")
    logging.info("=" * 60)
    
    try:
        while True:
            # Tryb nasłuchiwania (zielona LED)
            led_control.set_listening_mode(green_pin, red_pin, mock_mode)
            
            # Odczytaj chunk audio
            audio_chunk = audio_utils.read_audio_chunk(stream, chunk_size)
            
            if not audio_chunk:
                continue
            
            # VAD - wykryj mowę
            is_speech_detected = vad_utils.is_speech(vad, audio_chunk, sample_rate)
            
            # Dodaj do bufora
            complete_speech = speech_buffer.add_frame(audio_chunk, is_speech_detected)
            
            if complete_speech:
                # Tryb przetwarzania (czerwona LED)
                led_control.set_processing_mode(green_pin, red_pin, mock_mode)
                
                logging.info("Przetwarzanie wypowiedzi...")
                
                # Przetwórz wypowiedź
                entry = process_speech(
                    complete_speech, config, 
                    whisper_model, speaker_model, leader_tracker
                )
                
                if entry and entry['is_leader']:
                    # Zapisz do pliku
                    save_dialog_entry(config['data']['dialog_file'], entry)
                    
                    # Wyślij przez ZeroMQ
                    zmq_utils.publish_message(zmq_publisher, entry, topic="dialog")
                    
                    # Czekaj na odpowiedź (z timeoutem)
                    logging.info("Oczekiwanie na odpowiedź...")
                    response = zmq_utils.receive_message(zmq_subscriber, timeout_ms=5000)
                    
                    if response and 'text' in response:
                        handle_response(response['text'], config)
                    else:
                        logging.warning("Brak odpowiedzi od LLM")
    
    except KeyboardInterrupt:
        logging.info("\nPrzerwano przez użytkownika")
    
    finally:
        # Cleanup
        audio_utils.close_audio_stream(audio, stream)
        led_control.cleanup_leds(mock_mode)
        logging.info("Zamknięto system")


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def main():
    """Główna funkcja programu."""
    
    # Załaduj konfigurację
    config = load_config()
    
    # Ustaw logowanie
    setup_logging(config['logs']['level'], config['logs']['file'])
    
    logging.info("=" * 60)
    logging.info("WATUS.PY - Robot Watuś - Inicjalizacja")
    logging.info("=" * 60)
    
    # Wybierz urządzenie audio
    audio_device = audio_utils.select_audio_device()

    # Sprawdź częstotliwość urządzenia i dostosuj jeśli trzeba (priorytet dla kompatybilności VAD)
    config = audio_utils.adjust_config_for_vad_compatibility(config, audio_device)
    
    # Inicjalizuj LED
    led_control.setup_leds(
        config['led']['green_pin'],
        config['led']['red_pin'],
        config['led']['mock_mode']
    )
    
    # Załaduj modele
    logging.info("Ładowanie modeli AI...")
    
    whisper_model = transcription.load_whisper_model(
        config['whisper']['model_size'],
        config['whisper']['device'],
        config['whisper']['compute_type']
    )
    
    speaker_model = speaker_recognition.load_speaker_model(
        device=config['whisper']['device']
    )
    
    # Inicjalizuj leader tracker
    leader_tracker = speaker_recognition.LeaderTracker(
        config['speaker_recognition']['leader_timeout_seconds'],
        config['speaker_recognition']['threshold']
    )
    
    # Inicjalizuj ZeroMQ
    logging.info("Inicjalizacja ZeroMQ...")
    
    zmq_pub_context, zmq_pub_socket = zmq_utils.create_zmq_publisher(
        config['zmq']['dialog_publisher']
    )
    
    zmq_sub_context, zmq_sub_socket = zmq_utils.create_zmq_subscriber(
        config['zmq']['response_subscriber'],
        topic="response"
    )
    
    # Obsługa sygnałów
    def signal_handler(sig, frame):
        logging.info("\nOtrzymano sygnał zakończenia")
        zmq_utils.close_zmq_socket(zmq_pub_context, zmq_pub_socket)
        zmq_utils.close_zmq_socket(zmq_sub_context, zmq_sub_socket)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Uruchom główną pętlę
    try:
        main_loop(
            config, audio_device,
            whisper_model, speaker_model, leader_tracker,
            zmq_pub_socket, zmq_sub_socket
        )
    finally:
        zmq_utils.close_zmq_socket(zmq_pub_context, zmq_pub_socket)
        zmq_utils.close_zmq_socket(zmq_sub_context, zmq_sub_socket)


if __name__ == "__main__":
    main()
