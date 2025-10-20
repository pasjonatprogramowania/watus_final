#!/usr/bin/env python3
"""
REPORTER.PY - Agent agregujący dane i komunikujący się z LLM.

Funkcjonalność:
- Odbiera dialog.jsonl przez ZeroMQ
- Zbiera dane z innych źródeł (kamera, LiDAR, scenariusze)
- Tworzy wspólny kontekst z timestampem
- Wysyła POST do LLM
- Odbiera odpowiedź i wysyła przez ZeroMQ
"""

import json
import logging
import time
import signal
import sys
import requests
from typing import Any
from datetime import datetime, timedelta
from pathlib import Path

# Importy lokalnych modułów
import zmq_utils


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
        log_level: Poziom logowania
        log_file: Ścieżka do pliku logu
    """
    # Dodaj suffix do pliku logu
    log_path = Path(log_file)
    reporter_log = log_path.parent / f"{log_path.stem}_reporter{log_path.suffix}"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(reporter_log, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


# ============================================================================
# AGREGACJA DANYCH
# ============================================================================

def read_jsonl_file(file_path: str, max_entries: int = 50) -> list[dict[str, Any]]:
    """
    Odczytuje wpisy z pliku JSONL.
    
    Args:
        file_path: Ścieżka do pliku
        max_entries: Maksymalna liczba ostatnich wpisów
    
    Returns:
        Lista wpisów
    """
    entries = []
    
    try:
        if not Path(file_path).exists():
            logging.debug(f"Plik nie istnieje: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Błąd parsowania JSON: {e}")
        
        # Zwróć ostatnie N wpisów
        return entries[-max_entries:] if entries else []
    
    except Exception as e:
        logging.error(f"Błąd odczytu {file_path}: {e}")
        return []


def filter_recent_entries(entries: list[dict[str, Any]], 
                         minutes: int = 10) -> list[dict[str, Any]]:
    """
    Filtruje wpisy z ostatnich N minut.
    
    Args:
        entries: Lista wpisów
        minutes: Liczba minut wstecz
    
    Returns:
        Przefiltrowana lista
    """
    if not entries:
        return []
    
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent = []
    
    for entry in entries:
        try:
            timestamp = datetime.fromisoformat(entry.get('timestamp', ''))
            if timestamp >= cutoff_time:
                recent.append(entry)
        except (ValueError, TypeError):
            continue
    
    return recent


def aggregate_data(config: dict[str, Any]) -> dict[str, Any]:
    """
    Agreguje dane ze wszystkich źródeł.
    
    Args:
        config: Konfiguracja
    
    Returns:
        Zagregowane dane
    """
    data_config = config['data']
    
    # Odczytaj dane z plików
    dialog_entries = read_jsonl_file(data_config['dialog_file'], max_entries=20)
    camera_entries = read_jsonl_file(data_config['camera_file'], max_entries=10)
    lidar_entries = read_jsonl_file(data_config['lidar_file'], max_entries=10)
    scenario_entries = read_jsonl_file(data_config['scenarios_file'], max_entries=5)
    
    # Filtruj ostatnie wpisy (10 minut)
    dialog_recent = filter_recent_entries(dialog_entries, minutes=10)
    camera_recent = filter_recent_entries(camera_entries, minutes=5)
    lidar_recent = filter_recent_entries(lidar_entries, minutes=5)
    
    # Stwórz kontekst
    context = {
        "timestamp": datetime.now().isoformat(),
        "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dialog": dialog_recent,
        "camera": camera_recent,
        "lidar": lidar_recent,
        "scenarios": scenario_entries,
        "summary": {
            "dialog_count": len(dialog_recent),
            "camera_count": len(camera_recent),
            "lidar_count": len(lidar_recent),
            "scenario_count": len(scenario_entries)
        }
    }
    
    logging.info(f"Zagregowano dane: dialog={len(dialog_recent)}, "
                f"camera={len(camera_recent)}, lidar={len(lidar_recent)}, "
                f"scenarios={len(scenario_entries)}")
    
    return context


# ============================================================================
# KOMUNIKACJA Z LLM
# ============================================================================

def prepare_llm_prompt(context: dict[str, Any], user_message: str) -> str:
    """
    Przygotowuje prompt dla LLM.
    
    Args:
        context: Zagregowany kontekst
        user_message: Wiadomość użytkownika
    
    Returns:
        Prompt systemowy
    """
    # Format dialogu
    dialog_text = ""
    for entry in context['dialog'][-5:]:  # Ostatnie 5 wypowiedzi
        speaker = "LIDER" if entry.get('is_leader') else "NIEZNANY"
        dialog_text += f"{speaker}: {entry['text']}\n"
    
    # Format danych z kamery
    camera_text = ""
    if context['camera']:
        last_camera = context['camera'][-1]
        camera_text = f"Kamera: {last_camera.get('description', 'brak danych')}\n"
    
    # Format danych z LiDAR
    lidar_text = ""
    if context['lidar']:
        last_lidar = context['lidar'][-1]
        lidar_text = f"LiDAR: {last_lidar.get('obstacles', 'brak przeszkód')}\n"
    
    # Format scenariuszy
    scenario_text = ""
    if context['scenarios']:
        scenario_text = "Aktywne scenariusze:\n"
        for scenario in context['scenarios']:
            scenario_text += f"- {scenario.get('name', 'unknown')}: {scenario.get('status', 'active')}\n"
    
    # Złóż prompt
    prompt = f"""Jesteś robotem Watuś - pomocnym asystentem domowym.

KONTEKST SYSTEMOWY:
Czas: {context['system_time']}

OSTATNIE ROZMOWY:
{dialog_text if dialog_text else "Brak ostatnich rozmów"}

PERCEPCJA:
{camera_text if camera_text else ""}
{lidar_text if lidar_text else ""}

{scenario_text if scenario_text else ""}

AKTUALNE ZAPYTANIE LIDERA:
{user_message}

Odpowiedz w sposób naturalny, pomocny i zwięzły (max 2-3 zdania).
"""
    
    return prompt


def send_to_llm(context: dict[str, Any], user_message: str, 
               config: dict[str, Any]) -> str | None:
    """
    Wysyła zapytanie do LLM i zwraca odpowiedź.
    
    Args:
        context: Zagregowany kontekst
        user_message: Wiadomość użytkownika
        config: Konfiguracja LLM
    
    Returns:
        Odpowiedź od LLM lub None
    """
    try:
        llm_config = config['llm']
        
        # Przygotuj prompt
        prompt = prepare_llm_prompt(context, user_message)
        
        # Przygotuj payload dla API LLM
        payload = {
            "model": llm_config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            "stream": False
        }
        
        logging.info(f"Wysyłanie zapytania do LLM: {llm_config['endpoint']}")
        
        # Wyślij request
        response = requests.post(
            llm_config['endpoint'],
            json=payload,
            timeout=llm_config['timeout']
        )
        
        response.raise_for_status()
        
        # Parsuj odpowiedź
        result = response.json()
        
        if 'message' in result and 'content' in result['message']:
            answer = result['message']['content'].strip()
            logging.info(f"Otrzymano odpowiedź: {answer[:100]}...")
            return answer
        else:
            logging.error(f"Nieoczekiwany format odpowiedzi: {result}")
            return None
    
    except requests.Timeout:
        logging.error("Timeout połączenia z LLM")
        return "Przepraszam, mam problem z przetwarzaniem. Spróbuj ponownie."
    
    except requests.RequestException as e:
        logging.error(f"Błąd komunikacji z LLM: {e}")
        return "Przepraszam, nie mogę teraz odpowiedzieć."
    
    except Exception as e:
        logging.error(f"Nieoczekiwany błąd LLM: {e}")
        return None


# ============================================================================
# GŁÓWNA PĘTLA
# ============================================================================

def main_loop(config: dict[str, Any], zmq_subscriber, zmq_publisher) -> None:
    """
    Główna pętla agregacji i komunikacji.
    
    Args:
        config: Konfiguracja
        zmq_subscriber: Socket do odbierania dialogów
        zmq_publisher: Socket do wysyłania odpowiedzi
    """
    logging.info("=" * 60)
    logging.info("REPORTER - Agent uruchomiony")
    logging.info("=" * 60)
    
    try:
        while True:
            # Odbierz wiadomość (z timeoutem 1s)
            message = zmq_utils.receive_message(zmq_subscriber, timeout_ms=1000)
            
            if message:
                logging.info(f"Otrzymano dialog: {message.get('text', '')}")
                
                # Sprawdź czy od lidera
                if not message.get('is_leader', False):
                    logging.info("Wiadomość nie od lidera, pomijam")
                    continue
                
                # Agreguj dane
                context = aggregate_data(config)
                
                # Wyślij do LLM
                user_message = message.get('text', '')
                response_text = send_to_llm(context, user_message, config)
                
                if response_text:
                    # Wyślij odpowiedź
                    response = {
                        "timestamp": datetime.now().isoformat(),
                        "text": response_text,
                        "source": "llm"
                    }
                    
                    zmq_utils.publish_message(
                        zmq_publisher, response, topic="response"
                    )
                    logging.info(f"Wysłano odpowiedź: {response_text[:100]}...")
                else:
                    logging.warning("Brak odpowiedzi od LLM")
    
    except KeyboardInterrupt:
        logging.info("\nPrzerwano przez użytkownika")


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
    logging.info("REPORTER.PY - Agent agregacji - Inicjalizacja")
    logging.info("=" * 60)
    
    # Inicjalizuj ZeroMQ
    logging.info("Inicjalizacja ZeroMQ...")
    
    # Subscriber do odbierania dialogów
    zmq_sub_context, zmq_sub_socket = zmq_utils.create_zmq_subscriber(
        config['zmq']['dialog_publisher'],
        topic="dialog"
    )
    
    # Publisher do wysyłania odpowiedzi
    zmq_pub_context, zmq_pub_socket = zmq_utils.create_zmq_publisher(
        config['zmq']['response_subscriber']
    )
    
    # Poczekaj chwilę na inicjalizację
    time.sleep(0.5)
    
    # Obsługa sygnałów
    def signal_handler(sig, frame):
        logging.info("\nOtrzymano sygnał zakończenia")
        zmq_utils.close_zmq_socket(zmq_sub_context, zmq_sub_socket)
        zmq_utils.close_zmq_socket(zmq_pub_context, zmq_pub_socket)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Uruchom główną pętlę
    try:
        main_loop(config, zmq_sub_socket, zmq_pub_socket)
    finally:
        zmq_utils.close_zmq_socket(zmq_sub_context, zmq_sub_socket)
        zmq_utils.close_zmq_socket(zmq_pub_context, zmq_pub_socket)


if __name__ == "__main__":
    main()
