
"""
Moduł do obsługi audio: nagrywanie, analiza głośności, wykrywanie urządzeń.
"""

import os
import sys
import warnings
import pyaudio
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Wycisz ostrzeżenia ALSA/JACK (normalne w Linux)
os.environ['PYAUDIO_ALSA_ERRORS'] = '0'
warnings.filterwarnings('ignore', category=RuntimeWarning)


def list_audio_devices() -> list[tuple[int, str]]:
    """
    Wyświetla listę dostępnych urządzeń audio.
    Ignoruje ostrzeżenia ALSA/JACK które są normalne w systemach Linux.
    
    Returns:
        Lista krotek (indeks, nazwa urządzenia)
    """
    # Tymczasowo przekieruj stderr aby wyciszyć ostrzeżenia ALSA
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        audio = pyaudio.PyAudio()
        devices = []
        
        # Przywróć stderr
        sys.stderr.close()
        sys.stderr = original_stderr
        
        print("\nDostępne urządzenia audio (wejściowe):")
        print("-" * 70)
        
        for i in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    name = info['name']
                    channels = info['maxInputChannels']
                    sample_rate = int(info['defaultSampleRate'])
                    
                    devices.append((i, name))
                    
                    # Wyświetl czytelną informację
                    print(f"  [{i:2d}] {name}")
                    print(f"       Kanały: {channels}, Częstotliwość: {sample_rate} Hz")
                    
                    logger.info(f"Urządzenie {i}: {name} ({channels}ch, {sample_rate}Hz)")
            except Exception as e:
                logger.debug(f"Błąd odczytu urządzenia {i}: {e}")
                continue
        
        print("-" * 70)
        print(f"Znaleziono {len(devices)} urządzeń wejściowych\n")
        
        audio.terminate()
        
        return devices
        
    except Exception as e:
        # Upewnij się że stderr jest przywrócony
        if sys.stderr != original_stderr:
            sys.stderr.close()
            sys.stderr = original_stderr
        
        logger.error(f"Błąd listowania urządzeń audio: {e}")
        return []


def select_audio_device() -> int:
    """
    Automatycznie wybiera pierwsze dostępne urządzenie audio z priorytetem dla pulse/default.

    Returns:
        Indeks wybranego urządzenia
    """
    devices = list_audio_devices()

    if not devices:
        logger.error("Brak dostępnych urządzeń audio!")
        raise RuntimeError("No audio input devices found")

    # Automatyczny wybór - preferuj urządzenia z pulse/default
    preferred_order = ["pulse", "default"]

    for pref in preferred_order:
        for idx, name in devices:
            if pref.lower() in name.lower():
                logger.info(f"Automatycznie wybrano urządzenie: {idx} - {name}")
                return idx

    # Jeśli nie znaleziono preferowanych, wybierz pierwsze dostępne
    idx, name = devices[0]
    logger.info(f"Wybrano pierwsze dostępne urządzenie: {idx} - {name}")
    return idx


def calculate_volume_db(audio_data: bytes, sample_width: int = 2) -> float:
    """
    Oblicza głośność w decybelach (dB).
    
    Args:
        audio_data: Surowe dane audio
        sample_width: Szerokość próbki w bajtach (2 dla 16-bit)
    
    Returns:
        Głośność w dB
    """
    if not audio_data:
        return -100.0
    
    # Konwersja na numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Oblicz RMS (Root Mean Square)
    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    
    # Zapobiegnij log(0)
    if rms < 1e-10:
        return -100.0
    
    # Konwersja na dB (referencja: maksymalna wartość 16-bit)
    db = 20 * np.log10(rms / 32768.0)
    
    return float(db)


def create_audio_stream(device_index: int, sample_rate: int, 
                       chunk_size: int) -> tuple[pyaudio.PyAudio, pyaudio.Stream]:
    """
    Tworzy strumień audio do nagrywania.
    
    Args:
        device_index: Indeks urządzenia audio
        sample_rate: Częstotliwość próbkowania (Hz)
        chunk_size: Rozmiar bufora
    
    Returns:
        Krotka (PyAudio object, Stream object)
    """
    audio = pyaudio.PyAudio()
    
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size
    )
    
    logger.info(f"Utworzono strumień audio: {sample_rate}Hz, chunk={chunk_size}")
    return audio, stream


def read_audio_chunk(stream: pyaudio.Stream, chunk_size: int) -> bytes:
    """
    Odczytuje kawałek danych audio ze strumienia.
    
    Args:
        stream: Strumień audio
        chunk_size: Rozmiar bufora
    
    Returns:
        Surowe dane audio
    """
    try:
        return stream.read(chunk_size, exception_on_overflow=False)
    except Exception as e:
        logger.error(f"Błąd odczytu audio: {e}")
        return b''


def get_device_sample_rate(device_index: int) -> int:
    """
    Sprawdza domyślną częstotliwość urządzenia audio.

    Args:
        device_index: Indeks urządzenia audio

    Returns:
        Domyślna częstotliwość urządzenia
    """
    try:
        audio = pyaudio.PyAudio()
        info = audio.get_device_info_by_index(device_index)
        default_rate = int(info['defaultSampleRate'])
        audio.terminate()
        return default_rate
    except Exception as e:
        logger.error(f"Błąd sprawdzania częstotliwości urządzenia: {e}")
        return 44100  # Typowa domyślna wartość


def get_compatible_sample_rate(device_index: int) -> int:
    """
    Sprawdza domyślną częstotliwość urządzenia audio i proponuje kompatybilną.

    Args:
        device_index: Indeks urządzenia audio

    Returns:
        Kompatybilna częstotliwość próbkowania
    """
    try:
        audio = pyaudio.PyAudio()
        info = audio.get_device_info_by_index(device_index)
        default_rate = int(info['defaultSampleRate'])
        audio.terminate()

        logger.info(f"Urządzenie {device_index} domyślna częstotliwość: {default_rate}Hz")

        # Preferowane częstotliwości dla VAD i modeli
        compatible_rates = [16000, 44100, 48000]

        # Znajdź najbliższą kompatybilną częstotliwość
        for rate in compatible_rates:
            if abs(default_rate - rate) < 1000:  # Tolerancja 1000Hz
                logger.info(f"Użyję kompatybilnej częstotliwości: {rate}Hz")
                return rate

        # Fallback do pierwszej kompatybilnej
        logger.warning(f"Domyślna częstotliwość {default_rate}Hz niekompatybilna, używam {compatible_rates[0]}Hz")
        return compatible_rates[0]

    except Exception as e:
        logger.error(f"Błąd sprawdzania częstotliwości urządzenia: {e}")
        return 16000  # Bezpieczna domyślna wartość


def adjust_config_for_vad_compatibility(config: dict, device_index: int) -> dict:
    """
    Dostosowuje konfigurację audio z priorytetem dla kompatybilności VAD (16kHz).

    Args:
        config: Oryginalna konfiguracja
        device_index: Indeks urządzenia audio

    Returns:
        Dostosowana konfiguracja
    """
    try:
        device_rate = get_device_sample_rate(device_index)
        original_rate = config['audio']['sample_rate']

        # VAD wymaga dokładnie 16kHz, 32kHz lub 48kHz
        vad_compatible_rates = [16000, 32000, 48000]
        device_compatible_rates = [16000, 22050, 44100, 48000]  # Wspólne dla urządzeń

        # Najlepsza kolejność priorytetów:
        # 1. Zachowaj oryginalną konfigurację jeśli jest kompatybilna z VAD
        # 2. Wybierz 16kHz jeśli urządzenie obsługuje
        # 3. Znajdź najbliższą kompatybilną z VAD

        if original_rate in vad_compatible_rates:
            # Oryginalna konfiguracja jest OK dla VAD
            chosen_rate = original_rate
        elif device_rate in [44100, 48000]:
            # Urządzenie preferuje wyższe częstotliwości, ale dostosuj do VAD
            if 48000 in vad_compatible_rates and abs(device_rate - 48000) <= 2000:
                chosen_rate = 48000
            elif 44100 in device_compatible_rates:
                chosen_rate = 44100
            else:
                chosen_rate = 16000  # Bezpieczna wartość
        else:
            # Sprawdź czy urządzenie obsługuje 16kHz
            if device_rate >= 16000:  # Urządzenie może produkować co najmniej 16kHz
                chosen_rate = 16000
            else:
                chosen_rate = device_rate  # Użyj tego co urządzenie obsługuje

        if chosen_rate != original_rate:
            logger.warning(f"Dostosowano sample_rate z {original_rate}Hz do {chosen_rate}Hz dla urządzenia {device_index} (kompatybilność VAD)")
            config['audio']['sample_rate'] = chosen_rate

    except Exception as e:
        logger.error(f"Błąd dostosowywania konfiguracji VAD: {e}")

    return config


def adjust_config_for_device(config: dict, device_index: int) -> dict:
    """
    Dostosowuje konfigurację audio do możliwości urządzenia.

    Args:
        config: Oryginalna konfiguracja
        device_index: Indeks urządzenia audio

    Returns:
        Dostosowana konfiguracja
    """
    try:
        compatible_rate = get_compatible_sample_rate(device_index)
        original_rate = config['audio']['sample_rate']

        if compatible_rate != original_rate:
            logger.warning(f"Dostosowano sample_rate z {original_rate}Hz do {compatible_rate}Hz dla urządzenia {device_index}")
            config['audio']['sample_rate'] = compatible_rate

    except Exception as e:
        logger.error(f"Błąd dostosowywania konfiguracji: {e}")

    return config


def close_audio_stream(audio: pyaudio.PyAudio, stream: pyaudio.Stream) -> None:
    """
    Zamyka strumień audio i zwalnia zasoby.

    Args:
        audio: Obiekt PyAudio
        stream: Strumień audio
    """
    try:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        logger.info("Zamknięto strumień audio")
    except Exception as e:
        logger.error(f"Błąd zamykania strumienia: {e}")
