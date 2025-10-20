
"""
Moduł Voice Activity Detection (VAD) - wykrywanie aktywności głosu.
"""

import webrtcvad
import collections
import logging

logger = logging.getLogger(__name__)


def create_vad(mode: int = 3) -> webrtcvad.Vad:
    """
    Tworzy obiekt VAD.
    
    Args:
        mode: Tryb agresywności (0-3, 3=najbardziej agresywny)
    
    Returns:
        Obiekt VAD
    """
    vad = webrtcvad.Vad(mode)
    logger.info(f"Utworzono VAD z trybem: {mode}")
    return vad


def is_speech(vad: webrtcvad.Vad, audio_data: bytes,
              sample_rate: int) -> bool:
    """
    Sprawdza czy fragment audio zawiera mowę.

    Args:
        vad: Obiekt VAD
        audio_data: Dane audio
        sample_rate: Częstotliwość próbkowania

    Returns:
        True jeśli wykryto mowę
    """
    try:
        # WebRTC VAD wymaga ramek o długości 10ms, 20ms lub 30ms
        # Oblicz oczekiwaną długość ramki dla danej częstotliwości próbkowania
        bytes_per_sample = 2  # 16-bit audio
        samples_per_30ms = int(sample_rate * 0.03)  # 30ms ramek
        bytes_per_30ms = samples_per_30ms * bytes_per_sample

        # Sprawdź czy dane mają właściwą długość do VAD
        if len(audio_data) == bytes_per_30ms:
            return vad.is_speech(audio_data, sample_rate)
        else:
            # Jeśli długość nie jest zgodna, spróbuj przetworzyć dostępne dane
            total_samples = len(audio_data) // bytes_per_sample
            expected_samples = samples_per_30ms

            if total_samples > 0:
                # Weż tyle co się da, ale zgłoś błąd jeśli długość jest nieodpowiednia
                logger.warning(f"Nieprawidłowa długość ramki VAD: {len(audio_data)} bajtów ({total_samples} próbki), oczekiwano {bytes_per_30ms} bajtów dla {sample_rate}Hz")
                # Spróbuj przetworzyć jeśli przynajmniej jakiś fragment
                # (choć to nie będzie poprawne dla VAD)
                if len(audio_data) >= bytes_per_30ms:
                    return vad.is_speech(audio_data[:bytes_per_30ms], sample_rate)
                else:
                    # Za mało danych, zakładamy brak mowy
                    return False
            else:
                return False

    except Exception as e:
        logger.error(f"Błąd VAD: {e}")
        return False


class SpeechBuffer:
    """
    Bufor do zbierania wypowiedzi z obsługą pre-speech buffer.
    """
    
    def __init__(self, pre_buffer_size: int, silence_threshold_frames: int):
        """
        Args:
            pre_buffer_size: Liczba ramek przed wykryciem mowy do zachowania
            silence_threshold_frames: Liczba ramek ciszy do uznania końca wypowiedzi
        """
        self.pre_buffer_size = pre_buffer_size
        self.silence_threshold_frames = silence_threshold_frames
        self.pre_buffer = collections.deque(maxlen=pre_buffer_size)
        self.speech_buffer = []
        self.silence_counter = 0
        self.is_recording = False
        logger.info(f"SpeechBuffer: pre={pre_buffer_size}, silence={silence_threshold_frames}")
    
    def add_frame(self, audio_frame: bytes, is_speech_frame: bool) -> bytes | None:
        """
        Dodaje ramkę audio do bufora.
        
        Args:
            audio_frame: Ramka audio
            is_speech_frame: Czy ramka zawiera mowę
        
        Returns:
            Kompletna wypowiedź (bytes) lub None jeśli jeszcze trwa
        """
        if not self.is_recording:
            # Czekamy na początek mowy
            self.pre_buffer.append(audio_frame)
            
            if is_speech_frame:
                # Rozpocznij nagrywanie
                self.is_recording = True
                self.speech_buffer = list(self.pre_buffer)
                self.silence_counter = 0
                logger.debug("Rozpoczęto nagrywanie wypowiedzi")
        else:
            # Nagrywamy
            self.speech_buffer.append(audio_frame)
            
            if is_speech_frame:
                self.silence_counter = 0
            else:
                self.silence_counter += 1
            
            # Sprawdź czy koniec wypowiedzi
            if self.silence_counter >= self.silence_threshold_frames:
                complete_speech = b''.join(self.speech_buffer)
                logger.info(f"Zakończono nagrywanie wypowiedzi: {len(complete_speech)} bajtów")
                self.reset()
                return complete_speech
        
        return None
    
    def reset(self) -> None:
        """Resetuje bufor do stanu początkowego."""
        self.speech_buffer = []
        self.silence_counter = 0
        self.is_recording = False
        self.pre_buffer.clear()


def calculate_silence_frames(silence_ms: int, frame_duration_ms: int) -> int:
    """
    Oblicza liczbę ramek odpowiadającą czasowi ciszy.
    
    Args:
        silence_ms: Czas ciszy w milisekundach
        frame_duration_ms: Długość ramki w milisekundach
    
    Returns:
        Liczba ramek
    """
    frames = int(silence_ms / frame_duration_ms)
    logger.debug(f"Obliczono ramki ciszy: {silence_ms}ms -> {frames} ramek")
    return frames


def calculate_pre_buffer_size(buffer_ms: int, frame_duration_ms: int) -> int:
    """
    Oblicza rozmiar pre-buffer w ramkach.
    
    Args:
        buffer_ms: Długość bufora w milisekundach
        frame_duration_ms: Długość ramki w milisekundach
    
    Returns:
        Rozmiar bufora w ramkach
    """
    size = int(buffer_ms / frame_duration_ms)
    logger.debug(f"Obliczono rozmiar pre-buffer: {buffer_ms}ms -> {size} ramek")
    return size
