
"""
Moduł transkrypcji mowy na tekst (Faster-Whisper).
"""

from faster_whisper import WhisperModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_whisper_model(model_size: str = "base", device: str = "cuda", 
                      compute_type: str = "float16") -> WhisperModel:
    """
    Ładuje model Faster-Whisper.
    
    Args:
        model_size: Rozmiar modelu (tiny, base, small, medium, large)
        device: Urządzenie (cuda/cpu)
        compute_type: Typ obliczeń (float16, int8, float32)
    
    Returns:
        Model Whisper
    """
    logger.info(f"Ładowanie Whisper: model={model_size}, device={device}")
    
    try:
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root="models/whisper"
        )
        logger.info("Model Whisper załadowany")
        return model
    except Exception as e:
        logger.error(f"Błąd ładowania Whisper: {e}")
        logger.warning("Próba załadowania na CPU...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        return model


def transcribe_audio(model: WhisperModel, audio_data: bytes, 
                     sample_rate: int, language: str = "pl") -> tuple[str, float]:
    """
    Transkrybuje audio na tekst.
    
    Args:
        model: Model Whisper
        audio_data: Surowe dane audio
        sample_rate: Częstotliwość próbkowania
        language: Język (pl)
    
    Returns:
        Krotka (tekst, czas trwania w sekundach)
    """
    try:
        # Konwersja bytes -> numpy array -> float32
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Transkrypcja - disable VAD filter for better transcription of short utterances
        segments, info = model.transcribe(
            audio_float,
            language=language,
            vad_filter=False
        )
        
        # Zbierz tekst
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        text = " ".join(text_parts)
        duration = len(audio_float) / sample_rate
        
        logger.info(f"Transkrypcja: '{text}' ({duration:.2f}s)")
        return text, duration
    
    except Exception as e:
        logger.error(f"Błąd transkrypcji: {e}")
        return "", 0.0


def detect_wake_word(text: str, wake_phrase: str,
                    alternatives: list, threshold: float = 0.7) -> bool:
    """
    Wykrywa słowo wybudzające (wake word).

    Args:
        text: Transkrybowany tekst
        wake_phrase: Główna fraza wybudzająca
        alternatives: Lista alternatywnych fraz
        threshold: Próg podobieństwa (0-1)

    Returns:
        True jeśli wykryto wake word
    """
    text_lower = text.lower().strip()

    # Dokładne dopasowanie
    all_phrases = [wake_phrase] + alternatives
    for phrase in all_phrases:
        if phrase.lower() in text_lower:
            logger.info(f"Wake word wykryte: '{phrase}'")
            return True

    # Częściowe podobieństwo oparte na substringach słów
    for phrase in all_phrases:
        phrase_lower = phrase.lower()
        phrase_words = phrase_lower.split()
        text_words = text_lower.split()

        if len(phrase_words) == 0:
            continue

        # Sprawdzamy czy każde słowo z frazy można znaleźć jako substring w słowach tekstu
        matched_words = 0
        for phrase_word in phrase_words:
            for text_word in text_words:
                # Sprawdzamy czy jedno słowo zawiera się w drugim (w obie strony)
                if phrase_word in text_word or text_word in phrase_word:
                    matched_words += 1
                    break  # Znaleźliśmy dopasowanie dla tego słowa frazy

        if matched_words > 0:
            match_ratio = matched_words / len(phrase_words)
            if match_ratio >= threshold:
                logger.info(f"Wake word wykryte (podobieństwo {match_ratio:.2f}): '{phrase}'")
                return True

        # Alternatywnie: prosta metoda oparta na wspólnych słowach (bez wymagania kolejności)
        phrase_word_set = set(phrase_words)
        text_word_set = set(text_words)
        common_words = phrase_word_set & text_word_set
        if common_words:
            common_ratio = len(common_words) / len(phrase_word_set)
            if common_ratio >= threshold:
                logger.info(f"Wake word wykryte (wspólne słowa {common_ratio:.2f}): '{phrase}'")
                return True

    return False
