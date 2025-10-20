
"""
Moduł rozpoznawania mówcy (Speaker Recognition) przy użyciu Resemblyzer.
"""

import torch
import torchaudio
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import logging
import time
import io

logger = logging.getLogger(__name__)


def load_speaker_model(device: str = "cuda") -> VoiceEncoder:
    """
    Ładuje pretrenowany model Resemblyzer do rozpoznawania mówcy.

    Args:
        device: Urządzenie (cuda/cpu)

    Returns:
        Model VoiceEncoder z Resemblyzer
    """
    logger.info("Ładowanie modelu Resemblyzer")

    try:
        encoder = VoiceEncoder(device=device)
    except Exception as e:
        logger.warning(f"Failed to load VoiceEncoder on {device}: {e}")
        logger.info("Falling back to CPU")
        device = "cpu"
        encoder = VoiceEncoder(device="cpu")

    logger.info(f"Model Resemblyzer załadowany na: {device}")
    return encoder


def extract_embedding(model: VoiceEncoder, audio_data: bytes,
                     sample_rate: int) -> np.ndarray:
    """
    Ekstraktuje embedding z audio.

    Args:
        model: Model VoiceEncoder z Resemblyzer
        audio_data: Surowe dane audio
        sample_rate: Częstotliwość próbkowania

    Returns:
        Wektor embedding (256-dim dla Resemblyzer)
    """
    try:
        # Konwersja bytes -> numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Preprocessing z Resemblyzer (wyrównuje poziom, filtruje itp.)
        processed_wav = preprocess_wav(audio_array, source_sr=sample_rate)

        # Ekstraktuj embedding używając VoiceEncoder
        embedding = model.embed_utterance(processed_wav)

        return embedding.squeeze()

    except Exception as e:
        logger.error(f"Błąd ekstrakcji embeddingu: {e}")
        return np.array([])


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Oblicza podobieństwo kosinusowe między embeddingami.
    
    Args:
        embedding1: Pierwszy embedding
        embedding2: Drugi embedding
    
    Returns:
        Podobieństwo (0-1)
    """
    if embedding1.size == 0 or embedding2.size == 0:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    
    return float(similarity)


class LeaderTracker:
    """
    Śledzi lidera (pierwszego rozpoznanego mówcę).
    """
    
    def __init__(self, timeout_seconds: int = 180, similarity_threshold: float = 0.85):
        """
        Args:
            timeout_seconds: Timeout lidera w sekundach
            similarity_threshold: Próg podobieństwa do uznania za tego samego mówcę
        """
        self.timeout_seconds = timeout_seconds
        self.similarity_threshold = similarity_threshold
        self.leader_embedding: np.ndarray | None = None
        self.last_interaction_time: float | None = None
        logger.info(f"LeaderTracker: timeout={timeout_seconds}s, threshold={similarity_threshold}")
    
    def set_leader(self, embedding: np.ndarray) -> None:
        """
        Ustawia nowego lidera.
        
        Args:
            embedding: Embedding nowego lidera
        """
        self.leader_embedding = embedding
        self.last_interaction_time = time.time()
        logger.info("Ustawiono nowego lidera")
    
    def is_leader(self, embedding: np.ndarray) -> bool:
        """
        Sprawdza czy embedding należy do lidera.
        
        Args:
            embedding: Embedding do sprawdzenia
        
        Returns:
            True jeśli to lider
        """
        if self.leader_embedding is None:
            return False
        
        # Sprawdź timeout
        if self._is_timeout():
            logger.info("Timeout lidera - reset")
            self.reset()
            return False
        
        # Oblicz podobieństwo
        similarity = calculate_similarity(embedding, self.leader_embedding)
        logger.debug(f"Podobieństwo do lidera: {similarity:.3f}")
        
        is_same = similarity >= self.similarity_threshold
        
        if is_same:
            # Odśwież czas interakcji
            self.last_interaction_time = time.time()
        
        return is_same
    
    def _is_timeout(self) -> bool:
        """Sprawdza czy upłynął timeout."""
        if self.last_interaction_time is None:
            return False
        
        elapsed = time.time() - self.last_interaction_time
        return elapsed > self.timeout_seconds
    
    def has_leader(self) -> bool:
        """Sprawdza czy jest ustawiony lider."""
        if self.leader_embedding is None:
            return False
        
        if self._is_timeout():
            self.reset()
            return False
        
        return True
    
    def reset(self) -> None:
        """Resetuje lidera."""
        self.leader_embedding = None
        self.last_interaction_time = None
        logger.info("Reset lidera")
