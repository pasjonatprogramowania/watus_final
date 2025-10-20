
"""
Testy jednostkowe dla modułu audio_utils.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

import audio_utils


def test_calculate_volume_db():
    """Test obliczania głośności w dB."""
    # Cicha próbka
    quiet_audio = np.zeros(1000, dtype=np.int16).tobytes()
    db = audio_utils.calculate_volume_db(quiet_audio)
    assert db < -50, "Cicha próbka powinna mieć niski dB"
    
    # Głośna próbka
    loud_audio = (np.ones(1000, dtype=np.int16) * 10000).tobytes()
    db = audio_utils.calculate_volume_db(loud_audio)
    assert db > -20, "Głośna próbka powinna mieć wysoki dB"
    
    # Pusta próbka
    empty_audio = b''
    db = audio_utils.calculate_volume_db(empty_audio)
    assert db == -100.0, "Pusta próbka powinna zwrócić -100 dB"


def test_calculate_volume_db_realistic():
    """Test z realistycznymi danymi audio."""
    # Symulacja normalnej mowy (-30 do -20 dB)
    speech_audio = (np.random.randint(-5000, 5000, 1000, dtype=np.int16)).tobytes()
    db = audio_utils.calculate_volume_db(speech_audio)
    assert -50 < db < 0, f"Realistyczna mowa powinna być w zakresie -50 do 0 dB, otrzymano: {db}"


def test_list_audio_devices():
    """Test listowania urządzeń audio."""
    devices = audio_utils.list_audio_devices()
    assert isinstance(devices, list), "Powinno zwrócić listę"
    # Możemy nie mieć urządzeń w środowisku testowym
    if devices:
        idx, name = devices[0]
        assert isinstance(idx, int), "Indeks powinien być liczbą"
        assert isinstance(name, str), "Nazwa powinna być stringiem"


def test_read_audio_chunk_mock():
    """Test odczytu chunk z mockiem."""
    # Ten test wymaga mocka stream, pomijamy w podstawowej wersji
    pass
