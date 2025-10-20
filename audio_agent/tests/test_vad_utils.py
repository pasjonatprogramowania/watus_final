
"""
Testy jednostkowe dla modułu vad_utils.
"""

import pytest
import sys
sys.path.insert(0, '..')

import vad_utils


def test_create_vad():
    """Test tworzenia obiektu VAD."""
    vad = vad_utils.create_vad(mode=3)
    assert vad is not None, "VAD powinien zostać utworzony"


def test_calculate_silence_frames():
    """Test obliczania liczby ramek ciszy."""
    frames = vad_utils.calculate_silence_frames(silence_ms=430, frame_duration_ms=30)
    expected = int(430 / 30)
    assert frames == expected, f"Powinno być {expected} ramek, otrzymano {frames}"
    
    frames = vad_utils.calculate_silence_frames(silence_ms=1000, frame_duration_ms=20)
    assert frames == 50, "1000ms / 20ms = 50 ramek"


def test_calculate_pre_buffer_size():
    """Test obliczania rozmiaru pre-buffer."""
    size = vad_utils.calculate_pre_buffer_size(buffer_ms=300, frame_duration_ms=30)
    expected = int(300 / 30)
    assert size == expected, f"Powinno być {expected}, otrzymano {size}"


def test_speech_buffer_initialization():
    """Test inicjalizacji bufora mowy."""
    buffer = vad_utils.SpeechBuffer(pre_buffer_size=10, silence_threshold_frames=15)
    assert buffer.pre_buffer_size == 10
    assert buffer.silence_threshold_frames == 15
    assert buffer.is_recording == False
    assert buffer.silence_counter == 0


def test_speech_buffer_recording():
    """Test nagrywania wypowiedzi."""
    buffer = vad_utils.SpeechBuffer(pre_buffer_size=3, silence_threshold_frames=2)
    
    # Dodaj kilka ramek ciszy (pre-buffer)
    result = buffer.add_frame(b'silence1', False)
    assert result is None, "Nie powinno jeszcze zwrócić wypowiedzi"
    
    result = buffer.add_frame(b'silence2', False)
    assert result is None
    
    # Wykryj mowę - rozpoczęcie nagrywania
    result = buffer.add_frame(b'speech1', True)
    assert result is None
    assert buffer.is_recording == True, "Nagrywanie powinno się rozpocząć"
    
    # Dodaj więcej mowy
    result = buffer.add_frame(b'speech2', True)
    assert result is None
    
    # Dodaj ciszę (nie wystarczająco długą)
    result = buffer.add_frame(b'silence3', False)
    assert result is None
    
    # Dodaj drugą ramkę ciszy - koniec wypowiedzi
    result = buffer.add_frame(b'silence4', False)
    assert result is not None, "Powinno zwrócić kompletną wypowiedź"
    assert isinstance(result, bytes), "Powinno zwrócić bytes"
    assert buffer.is_recording == False, "Nagrywanie powinno się zakończyć"


def test_speech_buffer_reset():
    """Test resetowania bufora."""
    buffer = vad_utils.SpeechBuffer(pre_buffer_size=3, silence_threshold_frames=2)
    buffer.is_recording = True
    buffer.speech_buffer = [b'test']
    buffer.silence_counter = 5
    
    buffer.reset()
    
    assert buffer.is_recording == False
    assert len(buffer.speech_buffer) == 0
    assert buffer.silence_counter == 0
    assert len(buffer.pre_buffer) == 0
