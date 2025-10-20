
"""
Testy jednostkowe dla modułu transcription.
"""

import pytest
import sys
sys.path.insert(0, '..')

import transcription


def test_detect_wake_word_exact_match():
    """Test wykrywania wake word - dokładne dopasowanie."""
    result = transcription.detect_wake_word(
        text="Hej Watusiu, jak się masz?",
        wake_phrase="hej watusiu",
        alternatives=["hej watuś"],
        threshold=0.7
    )
    assert result == True, "Powinno wykryć wake word"


def test_detect_wake_word_alternative():
    """Test wykrywania wake word - alternatywna fraza."""
    result = transcription.detect_wake_word(
        text="Hej Watuś, pomóż mi",
        wake_phrase="hej watusiu",
        alternatives=["hej watuś", "ej watusiu"],
        threshold=0.7
    )
    assert result == True, "Powinno wykryć alternatywny wake word"


def test_detect_wake_word_no_match():
    """Test wykrywania wake word - brak dopasowania."""
    result = transcription.detect_wake_word(
        text="To jest zwykła rozmowa",
        wake_phrase="hej watusiu",
        alternatives=["hej watuś"],
        threshold=0.7
    )
    assert result == False, "Nie powinno wykryć wake word"


def test_detect_wake_word_partial_match():
    """Test wykrywania wake word - częściowe dopasowanie."""
    result = transcription.detect_wake_word(
        text="Hej Watus pomóż",
        wake_phrase="hej watusiu",
        alternatives=["hej watuś"],
        threshold=0.7
    )
    # Zależy od progu - "hej watus" vs "hej watusiu" = 2/2 wspólnych słów
    assert result == True, "Powinno wykryć przez podobieństwo"


def test_detect_wake_word_case_insensitive():
    """Test wykrywania wake word - case insensitive."""
    result = transcription.detect_wake_word(
        text="HEJ WATUSIU JAK TAM",
        wake_phrase="hej watusiu",
        alternatives=[],
        threshold=0.7
    )
    assert result == True, "Powinno być case insensitive"
