
"""
Testy jednostkowe dla modułu led_control.
"""

import pytest
import sys
sys.path.insert(0, '..')

import led_control


def test_setup_leds_mock():
    """Test inicjalizacji LED w trybie mock."""
    # Nie powinno rzucić wyjątku
    led_control.setup_leds(green_pin=17, red_pin=27, mock_mode=True)


def test_set_led_state_mock():
    """Test ustawiania stanu LED w trybie mock."""
    # Nie powinno rzucić wyjątku
    led_control.set_led_state(pin=17, state=True, mock_mode=True)
    led_control.set_led_state(pin=17, state=False, mock_mode=True)


def test_set_listening_mode_mock():
    """Test trybu nasłuchiwania w mock."""
    led_control.set_listening_mode(green_pin=17, red_pin=27, mock_mode=True)


def test_set_processing_mode_mock():
    """Test trybu przetwarzania w mock."""
    led_control.set_processing_mode(green_pin=17, red_pin=27, mock_mode=True)


def test_cleanup_leds_mock():
    """Test cleanup LED w mock."""
    led_control.cleanup_leds(mock_mode=True)
