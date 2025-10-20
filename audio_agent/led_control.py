
"""
Moduł kontroli LED (GPIO).
"""

import logging

logger = logging.getLogger(__name__)

# Mock GPIO dla systemów bez GPIO
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO niedostępne - tryb mock")


def setup_leds(green_pin: int, red_pin: int, mock_mode: bool = False) -> None:
    """
    Inicjalizuje piny LED.
    
    Args:
        green_pin: Numer pinu zielonej LED
        red_pin: Numer pinu czerwonej LED
        mock_mode: Tryb mock (bez prawdziwego GPIO)
    """
    if mock_mode or not GPIO_AVAILABLE:
        logger.info(f"LED mock mode: green={green_pin}, red={red_pin}")
        return
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(green_pin, GPIO.OUT)
    GPIO.setup(red_pin, GPIO.OUT)
    logger.info(f"LED setup: green={green_pin}, red={red_pin}")


def set_led_state(pin: int, state: bool, mock_mode: bool = False) -> None:
    """
    Ustawia stan LED.
    
    Args:
        pin: Numer pinu
        state: True = włączona, False = wyłączona
        mock_mode: Tryb mock
    """
    if mock_mode or not GPIO_AVAILABLE:
        logger.debug(f"LED mock: pin={pin}, state={'ON' if state else 'OFF'}")
        return
    
    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)


def set_listening_mode(green_pin: int, red_pin: int, mock_mode: bool = False) -> None:
    """
    Ustawia tryb nasłuchiwania (zielona LED).
    
    Args:
        green_pin: Pin zielonej LED
        red_pin: Pin czerwonej LED
        mock_mode: Tryb mock
    """
    set_led_state(green_pin, True, mock_mode)
    set_led_state(red_pin, False, mock_mode)
    logger.debug("Tryb: nasłuchiwanie (zielona LED)")


def set_processing_mode(green_pin: int, red_pin: int, mock_mode: bool = False) -> None:
    """
    Ustawia tryb przetwarzania (czerwona LED).
    
    Args:
        green_pin: Pin zielonej LED
        red_pin: Pin czerwonej LED
        mock_mode: Tryb mock
    """
    set_led_state(green_pin, False, mock_mode)
    set_led_state(red_pin, True, mock_mode)
    logger.debug("Tryb: przetwarzanie (czerwona LED)")


def cleanup_leds(mock_mode: bool = False) -> None:
    """
    Czyści GPIO.
    
    Args:
        mock_mode: Tryb mock
    """
    if mock_mode or not GPIO_AVAILABLE:
        logger.info("LED cleanup (mock)")
        return
    
    GPIO.cleanup()
    logger.info("LED cleanup")
