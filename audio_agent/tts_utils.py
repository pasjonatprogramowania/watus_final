
"""
Moduł Text-to-Speech przy użyciu Edge-TTS (Microsoft).
"""

import edge_tts
import asyncio
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


async def _synthesize_speech_async(text: str, voice: str, rate: str, 
                                   volume: str, output_file: str) -> None:
    """
    Asynchroniczna synteza mowy (wewnętrzna).
    
    Args:
        text: Tekst do syntezowania
        voice: Głos (np. pl-PL-MarekNeural)
        rate: Szybkość (+0%, +10%, -10%)
        volume: Głośność (+0%, +10%, -10%)
        output_file: Ścieżka wyjściowa
    """
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    await communicate.save(output_file)


def synthesize_speech(text: str, voice: str = "pl-PL-MarekNeural", 
                      rate: str = "+0%", volume: str = "+0%",
                      output_file: str | None = None) -> str:
    """
    Syntezuje mowę z tekstu.
    
    Args:
        text: Tekst do syntezowania
        voice: Głos (np. pl-PL-MarekNeural, pl-PL-ZofiaNeural)
        rate: Szybkość mówienia
        volume: Głośność
        output_file: Plik wyjściowy (jeśli None, używa temp)
    
    Returns:
        Ścieżka do pliku audio
    """
    try:
        if output_file is None:
            # Użyj pliku tymczasowego
            fd, output_file = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
        
        logger.info(f"Synteza mowy: '{text[:50]}...'")
        
        # Uruchom async funkcję
        asyncio.run(_synthesize_speech_async(text, voice, rate, volume, output_file))
        
        logger.info(f"Mowa zsyntezowana: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Błąd syntezy mowy: {e}")
        return ""


def play_audio_file(file_path: str) -> None:
    """
    Odgrywa plik audio.
    
    Args:
        file_path: Ścieżka do pliku
    """
    try:
        # Użyj mpg123 lub pygame do odtwarzania
        import subprocess
        subprocess.run(["mpg123", "-q", file_path], check=True)
        logger.info(f"Odtworzono: {file_path}")
    except FileNotFoundError:
        try:
            # Fallback na ffplay
            import subprocess
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-hide_banner", 
                          "-loglevel", "error", file_path], check=True)
            logger.info(f"Odtworzono (ffplay): {file_path}")
        except Exception as e:
            logger.error(f"Błąd odtwarzania (brak mpg123/ffplay): {e}")
    except Exception as e:
        logger.error(f"Błąd odtwarzania: {e}")


def speak(text: str, voice: str = "pl-PL-MarekNeural", 
         rate: str = "+0%", volume: str = "+0%") -> None:
    """
    Syntetyzuje i odgrywa mowę.
    
    Args:
        text: Tekst do powiedzenia
        voice: Głos
        rate: Szybkość
        volume: Głośność
    """
    audio_file = synthesize_speech(text, voice, rate, volume)
    if audio_file:
        play_audio_file(audio_file)
        # Usuń plik tymczasowy
        try:
            os.remove(audio_file)
        except:
            pass


def list_available_voices() -> list:
    """
    Listuje dostępne głosy Edge-TTS.
    
    Returns:
        Lista dostępnych głosów
    """
    try:
        voices = asyncio.run(edge_tts.list_voices())
        polish_voices = [v for v in voices if v['Locale'].startswith('pl')]
        
        logger.info(f"Dostępne polskie głosy: {len(polish_voices)}")
        for voice in polish_voices:
            logger.info(f"  - {voice['ShortName']}: {voice['Gender']}")
        
        return polish_voices
    except Exception as e:
        logger.error(f"Błąd listowania głosów: {e}")
        return []
