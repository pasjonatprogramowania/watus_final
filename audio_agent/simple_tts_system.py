#!/usr/bin/env python3
"""
Simple TTS (Speech-to-Text) System with Speaker Recognition and Continuous Monitoring.

This system:
- Continuously listens for speech using VAD
- Identifies if the speaker is the "leader" (previously recognized speaker)
- Transcribes what they said using Whisper
- Responds with TTS (half-duplex: either listening or speaking)
- Prints the speaker identity and transcription only when not speaking
- Sends transcription to API endpoint asynchronously
"""

import json
import logging
import time
from typing import Any
import sys
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Import local modules
import audio_utils
import speaker_recognition
import transcription
import vad_utils
import tts_utils

# Thread executor for async requests
executor = ThreadPoolExecutor(max_workers=5)


def load_config(config_path: str = "config.json") -> dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        sys.exit(1)


def setup_logging() -> None:
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


async def send_transcription_to_api(text: str, speaker: str, endpoint_url: str) -> None:
    """
    Send transcription to API webhook endpoint asynchronously.

    Args:
        text: Transcribed text
        speaker: Speaker identifier (leader/other_person)
        endpoint_url: API endpoint URL (webhook)
    """
    payload = {
        "content": text
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint_url, json=payload,headers=headers, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    output = result.get("output", "No response")
                    logging.info(f" API Response: {output}")
                    print(f"\n AI Response: {output}\n")
                else:
                    logging.warning(f"API returned status {response.status}")
    except asyncio.TimeoutError:
        logging.error(f"Timeout sending to API: {endpoint_url}")
    except Exception as e:
        logging.error(f"Error sending to API: {e}")


def send_transcription_sync(text: str, speaker: str, endpoint_url: str) -> None:
    """
    Wrapper to send transcription in a non-blocking way.

    Args:
        text: Transcribed text
        speaker: Speaker identifier
        endpoint_url: API endpoint URL
    """
    if not endpoint_url:
        logging.warning("No API endpoint configured, skipping POST request")
        return

    # Run async function in executor to not block main thread
    asyncio.run(send_transcription_to_api(text, speaker, endpoint_url))


def record_audio_for_duration(duration_seconds: int, config: dict[str, Any]) -> bytes:
    """
    Record audio for a fixed duration.

    Args:
        duration_seconds: How long to record (in seconds)
        config: Configuration dictionary

    Returns:
        Raw audio data as bytes
    """
    sample_rate = config['audio']['sample_rate']
    chunk_duration_ms = config['audio']['chunk_duration_ms']
    chunk_size = int(sample_rate * chunk_duration_ms / 1000)

    # Select audio device
    device_index = audio_utils.select_audio_device()

    # Create audio stream
    audio, stream = audio_utils.create_audio_stream(device_index, sample_rate, chunk_size)

    print(f"Recording for {duration_seconds} seconds... Speak now!")

    # Record audio
    frames = []
    total_chunks = int(duration_seconds * 1000 / chunk_duration_ms)

    try:
        for _ in range(total_chunks):
            chunk = audio_utils.read_audio_chunk(stream, chunk_size)
            if chunk:
                frames.append(chunk)
    finally:
        audio_utils.close_audio_stream(audio, stream)

    # Combine all chunks
    audio_data = b''.join(frames)
    print("Recording complete.")

    return audio_data


def process_audio_and_identify_speaker(audio_data: bytes, config: dict[str, Any],
                                       whisper_model, speaker_model, leader_tracker) -> dict[str, Any] | None:
    """
    Process recorded audio and identify the speaker.

    Args:
        audio_data: Raw audio data
        config: Configuration
        whisper_model: Whisper model for transcription
        speaker_model: Speaker recognition model
        leader_tracker: Speaker recognition tracker

    Returns:
        Dictionary with speaker info and transcription, or None if failed
    """
    sample_rate = config['audio']['sample_rate']

    # 1. Check volume
    volume_db = audio_utils.calculate_volume_db(audio_data)
    print(f"Volume: {volume_db:.1f} dB")

    if volume_db < config['audio']['volume_threshold_db']:
        print("Audio too quiet, skipping...")
        return None

    # 2. Transcribe
    print("Transcribing audio...")
    text, duration = transcription.transcribe_audio(
        whisper_model, audio_data, sample_rate,
        config['whisper']['language']
    )

    if not text or text.strip() == "":
        print("No transcription detected or empty text")
        print(f"Audio duration: {len(audio_data) / (sample_rate * 2):.2f}s")  # 2 bytes per sample
        return None

    print(f"Transcription: '{text}' ({duration:.2f}s)")

    # 3. Extract speaker embedding
    embedding = speaker_recognition.extract_embedding(
        speaker_model, audio_data, sample_rate
    )

    if embedding.size == 0:
        print("Failed to extract speaker embedding")
        return None

    # 4. Identify speaker
    speaker = "unknown"
    is_leader = False

    if not leader_tracker.has_leader():
        # Set as new leader
        leader_tracker.set_leader(embedding)
        speaker = "leader"
        is_leader = True
        print("New leader recognized!")
    else:
        # Check if this is the leader
        if leader_tracker.is_leader(embedding):
            speaker = "leader"
            is_leader = True
            print("Leader speaking again.")
        else:
            speaker = "other_person"
            print("Different person speaking.")

    # 5. Return result
    result = {
        "speaker": speaker,
        "text": text,
        "duration": round(duration, 2),
        "volume_db": round(volume_db, 1),
        "is_leader": is_leader,
        "timestamp": time.time()
    }

    return result


def print_speaker_and_text(result: dict[str, Any]) -> None:
    """Print the speaker and their transcribed text."""
    speaker = result['speaker']
    text = result['text']
    duration = result['duration']

    print("\n" + "=" * 60)
    print(f"SPEAKER: {speaker.upper()}")
    print(f"SAID: {text}")
    print(f"DURATION: {duration} seconds")
    print("=" * 60 + "\n")


def simple_audio_test(config: dict[str, Any], whisper_model, speaker_model, leader_tracker) -> None:
    """
    Simple audio recording and processing - record for fixed duration on button press.

    Args:
        config: Configuration
        whisper_model: Whisper model for transcription
        speaker_model: Speaker recognition model
        leader_tracker: Leader tracker
    """
    print("\n" + "=" * 60)
    print("SIMPLE SPEAKER RECOGNITION SYSTEM - Button-press recording")
    print("Press Enter to record audio, speak clearly for best results.")
    print("First person speaking becomes the LEADER, others are OTHER_PERSON")
    print("Press Ctrl+C to exit")
    print("=" * 60)

    # Get API endpoint from config
    api_endpoint = config.get('api', {}).get('endpoint_url', '')
    if api_endpoint:
        print(f"API endpoint configured: {api_endpoint}")
    else:
        print("Warning: No API endpoint configured in config.json")

    try:
        while True:
            input("\nPress Enter to start recording...")
            print("Recording for 5 seconds... Speak now!")

            # Record for fixed 5 seconds
            audio_data = record_audio_for_duration(5, config)

            # Process it
            result = process_audio_and_identify_speaker(
                audio_data, config, whisper_model, speaker_model, leader_tracker
            )

            if result:
                print_speaker_and_text(result)

                # Send to API asynchronously in background
                executor.submit(
                    send_transcription_sync,
                    result['text'],
                    result['speaker'],
                    api_endpoint
                )
                print("Sending transcription to API...")
            else:
                print("Could not process the audio - try speaking louder and clearer.")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        executor.shutdown(wait=True)


def main():
    """Main function."""
    setup_logging()

    print("Loading configuration...")
    config = load_config()

    print("Loading Whisper model...")
    whisper_model = transcription.load_whisper_model(
        config['whisper']['model_size'],
        config['whisper']['device'],
        config['whisper']['compute_type']
    )

    print("Loading speaker recognition model...")
    speaker_model = speaker_recognition.load_speaker_model(
        device=config['whisper']['device']
    )

    print("Setting up leader tracker...")
    leader_tracker = speaker_recognition.LeaderTracker(
        config['speaker_recognition']['leader_timeout_seconds'],
        config['speaker_recognition']['threshold']
    )

    # Start simple audio test
    simple_audio_test(config, whisper_model, speaker_model, leader_tracker)

    print("Goodbye!")


if __name__ == "__main__":
    main()