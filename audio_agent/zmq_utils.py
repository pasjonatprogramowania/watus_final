
"""
Moduł komunikacji ZeroMQ.
"""

import zmq
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_zmq_publisher(address: str) -> tuple[zmq.Context, zmq.Socket]:
    """
    Tworzy publisher ZeroMQ.
    
    Args:
        address: Adres (np. tcp://127.0.0.1:5555)
    
    Returns:
        Krotka (Context, Socket)
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(address)
    logger.info(f"ZMQ Publisher utworzony: {address}")
    return context, socket


def create_zmq_subscriber(address: str, topic: str = "") -> tuple[zmq.Context, zmq.Socket]:
    """
    Tworzy subscriber ZeroMQ.
    
    Args:
        address: Adres do połączenia
        topic: Temat subskrypcji (pusty = wszystko)
    
    Returns:
        Krotka (Context, Socket)
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(address)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    logger.info(f"ZMQ Subscriber połączony: {address}, topic='{topic}'")
    return context, socket


def publish_message(socket: zmq.Socket, message: dict[str, Any], 
                   topic: str = "") -> None:
    """
    Publikuje wiadomość JSON.
    
    Args:
        socket: Socket ZeroMQ
        message: Wiadomość do wysłania
        topic: Temat (opcjonalny)
    """
    try:
        json_str = json.dumps(message, ensure_ascii=False)
        if topic:
            full_message = f"{topic} {json_str}"
        else:
            full_message = json_str
        
        socket.send_string(full_message)
        logger.debug(f"Opublikowano wiadomość: {topic}")
    except Exception as e:
        logger.error(f"Błąd publikacji wiadomości: {e}")


def receive_message(socket: zmq.Socket, timeout_ms: int | None = None) -> dict[str, Any] | None:
    """
    Odbiera wiadomość JSON.
    
    Args:
        socket: Socket ZeroMQ
        timeout_ms: Timeout w milisekundach (None = blokujący)
    
    Returns:
        Wiadomość lub None w przypadku timeout
    """
    try:
        if timeout_ms is not None:
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        
        message = socket.recv_string()
        
        # Usuń topic jeśli istnieje
        if ' ' in message:
            _, json_str = message.split(' ', 1)
        else:
            json_str = message
        
        return json.loads(json_str)
    
    except zmq.Again:
        logger.debug("Timeout odbierania wiadomości")
        return None
    except Exception as e:
        logger.error(f"Błąd odbierania wiadomości: {e}")
        return None


def close_zmq_socket(context: zmq.Context, socket: zmq.Socket) -> None:
    """
    Zamyka socket i kontekst ZeroMQ.
    
    Args:
        context: Kontekst ZeroMQ
        socket: Socket ZeroMQ
    """
    try:
        socket.close()
        context.term()
        logger.info("Zamknięto socket ZeroMQ")
    except Exception as e:
        logger.error(f"Błąd zamykania ZeroMQ: {e}")
