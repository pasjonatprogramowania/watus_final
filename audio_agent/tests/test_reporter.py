
"""
Testy jednostkowe dla modułu reporter.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '..')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import reporter


def test_read_jsonl_file_empty():
    """Test odczytu nieistniejącego pliku."""
    entries = reporter.read_jsonl_file("nonexistent.jsonl")
    assert entries == [], "Nieistniejący plik powinien zwrócić pustą listę"


def test_read_jsonl_file():
    """Test odczytu pliku JSONL."""
    # Utwórz tymczasowy plik
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        json.dump({"test": "data1"}, f, ensure_ascii=False)
        f.write('\n')
        json.dump({"test": "data2"}, f, ensure_ascii=False)
        f.write('\n')
        temp_path = f.name
    
    try:
        entries = reporter.read_jsonl_file(temp_path, max_entries=10)
        assert len(entries) == 2, "Powinno odczytać 2 wpisy"
        assert entries[0]["test"] == "data1"
        assert entries[1]["test"] == "data2"
    finally:
        Path(temp_path).unlink()


def test_read_jsonl_file_max_entries():
    """Test limitu wpisów."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        for i in range(10):
            json.dump({"id": i}, f, ensure_ascii=False)
            f.write('\n')
        temp_path = f.name
    
    try:
        entries = reporter.read_jsonl_file(temp_path, max_entries=5)
        assert len(entries) == 5, "Powinno odczytać tylko 5 ostatnich wpisów"
        assert entries[0]["id"] == 5, "Powinny być ostatnie wpisy (5-9)"
        assert entries[-1]["id"] == 9
    finally:
        Path(temp_path).unlink()


def test_filter_recent_entries():
    """Test filtrowania ostatnich wpisów."""
    now = datetime.now()
    entries = [
        {"timestamp": (now - timedelta(minutes=2)).isoformat(), "data": "recent"},
        {"timestamp": (now - timedelta(minutes=20)).isoformat(), "data": "old"},
        {"timestamp": (now - timedelta(minutes=5)).isoformat(), "data": "recent2"},
    ]
    
    recent = reporter.filter_recent_entries(entries, minutes=10)
    assert len(recent) == 2, "Powinno zwrócić 2 ostatnie wpisy"
    assert recent[0]["data"] == "recent"
    assert recent[1]["data"] == "recent2"


def test_filter_recent_entries_empty():
    """Test filtrowania pustej listy."""
    recent = reporter.filter_recent_entries([], minutes=10)
    assert recent == [], "Pusta lista powinna pozostać pusta"


def test_prepare_llm_prompt():
    """Test przygotowania promptu dla LLM."""
    context = {
        "system_time": "2025-10-16 12:00:00",
        "dialog": [
            {"is_leader": True, "text": "Cześć Watusiu"},
            {"is_leader": True, "text": "Jak się masz?"}
        ],
        "camera": [{"description": "widać osobę"}],
        "lidar": [{"obstacles": "brak przeszkód"}],
        "scenarios": [{"name": "patrol", "status": "active"}]
    }
    
    prompt = reporter.prepare_llm_prompt(context, "Jaka jest pogoda?")
    
    assert "Watuś" in prompt, "Prompt powinien zawierać nazwę robota"
    assert "Jaka jest pogoda?" in prompt, "Prompt powinien zawierać zapytanie"
    assert "Cześć Watusiu" in prompt, "Prompt powinien zawierać dialog"
    assert "widać osobę" in prompt, "Prompt powinien zawierać dane z kamery"
