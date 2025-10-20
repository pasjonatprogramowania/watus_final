
"""
Testy jednostkowe dla modułu speaker_recognition.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

import speaker_recognition


def test_calculate_similarity_identical():
    """Test podobieństwa identycznych embeddingów."""
    embedding = np.array([1.0, 2.0, 3.0, 4.0])
    similarity = speaker_recognition.calculate_similarity(embedding, embedding)
    assert abs(similarity - 1.0) < 0.001, "Identyczne embeddingi powinny mieć similarity=1.0"


def test_calculate_similarity_different():
    """Test podobieństwa różnych embeddingów."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    similarity = speaker_recognition.calculate_similarity(emb1, emb2)
    assert abs(similarity) < 0.001, "Ortogonalne embeddingi powinny mieć similarity≈0"


def test_calculate_similarity_empty():
    """Test podobieństwa pustych embeddingów."""
    emb1 = np.array([])
    emb2 = np.array([1.0, 2.0])
    similarity = speaker_recognition.calculate_similarity(emb1, emb2)
    assert similarity == 0.0, "Puste embeddingi powinny zwrócić 0"


def test_leader_tracker_initialization():
    """Test inicjalizacji tracker lidera."""
    tracker = speaker_recognition.LeaderTracker(timeout_seconds=180, similarity_threshold=0.85)
    assert tracker.timeout_seconds == 180
    assert tracker.similarity_threshold == 0.85
    assert tracker.has_leader() == False


def test_leader_tracker_set_leader():
    """Test ustawiania lidera."""
    tracker = speaker_recognition.LeaderTracker(timeout_seconds=180, similarity_threshold=0.85)
    embedding = np.array([1.0, 2.0, 3.0])
    
    tracker.set_leader(embedding)
    
    assert tracker.has_leader() == True
    assert tracker.leader_embedding is not None
    assert np.array_equal(tracker.leader_embedding, embedding)


def test_leader_tracker_is_leader():
    """Test sprawdzania czy to lider."""
    tracker = speaker_recognition.LeaderTracker(timeout_seconds=180, similarity_threshold=0.85)
    
    # Ustaw lidera
    leader_emb = np.array([1.0, 2.0, 3.0, 4.0])
    tracker.set_leader(leader_emb)
    
    # Ten sam embedding (similarity=1.0)
    assert tracker.is_leader(leader_emb) == True
    
    # Bardzo podobny embedding
    similar_emb = leader_emb + 0.01  # Mała zmiana
    is_same = tracker.is_leader(similar_emb)
    # Powinno być True bo similarity bardzo bliska 1.0
    assert is_same == True
    
    # Zupełnie inny embedding
    different_emb = np.array([10.0, -20.0, 30.0, -40.0])
    assert tracker.is_leader(different_emb) == False


def test_leader_tracker_reset():
    """Test resetowania lidera."""
    tracker = speaker_recognition.LeaderTracker(timeout_seconds=180, similarity_threshold=0.85)
    
    embedding = np.array([1.0, 2.0, 3.0])
    tracker.set_leader(embedding)
    assert tracker.has_leader() == True
    
    tracker.reset()
    assert tracker.has_leader() == False
    assert tracker.leader_embedding is None
