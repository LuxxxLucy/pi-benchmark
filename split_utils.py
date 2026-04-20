"""Deterministic hash-based dataset splitting.

Shared by train_classifier.py and run_benchmark.py to ensure
train/test partitions are consistent and non-overlapping.
"""
import hashlib


def is_test_partition(text: str, test_fraction: float = 0.3, seed: int = 42) -> bool:
    """Deterministic hash-based split. Returns True if sample belongs to test set."""
    h = hashlib.md5(f"{seed}:{text}".encode()).hexdigest()
    return (int(h, 16) % 1000) < int(test_fraction * 1000)
