"""Тесты для детерминизма и персонализации инстансов."""

from __future__ import annotations

import numpy as np
import pytest

from ml3_grade.instances import student_seed, make_cnn_zoo


def test_seed_deterministic():
    s1 = student_seed("04-cnn-zoo", "alice")
    s2 = student_seed("04-cnn-zoo", "alice")
    assert s1 == s2


def test_seed_changes_per_student():
    s_alice = student_seed("04-cnn-zoo", "alice")
    s_bob   = student_seed("04-cnn-zoo", "bob")
    assert s_alice != s_bob


def test_seed_changes_per_hw():
    s1 = student_seed("04-cnn-zoo", "alice")
    s2 = student_seed("05-rnn", "alice")
    assert s1 != s2


def test_make_cnn_zoo_two_students_differ(tmp_path):
    """Два студента должны получить заметно разные test split'ы."""
    # Синтетический pool
    rng = np.random.default_rng(0)
    num_classes = 40
    n = 10_000
    images = rng.integers(0, 255, size=(n, 3, 16, 16), dtype=np.uint8)
    labels = rng.integers(0, num_classes, size=n, dtype=np.int64)

    pool = tmp_path / "data" / "hidden" / "04-cnn-zoo" / "pool.npz"
    pool.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pool, images=images, labels=labels)

    alice = make_cnn_zoo(tmp_path, "04-cnn-zoo", "alice",
                        pool_path=pool, test_size=500)
    bob = make_cnn_zoo(tmp_path, "04-cnn-zoo", "bob",
                       pool_path=pool, test_size=500)

    a = np.load(alice.hidden_path)
    b = np.load(bob.hidden_path)

    # Размеры одинаковые
    assert a["labels"].shape == b["labels"].shape == (500,)

    # Но label-распределения разные
    assert not np.array_equal(a["labels"], b["labels"])

    # И дропнутые классы разные (хотя бы частично)
    assert alice.description["dropped_classes"] != bob.description["dropped_classes"] \
        or alice.description["class_counts"] != bob.description["class_counts"]


def test_make_cnn_zoo_deterministic(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(5000, 3, 16, 16), dtype=np.uint8)
    labels = rng.integers(0, 40, size=5000, dtype=np.int64)
    pool = tmp_path / "pool.npz"
    np.savez_compressed(pool, images=images, labels=labels)

    first = make_cnn_zoo(tmp_path, "04-cnn-zoo", "eve",
                         pool_path=pool, test_size=400)
    a = np.load(first.hidden_path)

    # Второй вызов — тот же сабжект, должны получить идентично
    second = make_cnn_zoo(tmp_path, "04-cnn-zoo", "eve",
                          pool_path=pool, test_size=400)
    b = np.load(second.hidden_path)

    assert np.array_equal(a["labels"], b["labels"])
    assert np.array_equal(a["images"], b["images"])


def test_dropped_classes_not_in_test(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(5000, 3, 16, 16), dtype=np.uint8)
    labels = rng.integers(0, 40, size=5000, dtype=np.int64)
    pool = tmp_path / "pool.npz"
    np.savez_compressed(pool, images=images, labels=labels)

    inst = make_cnn_zoo(tmp_path, "04-cnn-zoo", "carol",
                        pool_path=pool, test_size=400)
    test_labels = np.load(inst.hidden_path)["labels"]
    for dropped in inst.description["dropped_classes"]:
        assert int(dropped) not in test_labels.tolist()
