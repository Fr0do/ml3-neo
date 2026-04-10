"""Per-student task instance generator.

Идея: каждый студент получает *свой* hidden test split и *свой* data twist,
детерминированно выведенные из `sha256(hw_id + student + course_salt)`.

Следствия:
- Копирование чужого `predictions.npy` не работает: два студента
  сравниваются против разных labels.
- Копирование чужого кода работает только если код действительно запускается
  на своих данных (см. `execute.py`).
- Ablations, которые студент показывает в MODEL.md, зависят от специфики
  его инстанса: probing-вопросы judge'а укоренены именно в них.

Пул большого датасета лежит в `data/hidden/<hw_id>/pool.npz` (единый для
курса, gitignored). Генератор берёт оттуда `test_size` элементов по
stratified-сэмплированию с seed от студента.

Формат выходного файла — тот же, что ожидает `eval.py`:
    test.npz: images (N, C, H, W) uint8, labels (N,) int64
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

COURSE_SALT = "ml3-neo-2026"


def student_seed(hw_id: str, student: str, extra: str = "") -> int:
    """32-bit детерминированный seed на пару (hw, студент).

    Соль курса вшита, чтобы один и тот же student не получал одинаковые
    инстансы в разные годы.
    """
    payload = f"{COURSE_SALT}::{hw_id}::{student}::{extra}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# Общий API
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """Описание персонального инстанса ДЗ для одного студента."""
    hw_id: str
    student: str
    seed: int
    hidden_path: Path                 # где лежит его test.npz
    meta_path: Path                   # sidecar с описанием (что подсунули, сколько классов и т.п.)
    description: dict                 # тот же JSON, что и meta_path; для probing


# ---------------------------------------------------------------------------
# Конкретный генератор для 04-cnn-zoo
# ---------------------------------------------------------------------------

def make_cnn_zoo(
    root: Path,
    hw_id: str,
    student: str,
    pool_path: Optional[Path] = None,
    test_size: int = 4000,
) -> Instance:
    """Генерирует personal hidden test для 04-cnn-zoo.

    Стратегия:
    - Берём глобальный pool (10k элементов).
    - Sampler: stratified по labels, но с добавленным per-student шумом на
      частоту каждого класса → у одного в test больше класса 7, у другого —
      больше класса 23. Это не ломает метрику, но меняет «форму» задачи.
    - Dropping: у каждого студента случайно выбираются 3 класса, которые
      полностью исключены из test (студент этого не знает заранее — узнаёт
      из description после сабмита). Модель, натренированная на всех 40,
      всё ещё работает; модель, случайно переобученная на какой-то класс
      — пострадает.
    """
    rng = np.random.default_rng(student_seed(hw_id, student))

    pool = pool_path or (root / "data" / "hidden" / hw_id / "pool.npz")
    if not pool.exists():
        raise FileNotFoundError(
            f"pool not found: {pool}\n"
            f"run `ml3 fetch-data {hw_id}` on the instructor machine first"
        )

    data = np.load(pool)
    images, labels = data["images"], data["labels"]
    num_classes = int(labels.max()) + 1

    # 1. Drop 3 classes
    dropped = rng.choice(num_classes, size=3, replace=False).tolist()
    mask = ~np.isin(labels, dropped)
    images_f = images[mask]
    labels_f = labels[mask]

    # 2. Per-class frequency twist: reweight stratified sampling
    weights = rng.dirichlet(np.ones(num_classes) * 3.0)  # ≈ uniform but jittered
    weights = np.where(np.isin(np.arange(num_classes), dropped), 0.0, weights)
    weights = weights / weights.sum()

    # 3. Stratified sample
    per_class_target = (weights * test_size).astype(int)
    # правка до нужного размера
    deficit = test_size - per_class_target.sum()
    if deficit > 0:
        # добавим остаток в самые "тяжёлые" классы
        order = np.argsort(-weights)
        for i in range(deficit):
            per_class_target[order[i % len(order)]] += 1

    picks: list[int] = []
    for c in range(num_classes):
        if per_class_target[c] == 0:
            continue
        idx = np.where(labels_f == c)[0]
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(per_class_target[c], len(idx)), replace=False)
        picks.extend(take.tolist())

    rng.shuffle(picks)
    picks_arr = np.array(picks, dtype=np.int64)
    test_images = images_f[picks_arr]
    test_labels = labels_f[picks_arr]

    # 4. Сохраняем
    out_dir = root / "grader" / "state" / "instances" / hw_id / student
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = out_dir / "test.npz"
    np.savez_compressed(hidden_path, images=test_images, labels=test_labels)

    description = {
        "hw_id": hw_id,
        "student": student,
        "seed": int(student_seed(hw_id, student)),
        "num_classes": int(num_classes),
        "dropped_classes": dropped,
        "test_size": int(len(picks_arr)),
        "class_counts": {
            int(c): int((test_labels == c).sum()) for c in range(num_classes)
        },
        "least_populated_class": int(
            min(
                (int(c) for c in range(num_classes) if int((test_labels == c).sum()) > 0),
                key=lambda c: int((test_labels == c).sum()),
            )
        ),
    }
    meta_path = out_dir / "description.json"
    meta_path.write_text(json.dumps(description, indent=2, ensure_ascii=False))

    return Instance(
        hw_id=hw_id,
        student=student,
        seed=description["seed"],
        hidden_path=hidden_path,
        meta_path=meta_path,
        description=description,
    )


# ---------------------------------------------------------------------------
# Генератор для 02-pytorch
# ---------------------------------------------------------------------------

def make_pytorch(
    root: Path,
    hw_id: str,
    student: str,
    test_size: int = 2000,
) -> Instance:
    """Генерирует personal hidden test для 02-pytorch.

    Стратегия:
    - 10 классов (MNIST-style), shape (1, 28, 28), uint8.
    - Gaussian blobs с добавленным шумом. У каждого студента своё
      dirichlet-взвешенное соотношение классов.
    """
    seed = student_seed(hw_id, student)
    rng = np.random.default_rng(seed)

    num_classes = 10
    # Per-student class proportions
    class_weights = rng.dirichlet(np.ones(num_classes) * 2.0)
    per_class = (class_weights * test_size).astype(int)
    deficit = test_size - per_class.sum()
    order = np.argsort(-class_weights)
    for i in range(deficit):
        per_class[order[i % num_classes]] += 1

    # Gaussian blob centers (fixed grid, but offset by student rng)
    centers = rng.uniform(0.1, 0.9, size=(num_classes, 2))

    images_list, labels_list = [], []
    for c in range(num_classes):
        n = int(per_class[c])
        if n == 0:
            continue
        cx, cy = centers[c]
        # 2-D coordinates → 28×28 pixel positions
        xs = rng.normal(cx * 28, 3.5, size=n).clip(0, 27)
        ys = rng.normal(cy * 28, 3.5, size=n).clip(0, 27)
        # Build grayscale images: bright blob on noisy background
        imgs = rng.integers(0, 40, size=(n, 28, 28), dtype=np.uint8)
        xi = np.round(xs).astype(int).clip(0, 27)
        yi = np.round(ys).astype(int).clip(0, 27)
        for i in range(n):
            # Draw a small bright circle around the blob center
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx * dx + dy * dy <= 9:
                        py = (yi[i] + dy) % 28
                        px = (xi[i] + dx) % 28
                        imgs[i, py, px] = min(255, int(imgs[i, py, px]) + 150 + int(rng.integers(0, 50)))
        images_list.append(imgs[:, np.newaxis, :, :])  # (N,1,28,28)
        labels_list.append(np.full(n, c, dtype=np.int64))

    test_images = np.concatenate(images_list, axis=0)
    test_labels = np.concatenate(labels_list, axis=0)
    shuffle_idx = rng.permutation(len(test_labels))
    test_images = test_images[shuffle_idx]
    test_labels = test_labels[shuffle_idx]

    out_dir = root / "grader" / "state" / "instances" / hw_id / student
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = out_dir / "test.npz"
    np.savez_compressed(hidden_path, images=test_images, labels=test_labels)

    imbalance_ratio = float(per_class.max()) / max(1, int(per_class[per_class > 0].min()))
    description = {
        "hw_id": hw_id,
        "student": student,
        "seed": int(seed),
        "task": "mnist_style_classification",
        "num_classes": num_classes,
        "test_size": int(len(test_labels)),
        "imbalance_ratio": round(imbalance_ratio, 3),
        "class_counts": {int(c): int((test_labels == c).sum()) for c in range(num_classes)},
    }
    meta_path = out_dir / "description.json"
    meta_path.write_text(json.dumps(description, indent=2, ensure_ascii=False))

    return Instance(
        hw_id=hw_id,
        student=student,
        seed=int(seed),
        hidden_path=hidden_path,
        meta_path=meta_path,
        description=description,
    )


# ---------------------------------------------------------------------------
# Генератор для 03-mlp-norm
# ---------------------------------------------------------------------------

def make_mlp_norm(
    root: Path,
    hw_id: str,
    student: str,
    test_size: int = 2000,
) -> Instance:
    """Генерирует personal hidden test для 03-mlp-norm.

    Стратегия:
    - 5 классов, shape (1, 8, 8), uint8 (flatten → 64 features).
    - Смесь гауссиан с разными центрами, затем случайная rotation-матрица
      применяется к 64-мерному вектору признаков (twist).
    """
    seed = student_seed(hw_id, student)
    rng = np.random.default_rng(seed)

    num_classes = 5
    n_features = 64  # 8×8

    # Per-student Gaussian centers in 64-d space (low-rank: 2-d manifold rotated)
    # Generate 2-d centers, embed into 64-d via random projection
    centers_2d = rng.uniform(-3.0, 3.0, size=(num_classes, 2))
    embed = rng.standard_normal(size=(2, n_features))
    centers_64 = centers_2d @ embed  # (num_classes, 64)

    # Random rotation matrix (student-specific twist)
    rand_mat = rng.standard_normal(size=(n_features, n_features))
    Q, _ = np.linalg.qr(rand_mat)  # orthogonal matrix

    per_class = np.full(num_classes, test_size // num_classes, dtype=int)
    per_class[:test_size % num_classes] += 1

    images_list, labels_list = [], []
    for c in range(num_classes):
        n = int(per_class[c])
        feats = rng.normal(centers_64[c], 1.0, size=(n, n_features))
        feats = feats @ Q  # apply rotation twist
        # Normalize to [0, 255] uint8
        feats_min, feats_max = feats.min(), feats.max()
        feats_norm = (feats - feats_min) / max(feats_max - feats_min, 1e-8)
        imgs = (feats_norm * 255).astype(np.uint8).reshape(n, 1, 8, 8)
        images_list.append(imgs)
        labels_list.append(np.full(n, c, dtype=np.int64))

    test_images = np.concatenate(images_list, axis=0)
    test_labels = np.concatenate(labels_list, axis=0)
    shuffle_idx = rng.permutation(len(test_labels))
    test_images = test_images[shuffle_idx]
    test_labels = test_labels[shuffle_idx]

    out_dir = root / "grader" / "state" / "instances" / hw_id / student
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = out_dir / "test.npz"
    np.savez_compressed(hidden_path, images=test_images, labels=test_labels)

    description = {
        "hw_id": hw_id,
        "student": student,
        "seed": int(seed),
        "task": "tabular_mlp_classification",
        "num_classes": num_classes,
        "test_size": int(len(test_labels)),
        "n_features": n_features,
        "class_counts": {int(c): int((test_labels == c).sum()) for c in range(num_classes)},
        "twist": "random_rotation_applied",
    }
    meta_path = out_dir / "description.json"
    meta_path.write_text(json.dumps(description, indent=2, ensure_ascii=False))

    return Instance(
        hw_id=hw_id,
        student=student,
        seed=int(seed),
        hidden_path=hidden_path,
        meta_path=meta_path,
        description=description,
    )


# ---------------------------------------------------------------------------
# Генератор для 05-sequence
# ---------------------------------------------------------------------------

def make_sequence(
    root: Path,
    hw_id: str,
    student: str,
    test_size: int = 1500,
) -> Instance:
    """Генерирует personal hidden test для 05-sequence.

    Стратегия:
    - 5 классов, shape (1, 64, 64), uint8.
    - Каждый «образец» — 64-токенная последовательность (vocab=64),
      уложенная в матрицу 64×64 (строка = один сэмпл последовательности,
      всего 64 строки для богатства паттернов).
    - Класс определяется наличием key_tokens (student-specific) в фиксированных
      позициях. Остальные позиции — случайный шум.
    """
    seed = student_seed(hw_id, student)
    rng = np.random.default_rng(seed)

    num_classes = 5
    vocab_size = 64
    seq_len = 64

    # Student-specific key tokens per class (3 key tokens each)
    key_tokens = [
        rng.choice(vocab_size, size=3, replace=False).tolist()
        for _ in range(num_classes)
    ]
    # Student-specific key positions per class
    key_positions = [
        rng.choice(seq_len, size=3, replace=False).tolist()
        for _ in range(num_classes)
    ]

    per_class = np.full(num_classes, test_size // num_classes, dtype=int)
    per_class[:test_size % num_classes] += 1

    images_list, labels_list = [], []
    for c in range(num_classes):
        n = int(per_class[c])
        # Base: random noise sequences, shape (n, seq_len)
        seqs = rng.integers(0, vocab_size, size=(n, seq_len), dtype=np.uint8)
        # Inject key tokens at key positions
        for tok, pos in zip(key_tokens[c], key_positions[c]):
            seqs[:, pos] = tok
        # Layout: tile seq into 64×64 matrix (repeat sequence across rows)
        imgs = np.tile(seqs[:, np.newaxis, :], (1, seq_len, 1))  # (n, 64, 64)
        # Add per-row jitter to make it non-trivially repeating
        row_jitter = rng.integers(0, 8, size=(n, seq_len, seq_len), dtype=np.uint8)
        imgs = ((imgs.astype(np.int16) + row_jitter) % vocab_size).astype(np.uint8)
        # Restore key positions in row 0 (first row = clean signal)
        for tok, pos in zip(key_tokens[c], key_positions[c]):
            imgs[:, 0, pos] = tok
        imgs = imgs[:, np.newaxis, :, :]  # (n, 1, 64, 64)
        images_list.append(imgs)
        labels_list.append(np.full(n, c, dtype=np.int64))

    test_images = np.concatenate(images_list, axis=0)
    test_labels = np.concatenate(labels_list, axis=0)
    shuffle_idx = rng.permutation(len(test_labels))
    test_images = test_images[shuffle_idx]
    test_labels = test_labels[shuffle_idx]

    out_dir = root / "grader" / "state" / "instances" / hw_id / student
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = out_dir / "test.npz"
    np.savez_compressed(hidden_path, images=test_images, labels=test_labels)

    description = {
        "hw_id": hw_id,
        "student": student,
        "seed": int(seed),
        "task": "sequence_classification",
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "test_size": int(len(test_labels)),
        "key_tokens": key_tokens,
        "key_positions": key_positions,
        "class_counts": {int(c): int((test_labels == c).sum()) for c in range(num_classes)},
    }
    meta_path = out_dir / "description.json"
    meta_path.write_text(json.dumps(description, indent=2, ensure_ascii=False))

    return Instance(
        hw_id=hw_id,
        student=student,
        seed=int(seed),
        hidden_path=hidden_path,
        meta_path=meta_path,
        description=description,
    )


# ---------------------------------------------------------------------------
# Генератор для 06-transformer
# ---------------------------------------------------------------------------

def make_transformer(
    root: Path,
    hw_id: str,
    student: str,
    test_size: int = 1500,
) -> Instance:
    """Генерирует personal hidden test для 06-transformer.

    Стратегия:
    - 3 класса, shape (1, 32, 32), uint8.
    - Каждый «образец» — матрица token IDs 32×32 = 1024 токена (vocab=128).
    - Класс определяется позиционными паттернами в начале/середине/конце
      документа. У каждого студента разные позиции и токены для паттернов.
    """
    seed = student_seed(hw_id, student)
    rng = np.random.default_rng(seed)

    num_classes = 3
    vocab_size = 128
    h, w = 32, 32

    # Student-specific pattern positions (which rows carry the class signal)
    # Class 0: pattern in first rows, class 1: middle, class 2: last
    row_offsets = rng.integers(0, 5, size=num_classes).tolist()
    pattern_rows = [
        [row_offsets[0], row_offsets[0] + 1],           # near start
        [h // 2 + row_offsets[1], h // 2 + row_offsets[1] + 1],  # near middle
        [h - 2 - row_offsets[2], h - 1 - row_offsets[2]],        # near end
    ]
    # Student-specific signature tokens per class (4 tokens forming a pattern)
    signature_tokens = [
        rng.choice(vocab_size, size=4, replace=False).tolist()
        for _ in range(num_classes)
    ]
    # Positions within a row where pattern is placed
    pattern_cols = rng.choice(w, size=4, replace=False).tolist()

    per_class = np.full(num_classes, test_size // num_classes, dtype=int)
    per_class[:test_size % num_classes] += 1

    images_list, labels_list = [], []
    for c in range(num_classes):
        n = int(per_class[c])
        # Base: random token IDs
        docs = rng.integers(0, vocab_size, size=(n, h, w), dtype=np.uint8)
        # Inject signature pattern into the designated rows
        for row in pattern_rows[c]:
            row = max(0, min(h - 1, row))
            for tok, col in zip(signature_tokens[c], pattern_cols):
                docs[:, row, col] = tok
        imgs = docs[:, np.newaxis, :, :]  # (n, 1, 32, 32)
        images_list.append(imgs)
        labels_list.append(np.full(n, c, dtype=np.int64))

    test_images = np.concatenate(images_list, axis=0)
    test_labels = np.concatenate(labels_list, axis=0)
    shuffle_idx = rng.permutation(len(test_labels))
    test_images = test_images[shuffle_idx]
    test_labels = test_labels[shuffle_idx]

    out_dir = root / "grader" / "state" / "instances" / hw_id / student
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = out_dir / "test.npz"
    np.savez_compressed(hidden_path, images=test_images, labels=test_labels)

    description = {
        "hw_id": hw_id,
        "student": student,
        "seed": int(seed),
        "task": "document_classification",
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "test_size": int(len(test_labels)),
        "pattern_positions": {
            str(c): {"rows": pattern_rows[c], "cols": pattern_cols, "tokens": signature_tokens[c]}
            for c in range(num_classes)
        },
        "class_counts": {int(c): int((test_labels == c).sum()) for c in range(num_classes)},
    }
    meta_path = out_dir / "description.json"
    meta_path.write_text(json.dumps(description, indent=2, ensure_ascii=False))

    return Instance(
        hw_id=hw_id,
        student=student,
        seed=int(seed),
        hidden_path=hidden_path,
        meta_path=meta_path,
        description=description,
    )


# ---------------------------------------------------------------------------
# Регистр генераторов
# ---------------------------------------------------------------------------

_GENERATORS = {
    "02-pytorch": make_pytorch,
    "03-mlp-norm": make_mlp_norm,
    "04-cnn-zoo": make_cnn_zoo,
    "05-sequence": make_sequence,
    "06-transformer": make_transformer,
}


def make_instance(root: Path, hw_id: str, student: str) -> Instance:
    """Ищет генератор для данного hw_id в регистре."""
    gen = _GENERATORS.get(hw_id)
    if gen is None:
        raise SystemExit(
            f"no instance generator registered for {hw_id}. "
            f"Add one to grader/ml3_grade/instances.py::_GENERATORS."
        )
    return gen(root, hw_id, student)
