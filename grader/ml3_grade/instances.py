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
# Регистр генераторов
# ---------------------------------------------------------------------------

_GENERATORS = {
    "04-cnn-zoo": make_cnn_zoo,
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
