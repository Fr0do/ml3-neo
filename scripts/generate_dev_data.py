"""generate_dev_data.py — генерация публичных dev/train сетов для ДЗ.

Это СИНТЕТИЧЕСКИЕ данные для отладки и разработки.
Они не являются реальными датасетами и созданы исключительно чтобы
студенты могли запускать свой код без скачивания внешних данных.

Использование:
    python scripts/generate_dev_data.py

Результат:
    homework/<hw>/data/public/dev.npz
    homework/<hw>/data/public/train.npz

Формат файлов: {images: ndarray (N,C,H,W) uint8, labels: ndarray (N,) int64}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DEV_SIZE = 1000
TRAIN_SIZE = 3000
GLOBAL_SEED = 42  # единый seed, не персонализированный

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def save_split(out_dir: Path, name: str, images: np.ndarray, labels: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, images=images, labels=labels)
    print(f"  saved {path}  [{images.shape}, {labels.shape}]")


# ---------------------------------------------------------------------------
# 02-pytorch: MNIST-style, 10 классов, shape (1, 28, 28)
# ---------------------------------------------------------------------------


def generate_pytorch(seed: int = GLOBAL_SEED, dev_size: int = DEV_SIZE, train_size: int = TRAIN_SIZE):
    """Синтетические Gaussian-blob изображения 28x28, 10 классов."""
    hw = "02-pytorch"
    out_dir = REPO_ROOT / "homework" / hw / "data" / "public"
    print(f"\n[{hw}] Generating synthetic MNIST-style data ...")

    rng = np.random.default_rng(seed)
    num_classes = 10
    centers = rng.uniform(0.1, 0.9, size=(num_classes, 2))

    def make_split(n: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(split_seed)
        per_class = np.full(num_classes, n // num_classes, dtype=int)
        per_class[:n % num_classes] += 1
        imgs_list, lbl_list = [], []
        for c in range(num_classes):
            nc = int(per_class[c])
            cx, cy = centers[c]
            xs = r.normal(cx * 28, 3.5, size=nc).clip(0, 27)
            ys = r.normal(cy * 28, 3.5, size=nc).clip(0, 27)
            imgs = r.integers(0, 40, size=(nc, 28, 28), dtype=np.uint8)
            xi = np.round(xs).astype(int).clip(0, 27)
            yi = np.round(ys).astype(int).clip(0, 27)
            for i in range(nc):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dx * dx + dy * dy <= 9:
                            py = (yi[i] + dy) % 28
                            px = (xi[i] + dx) % 28
                            imgs[i, py, px] = min(255, int(imgs[i, py, px]) + 150 + int(r.integers(0, 50)))
            imgs_list.append(imgs[:, np.newaxis, :, :])
            lbl_list.append(np.full(nc, c, dtype=np.int64))
        images = np.concatenate(imgs_list, axis=0)
        labels = np.concatenate(lbl_list, axis=0)
        idx = r.permutation(len(labels))
        return images[idx], labels[idx]

    dev_images, dev_labels = make_split(dev_size, seed)
    train_images, train_labels = make_split(train_size, seed + 1)

    save_split(out_dir, "dev", dev_images, dev_labels)
    save_split(out_dir, "train", train_images, train_labels)

    meta = {
        "task": "mnist_style_classification",
        "num_classes": num_classes,
        "image_shape": [1, 28, 28],
        "note": "Synthetic data for debugging. NOT real MNIST.",
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 03-mlp-norm: tabular (8x8), 5 классов
# ---------------------------------------------------------------------------


def generate_mlp_norm(seed: int = GLOBAL_SEED, dev_size: int = DEV_SIZE, train_size: int = TRAIN_SIZE):
    """Синтетические Gaussian-mixture данные 8x8 (64 features), 5 классов."""
    hw = "03-mlp-norm"
    out_dir = REPO_ROOT / "homework" / hw / "data" / "public"
    print(f"\n[{hw}] Generating synthetic tabular (MLP) data ...")

    rng = np.random.default_rng(seed)
    num_classes = 5
    n_features = 64
    centers_2d = rng.uniform(-3.0, 3.0, size=(num_classes, 2))
    embed = rng.standard_normal(size=(2, n_features))
    centers_64 = centers_2d @ embed
    rand_mat = rng.standard_normal(size=(n_features, n_features))
    Q, _ = np.linalg.qr(rand_mat)

    def make_split(n: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(split_seed)
        per_class = np.full(num_classes, n // num_classes, dtype=int)
        per_class[:n % num_classes] += 1
        imgs_list, lbl_list = [], []
        for c in range(num_classes):
            nc = int(per_class[c])
            feats = r.normal(centers_64[c], 1.0, size=(nc, n_features))
            feats = feats @ Q
            f_min, f_max = feats.min(), feats.max()
            feats_norm = (feats - f_min) / max(f_max - f_min, 1e-8)
            imgs = (feats_norm * 255).astype(np.uint8).reshape(nc, 1, 8, 8)
            imgs_list.append(imgs)
            lbl_list.append(np.full(nc, c, dtype=np.int64))
        images = np.concatenate(imgs_list, axis=0)
        labels = np.concatenate(lbl_list, axis=0)
        idx = r.permutation(len(labels))
        return images[idx], labels[idx]

    dev_images, dev_labels = make_split(dev_size, seed)
    train_images, train_labels = make_split(train_size, seed + 1)

    save_split(out_dir, "dev", dev_images, dev_labels)
    save_split(out_dir, "train", train_images, train_labels)

    meta = {
        "task": "tabular_mlp_classification",
        "num_classes": num_classes,
        "image_shape": [1, 8, 8],
        "n_features": n_features,
        "note": "Synthetic data for debugging. Gaussian mixture + random rotation.",
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 05-sequence: token sequences 64x64, 5 классов
# ---------------------------------------------------------------------------


def generate_sequence(seed: int = GLOBAL_SEED, dev_size: int = DEV_SIZE, train_size: int = TRAIN_SIZE):
    """Синтетические последовательности токенов 64x64, 5 классов."""
    hw = "05-sequence"
    out_dir = REPO_ROOT / "homework" / hw / "data" / "public"
    print(f"\n[{hw}] Generating synthetic sequence data ...")

    rng = np.random.default_rng(seed)
    num_classes = 5
    vocab_size = 64
    seq_len = 64

    key_tokens = [rng.choice(vocab_size, size=3, replace=False).tolist() for _ in range(num_classes)]
    key_positions = [rng.choice(seq_len, size=3, replace=False).tolist() for _ in range(num_classes)]

    def make_split(n: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(split_seed)
        per_class = np.full(num_classes, n // num_classes, dtype=int)
        per_class[:n % num_classes] += 1
        imgs_list, lbl_list = [], []
        for c in range(num_classes):
            nc = int(per_class[c])
            seqs = r.integers(0, vocab_size, size=(nc, seq_len), dtype=np.uint8)
            for tok, pos in zip(key_tokens[c], key_positions[c]):
                seqs[:, pos] = tok
            imgs = np.tile(seqs[:, np.newaxis, :], (1, seq_len, 1))
            row_jitter = r.integers(0, 8, size=(nc, seq_len, seq_len), dtype=np.uint8)
            imgs = ((imgs.astype(np.int16) + row_jitter) % vocab_size).astype(np.uint8)
            for tok, pos in zip(key_tokens[c], key_positions[c]):
                imgs[:, 0, pos] = tok
            imgs_list.append(imgs[:, np.newaxis, :, :])
            lbl_list.append(np.full(nc, c, dtype=np.int64))
        images = np.concatenate(imgs_list, axis=0)
        labels = np.concatenate(lbl_list, axis=0)
        idx = r.permutation(len(labels))
        return images[idx], labels[idx]

    dev_images, dev_labels = make_split(dev_size, seed)
    train_images, train_labels = make_split(train_size, seed + 1)

    save_split(out_dir, "dev", dev_images, dev_labels)
    save_split(out_dir, "train", train_images, train_labels)

    meta = {
        "task": "sequence_classification",
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "image_shape": [1, 64, 64],
        "key_tokens": key_tokens,
        "key_positions": key_positions,
        "note": "Synthetic data for debugging. Token sequences with class-specific key tokens.",
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 06-transformer: token documents 32x32, 3 класса
# ---------------------------------------------------------------------------


def generate_transformer(seed: int = GLOBAL_SEED, dev_size: int = DEV_SIZE, train_size: int = TRAIN_SIZE):
    """Синтетические документы с позиционными паттернами 32x32, 3 класса."""
    hw = "06-transformer"
    out_dir = REPO_ROOT / "homework" / hw / "data" / "public"
    print(f"\n[{hw}] Generating synthetic transformer data ...")

    rng = np.random.default_rng(seed)
    num_classes = 3
    vocab_size = 128
    h, w = 32, 32

    row_offsets = rng.integers(0, 5, size=num_classes).tolist()
    pattern_rows = [
        [row_offsets[0], row_offsets[0] + 1],
        [h // 2 + row_offsets[1], h // 2 + row_offsets[1] + 1],
        [h - 2 - row_offsets[2], h - 1 - row_offsets[2]],
    ]
    signature_tokens = [rng.choice(vocab_size, size=4, replace=False).tolist() for _ in range(num_classes)]
    pattern_cols = rng.choice(w, size=4, replace=False).tolist()

    def make_split(n: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(split_seed)
        per_class = np.full(num_classes, n // num_classes, dtype=int)
        per_class[:n % num_classes] += 1
        imgs_list, lbl_list = [], []
        for c in range(num_classes):
            nc = int(per_class[c])
            docs = r.integers(0, vocab_size, size=(nc, h, w), dtype=np.uint8)
            for row in pattern_rows[c]:
                row = max(0, min(h - 1, row))
                for tok, col in zip(signature_tokens[c], pattern_cols):
                    docs[:, row, col] = tok
            imgs_list.append(docs[:, np.newaxis, :, :])
            lbl_list.append(np.full(nc, c, dtype=np.int64))
        images = np.concatenate(imgs_list, axis=0)
        labels = np.concatenate(lbl_list, axis=0)
        idx = r.permutation(len(labels))
        return images[idx], labels[idx]

    dev_images, dev_labels = make_split(dev_size, seed)
    train_images, train_labels = make_split(train_size, seed + 1)

    save_split(out_dir, "dev", dev_images, dev_labels)
    save_split(out_dir, "train", train_images, train_labels)

    meta = {
        "task": "document_classification",
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "image_shape": [1, 32, 32],
        "pattern_positions": {
            str(c): {"rows": pattern_rows[c], "cols": pattern_cols, "tokens": signature_tokens[c]}
            for c in range(num_classes)
        },
        "note": "Synthetic data for debugging. Documents with positional class patterns.",
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("generate_dev_data.py — synthetic debug datasets")
    print(f"GLOBAL_SEED={GLOBAL_SEED}, dev_size={DEV_SIZE}, train_size={TRAIN_SIZE}")
    print("=" * 60)

    generate_pytorch()
    generate_mlp_norm()
    generate_sequence()
    generate_transformer()

    print("\nDone. All synthetic datasets written.")
    print("NOTE: these are synthetic datasets for local development only.")
    print("Real hidden test sets are generated per-student via `ml3 make-instance`.")
