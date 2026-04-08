# Public data for hw 04-cnn-zoo

Этот каталог разворачивается командой инструктора:

```bash
ml3 fetch-data 04-cnn-zoo
```

После этого здесь появятся:

- `train.npz` — `images (45000, 3, 64, 64) uint8`, `labels (45000,) int64`
- `dev.npz` — `images (5000, 3, 64, 64) uint8`, `labels (5000,) int64`
- `test_index.json` — массив имён тестовых картинок (порядок фиксирован)

**Скрытый тест** живёт в `data/hidden/04-cnn-zoo/test.npz` и в репозитории
не лежит — только на машине грейдера.

Источник: подмножество 40 классов из TinyImageNet, ресайз 64×64. Скрипт
выборки и сборки — `grader/ml3_grade/datasets/tiny_imagenet_subset.py`.
