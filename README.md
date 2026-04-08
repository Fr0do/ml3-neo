# ml3-neo

Курс машинного обучения для магистратур по data science. Наследник `SergeyMalashenko/2025_ML3` с лекционной частью на основе `Dyakonov/DL_BOOK`.

**Дизайн-принципы:**

1. **Непрерывный UX** — лекция, семинар и ДЗ живут в одном Quarto-сайте, делят общую навигацию, нотацию и набор TikZ-схем.
2. **Интерактивность** — лекции на Quarto с raw LaTeX/TikZ (формулы и схемы редактируются как `.tex`-исходники), семинары и ДЗ — реактивные [Marimo](https://marimo.io) ноутбуки.
3. **Open-ended ДЗ с автогрейдингом** — каждое ДЗ имеет либо публичную метрику (leaderboard), либо rubric для LLM-судьи (Swarm-агенты Codex/Gemini), либо оба режима. Ручная проверка не требуется.
4. **Двухуровневость** — `basic` и `advanced` версии каждого модуля делят лекцию, расходятся в глубине семинара и сложности ДЗ.

## Структура

```
lectures/        Quarto .qmd с LaTeX/TikZ → HTML + PDF
seminars/        Marimo .py (basic + advanced)
homework/        Marimo starter + meta.yml + rubric.yml + eval.py
figures/tikz/    Общий каталог TikZ-схем (BN/LN/IN/GN, attention, ...)
grader/          Локальный CLI ml3-grade (leaderboard + judge через Swarm)
```

## Быстрый старт

```bash
pixi install
pixi run preview              # quarto preview
pixi run seminar 04-cnn basic # marimo edit seminars/04-cnn/basic.py
pixi run grade 04-cnn-zoo     # ml3 grade both 04-cnn-zoo
```

## Контентная схема

**Core (8 модулей, обязательны):** backprop → PyTorch → MLP+normalization → CNN → sequences → transformer → pretraining/LoRA → generative intro.

**Optional tracks:** CV / NLP-LLM / Efficient ML / Generative / Graphs / RL.

См. [`syllabus.qmd`](syllabus.qmd).

## Reference-модуль

Полностью реализован модуль **04-cnn** — он же шаблон для остальных. Все файлы помечены `← REFERENCE` в плане проекта.

## Лицензия и источники

Лекционная часть опирается на материалы А.Г. Дьяконова ([DL_BOOK](https://github.com/Dyakonov/DL_BOOK), [DL](https://github.com/Dyakonov/DL)) и стек семинаров С. Малашенко ([2025_ML3](https://github.com/SergeyMalashenko/2025_ML3)). Используется в образовательных целях.
