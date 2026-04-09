---
name: ml3-grade
description: |
  Грейдер курса ml3-neo. Запусти когда нужно оценить ДЗ студента: режимы
  leaderboard (метрика), judge (LLM-судья через Swarm), both (оба).
  Принимает аргументы вида `hw=04-cnn-zoo student=alice [pr=123]
  [mode=both|leaderboard|judge]`.
---

# ml3-grade

Локальный грейдер курса `ml3-neo`. Все режимы работают через CLI `ml3`,
который установлен в pixi-окружении курса. LLM-судья использует
подписочных Swarm-агентов (Codex / Gemini), не API-ключи.

## Когда вызывать

- Когда пользователь просит оценить ДЗ студента (одной командой или
  ссылкой на PR).
- Когда нужно посмотреть текущий leaderboard.
- Когда нужно перерендерить страницу leaderboards на сайте курса.

## Что делать

1. **Парсить аргументы.** Минимум — `hw=<id>`. Опционально — `student=<name>`,
   `pr=<n>`, `mode=both|leaderboard|judge`.
2. **Перейти в корень курса.** Если ты не в `ml3-neo/`, найди его (поднимайся
   до файла `_quarto.yml`).
3. **Опционально checkout PR.** Если передан `pr=<n>` — сделай `gh pr checkout <n>`,
   запомни ветку, потом вернёшь обратно.
4. **Запустить грейдер.** В зависимости от `mode`:

   ```bash
   pixi run ml3 grade leaderboard <hw> --student <name>
   pixi run ml3 grade judge       <hw> --student <name>
   pixi run ml3 grade both        <hw> --student <name>
   ```

   По умолчанию `mode=both`. CLI сам делает три анти-чит шага:
   - `make_instance(hw, student)` — генерирует персональный hidden test из seed;
   - `verify_submission(...)` — запускает ноутбук студента на инстансе и сверяет predictions;
   - `run_judge(..., instance_description=...)` — если нет `submission/ANSWERS.md`, второй фазой генерируются probing-вопросы.
5. **Распарсить JSON** в выводе. Достать `combined.total`, `metric_points`,
   `judge_points`, `judge_multiplier`, `execution_ok`,
   `aggregated.red_flags_triggered`, `judge.status` (`final` vs `awaiting_answers`),
   `judge.probes.questions` если `awaiting_answers`.
6. **Опубликовать комментарий.** Если работали с PR — `gh pr comment <n> --body-file -`
   с компактной таблицей и ссылкой на raw-ответы судей. **Если статус
   `awaiting_answers`** — пост должен включать probing-вопросы и просьбу
   запостить `submission/ANSWERS.md`; пометь оценку как промежуточную
   (× 0.5 к judge).
7. **Перерендерить leaderboard:** `pixi run ml3 render-leaderboards`.
8. **Вернуть бранч обратно**, если делали checkout PR.

## Формат комментария к PR

Финальный (есть ANSWERS.md):

```markdown
## ml3-grade — <hw>

| Component | Score |
|---|---:|
| metric (capped) | X.X / 60 |
| judge | X.X / 40 |
| execution | ok / fail |
| **TOTAL** | **X.X / 100** |

🚩 Red flags: <list or "none">

— ml3-grade · agents: codex, gemini · ensemble: median
```

Промежуточный (`status: awaiting_answers`):

```markdown
## ml3-grade — <hw> (промежуточный)

| Component | Score |
|---|---:|
| metric (capped) | X.X / 60 |
| judge × 0.5 | X.X / 40 |
| execution | ok / fail |
| **TOTAL (provisional)** | **X.X / 100** |

### Probing-вопросы

1. ...
2. ...
3. ...

Ответьте в `submission/ANSWERS.md` и перегрейдите. Без ответов judge-балл
останется × 0.5.
```

## Что НЕ делать

- Не вызывай Anthropic API напрямую — судьи только через `swarm`/`codex`/`gemini` CLI.
- Не редактируй файлы из `homework/<hw>/` (eval.py, meta.yml, rubric.yml,
  tests/, data/hidden/) — они контролируются курсом, не PR'ом студента.
- Не запускай грейдер в GitHub Actions: Swarm требует подписочные CLI,
  CI запустить не сможет.
- Не сохраняй raw-ответы судей в репозитории, только в `grader/.cache/` (gitignored).

## Если что-то сломалось

- `pixi run ml3 --help` — проверка, что CLI вообще установлен.
- `which swarm codex gemini` — проверка, что есть хотя бы один агент.
- Если `_extract_json` падает — посмотри raw-вывод агента в результате,
  возможно, агент вернул не-JSON. Попробуй другого агента или второй прогон.
- Если `eval.py` не находит hidden test — путь должен быть смонтирован
  на машине инструктора по `meta.yml.leaderboard.hidden_data`.
