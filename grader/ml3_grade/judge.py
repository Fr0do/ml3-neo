"""LLM-as-judge через подписочных Swarm-агентов.

Архитектура: грейдер не делает API-вызовов сам. Вместо этого он спавнит
агентов через `swarm` CLI (или `mcp__Swarm__Spawn`, если запускается из
Claude Code). Каждый агент получает один промпт с вшитым rubric, diff'ом
сабмита и содержимым ноутбука. Возвращает JSON по схеме из rubric.yml.

Несколько агентов (Codex + Gemini) запускаются параллельно, итог берётся
как медиана по `total` и пер-критерию.

Почему так:
- API-ключи не нужны (используются подписки),
- разные LLM компенсируют шум друг друга,
- инструктор может в любой момент посмотреть raw-ответы каждого агента.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .config import load_rubric


def run_judge(
    root: Path,
    hw_id: str,
    meta: dict,
    submission: Path,
    student: str,
    *,
    instance_description: dict | None = None,
) -> dict:
    """Двухфазный judge.

    Phase 1: ансамбль-оценка по rubric (прежнее поведение). В промпт
    инжектится описание персонального инстанса и, если есть, ANSWERS.md —
    так judge видит, отвечал ли студент на probing-вопросы.

    Phase 2 (только если ANSWERS.md отсутствует): генерируем probing-вопросы
    и возвращаем их вместе с результатом. Статус `awaiting_answers` значит,
    что это промежуточный грейд — инструктор должен запостить вопросы в PR,
    дождаться ответов студента, и перегрейдить повторно.

    Финальный judge-балл дополнительно уменьшается на 0.5× через
    scoring.combine_scores, если ANSWERS.md так и не пришёл после probing.
    """
    rubric = load_rubric(root, hw_id)
    judge_cfg = meta.get("judge", {})

    # Импорт сюда, чтобы не было циклов: probe импортирует _spawn_agent.
    from .probe import collect_answers, generate_probes

    answers_md = collect_answers(submission)

    agents = judge_cfg.get("agents", ["codex"])
    ensemble = judge_cfg.get("ensemble", "median")
    prompt = _build_prompt(
        root, hw_id, meta, rubric, submission,
        instance_description=instance_description,
        answers_md=answers_md,
    )

    raw_responses: list[dict[str, Any]] = []
    for agent in agents:
        try:
            resp = _spawn_agent(agent, prompt)
        except Exception as e:  # noqa: BLE001 — мы хотим продолжить ensemble
            raw_responses.append({"agent": agent, "ok": False, "error": str(e)})
            continue
        raw_responses.append({"agent": agent, "ok": True, "response": resp})

    parsed = [r["response"] for r in raw_responses if r.get("ok")]
    if not parsed:
        return {
            "ok": False,
            "error": "all judge agents failed",
            "raw": raw_responses,
        }

    aggregated = _aggregate(parsed, ensemble)

    status = "final" if answers_md else "awaiting_answers"
    probes: dict | None = None
    if not answers_md:
        probes = generate_probes(
            root, hw_id, meta, submission, instance_description,
            agent=agents[0] if agents else "codex",
        )

    return {
        "ok": True,
        "student": student,
        "agents_used": [r["agent"] for r in raw_responses if r.get("ok")],
        "ensemble": ensemble,
        "aggregated": aggregated,
        "status": status,
        "answered": answers_md is not None,
        "probes": probes,
        "raw": raw_responses,
    }


# ---------------------------------------------------------------------------
# Промпт
# ---------------------------------------------------------------------------

PROMPT_HEADER = (
    "Ты — независимый эксперт-судья учебных работ по deep learning. "
    "Тебе передан rubric (открытый, его видит и студент), а также материалы "
    "сабмита. Твоя задача — выставить оценки строго по критериям rubric и "
    "вернуть результат в указанном JSON-формате. Не выдумывай критерии, "
    "которых нет в rubric. Не оценивай то, что ты не видишь в материалах."
)


def _build_prompt(
    root: Path,
    hw_id: str,
    meta: dict,
    rubric: dict,
    submission: Path,
    *,
    instance_description: dict | None = None,
    answers_md: str | None = None,
) -> str:
    parts: list[str] = [PROMPT_HEADER, ""]

    parts.append("=" * 60)
    parts.append(f"## ДЗ: {hw_id} — {meta.get('title', '')}")
    parts.append(f"## Тип: {meta.get('type')}, вес: {meta.get('weight')}")
    parts.append("")
    parts.append("=" * 60)
    parts.append("## RUBRIC")
    parts.append("```yaml")
    import yaml
    parts.append(yaml.safe_dump(rubric, allow_unicode=True, sort_keys=False))
    parts.append("```")

    if instance_description is not None:
        parts.append("=" * 60)
        parts.append("## ИНСТАНС СТУДЕНТА (персональный)")
        parts.append(
            "У каждого студента свой hidden test и свой data twist. Учитывай "
            "это при оценке: ablations студента должны соотноситься именно "
            "с его инстансом, а не с абстрактной постановкой."
        )
        parts.append("```json")
        import json as _json
        parts.append(_json.dumps(instance_description, indent=2, ensure_ascii=False))
        parts.append("```")

    parts.append("=" * 60)
    parts.append("## MODEL.md (защита решения)")
    model_md = submission / "submission" / "MODEL.md"
    parts.append(model_md.read_text() if model_md.exists() else "*отсутствует*")

    parts.append("=" * 60)
    parts.append("## notebook.py (код решения)")
    nb = submission / "submission" / "notebook.py"
    if nb.exists():
        text = nb.read_text()
        if len(text) > 60_000:
            text = text[:60_000] + "\n\n... (truncated)"
        parts.append("```python")
        parts.append(text)
        parts.append("```")
    else:
        parts.append("*отсутствует*")

    if answers_md is not None:
        parts.append("=" * 60)
        parts.append("## ANSWERS.md — ответы студента на probing-вопросы")
        parts.append(
            "Эти ответы — второй раунд защиты. Оцени их на предмет того, "
            "действительно ли студент понимает свой код и свой инстанс, "
            "или отвечает общими словами. Если ответы пустые/общие — "
            "опусти judge_points сильно вниз."
        )
        parts.append(answers_md)

    parts.append("=" * 60)
    parts.append("## ИНСТРУКЦИЯ ВЫВОДА")
    parts.append(rubric.get("rubric", {}).get("output_format", "").strip())

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Спавн агентов
# ---------------------------------------------------------------------------

def _spawn_agent(agent: str, prompt: str) -> dict:
    """Запускает один Swarm-агент и парсит JSON-ответ.

    Сначала пробуем `swarm` CLI. Если его нет — fallback на прямой вызов
    `codex` / `gemini` CLI с stdin. Если и их нет — поднимаем исключение.
    """
    if shutil.which("swarm"):
        return _spawn_via_swarm(agent, prompt)

    if shutil.which(agent):
        return _spawn_via_native_cli(agent, prompt)

    raise RuntimeError(
        f"neither `swarm` nor `{agent}` found in PATH; "
        "install one of them or remove this agent from meta.yml"
    )


def _spawn_via_swarm(agent: str, prompt: str) -> dict:
    """Через `swarm spawn --agent <name> --prompt-file ...`.

    Точная сигнатура зависит от версии swarm CLI; здесь — наиболее общий
    интерфейс. Если у вас другой синтаксис — подправьте этот метод.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(prompt)
        prompt_path = f.name

    try:
        proc = subprocess.run(
            ["swarm", "spawn",
             "--agent", agent,
             "--prompt-file", prompt_path,
             "--format", "json",
             "--non-interactive"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"swarm spawn failed: {proc.stderr.strip()[:500]}"
            )
        return _extract_json(proc.stdout)
    finally:
        Path(prompt_path).unlink(missing_ok=True)


def _spawn_via_native_cli(agent: str, prompt: str) -> dict:
    """Fallback: прямой вызов CLI агента (codex/gemini) с stdin."""
    proc = subprocess.run(
        [agent, "--non-interactive"],
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{agent} CLI failed: {proc.stderr.strip()[:500]}"
        )
    return _extract_json(proc.stdout)


def _extract_json(text: str) -> dict:
    """Пытаемся вытащить JSON-блок из ответа агента.

    Агенты иногда оборачивают ответ в ```json ... ``` или добавляют
    преамбулу. Берём самый длинный сбалансированный {...}-блок.
    """
    text = text.strip()
    # быстрый путь
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ищем сбалансированный блок
    best = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        parsed = json.loads(candidate)
                        if best is None or len(candidate) > len(best[0]):
                            best = (candidate, parsed)
                    except json.JSONDecodeError:
                        pass
                    break
    if best is None:
        raise RuntimeError("no valid JSON in agent response")
    return best[1]


# ---------------------------------------------------------------------------
# Агрегация
# ---------------------------------------------------------------------------

def _aggregate(parsed: list[dict], ensemble: str) -> dict:
    """Сводим несколько ответов в одну оценку."""
    totals = [float(p.get("total", 0)) for p in parsed]

    all_scores: dict[str, list[float]] = {}
    for p in parsed:
        for cid, val in (p.get("scores") or {}).items():
            try:
                all_scores.setdefault(cid, []).append(float(val))
            except (TypeError, ValueError):
                pass

    def reduce(values: list[float]) -> float:
        if not values:
            return 0.0
        if ensemble == "median":
            return float(statistics.median(values))
        if ensemble == "mean":
            return float(statistics.fmean(values))
        if ensemble == "min":
            return float(min(values))
        return float(statistics.median(values))

    red_flags: set[str] = set()
    for p in parsed:
        red_flags.update(p.get("red_flags_triggered") or [])

    return {
        "scores": {cid: reduce(vals) for cid, vals in all_scores.items()},
        "total": reduce(totals),
        "red_flags_triggered": sorted(red_flags),
        "n_judges": len(parsed),
    }
