"""Генератор probing-вопросов для двухфазного judge.

После первичной оценки по rubric мы спавним ещё один agent-вызов, который
получает код студента, MODEL.md и description личного инстанса. Его
задача — выдать 2–3 вопроса, которые:

1. Укоренены в специфике *этого* инстанса (классы, которые дропнуты,
   перекос в распределении, конкретные числа из MODEL.md).
2. Проверяют понимание, а не могут быть ответом на стандартной LLM.
3. Отличимы от общих «а расскажите про BatchNorm».

Примеры того, что мы хотим получить:
- «В вашем test-инстансе класс 17 имеет всего 42 примера. В MODEL.md вы
  пишете, что использовали class-balanced loss. Каким был mean recall на
  dev-сете для класса 17 без class-balancing, и какое ablation в вашем
  notebook это показывает?»
- «Ваша шапка [stage4 → avg pool → fc] выдаёт 256-мерные features.
  Почему выбор 256, а не 512, при batch_size=256?»

Если студент не прислал ANSWERS.md на эти вопросы в течение дедлайна —
judge_points умножается на 0.5 (см. scoring.py).

Вопросы НЕ хранятся в rubric.yml, потому что зависят от сабмита — у
каждого студента они уникальны. Это часть защиты от списывания.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .judge import _extract_json, _spawn_agent

PROBE_SYSTEM = """\
Ты — экзаменатор по deep learning. Твоя задача — на основе кода, MODEL.md
и описания ИНДИВИДУАЛЬНОГО ИНСТАНСА ДЗ студента сгенерировать 2-3
probing-вопроса, ответы на которые:

1. НЕВОЗМОЖНО нагуглить или сгенерировать общим LLM без доступа к этому
   конкретному коду, MODEL.md и инстансу.
2. Проверяют, делал ли студент свои эксперименты, или скопировал чужое.
3. Требуют числовых ответов, ссылок на конкретные ячейки ноутбука или
   графики — не общих рассуждений.

Жёсткие правила:
- Не спрашивай общих вещей ("объясните, что такое BatchNorm").
- Опирайся на конкретные числа, которые видишь в коде / MODEL.md /
  описании инстанса.
- Если studen-instance имеет dropped_classes = [17, 23, 31] — вопрос
  может быть про поведение модели на этих классах.
- Формулируй чётко, одним абзацем на вопрос.

Верни строго JSON:
{
  "questions": [
    "<вопрос 1>",
    "<вопрос 2>",
    "<вопрос 3>"
  ],
  "reasoning": "<краткое объяснение, почему именно эти вопросы>"
}
"""


def generate_probes(
    root: Path,
    hw_id: str,
    meta: dict,
    submission: Path,
    instance_description: dict | None,
    agent: str = "codex",
) -> dict:
    """Возвращает {questions: [...], reasoning: str, agent: str, ok: bool}."""
    prompt = _build_probe_prompt(hw_id, meta, submission, instance_description)
    try:
        resp = _spawn_agent(agent, prompt)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "agent": agent}
    questions = resp.get("questions") or []
    reasoning = resp.get("reasoning", "")
    return {
        "ok": True,
        "agent": agent,
        "questions": [q for q in questions if isinstance(q, str) and q.strip()],
        "reasoning": reasoning,
    }


def _build_probe_prompt(
    hw_id: str,
    meta: dict,
    submission: Path,
    instance_description: dict | None,
) -> str:
    parts: list[str] = [PROBE_SYSTEM, ""]
    parts.append("=" * 60)
    parts.append(f"## ДЗ: {hw_id} — {meta.get('title', '')}")

    parts.append("=" * 60)
    parts.append("## ИНСТАНС СТУДЕНТА")
    if instance_description:
        parts.append("```json")
        parts.append(json.dumps(instance_description, indent=2, ensure_ascii=False))
        parts.append("```")
    else:
        parts.append("*инстанс не сгенерирован — probing опирается только на код и MODEL.md*")

    parts.append("=" * 60)
    parts.append("## MODEL.md")
    model_md = submission / "submission" / "MODEL.md"
    parts.append(model_md.read_text() if model_md.exists() else "*отсутствует*")

    parts.append("=" * 60)
    parts.append("## notebook.py")
    nb = submission / "submission" / "notebook.py"
    if nb.exists():
        text = nb.read_text()
        if len(text) > 50_000:
            text = text[:50_000] + "\n\n... (truncated)"
        parts.append("```python")
        parts.append(text)
        parts.append("```")
    else:
        parts.append("*отсутствует*")

    parts.append("=" * 60)
    parts.append("Сгенерируй JSON с 2-3 probing-вопросами по правилам выше.")
    return "\n".join(parts)


def collect_answers(submission: Path) -> str | None:
    """Читает submission/ANSWERS.md если он существует."""
    answers = submission / "submission" / "ANSWERS.md"
    if answers.exists() and len(answers.read_text().strip()) > 20:
        return answers.read_text()
    return None
