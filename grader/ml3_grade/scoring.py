"""Combine leaderboard score + judge score + execution report → final points.

Формула живёт здесь, чтобы и студент, и грейдер видели одинаковую логику
без копипасты по JSON.

Анти-чит модификаторы:
- если execution verifier вернул `ok=False` → `metric_points = 0`
  (метрика опирается на predictions.npy, который не воспроизводится —
  значит, нет рабочего кода);
- если judge находится в статусе `awaiting_answers` (probing-вопросы
  сгенерированы, ANSWERS.md отсутствует) → judge_points × 0.5;
- если judge поднял red flag → judge_points = 0.
"""

from __future__ import annotations


def combine_scores(
    meta: dict,
    lb: dict,
    jg: dict,
    execution: dict | None = None,
) -> dict:
    weight_total = float(meta.get("weight", 100))
    lb_cfg = meta.get("leaderboard", {}) or {}
    jg_cfg = meta.get("judge", {}) or {}

    metric_w = float(lb_cfg.get("metric_weight", 0))
    judge_w = float(jg_cfg.get("weight", 0))

    # ── метрический пол ────────────────────────────────────────────────────
    metric_points = 0.0
    if lb.get("ok"):
        score = float(lb.get("score", 0))
        baseline = float(lb_cfg.get("baseline_score", 0))
        full = float(lb_cfg.get("full_credit_score", 1))
        if full > baseline:
            ratio = max(0.0, min(1.0, (score - baseline) / (full - baseline)))
        else:
            ratio = 1.0 if score >= baseline else 0.0
        metric_points = ratio * metric_w

    # ── execution verifier ────────────────────────────────────────────────
    execution_ok = True
    if execution is not None:
        execution_ok = bool(execution.get("ok"))
        if not execution_ok:
            metric_points = 0.0  # predictions не воспроизвелись → нет пола

    # ── judge ──────────────────────────────────────────────────────────────
    judge_points = 0.0
    judge_multiplier = 1.0
    if jg.get("ok"):
        agg = jg.get("aggregated", {}) or {}
        if agg.get("red_flags_triggered"):
            judge_points = 0.0
        else:
            judge_points = float(agg.get("total", 0))
            judge_points = min(judge_points, judge_w)

            # Awaiting probing answers → half credit.
            if jg.get("status") == "awaiting_answers":
                judge_multiplier = 0.5
            judge_points *= judge_multiplier

    bonus_points = 0.0  # реальные бонусы за top-N считает render-leaderboards
    total = min(weight_total, metric_points + judge_points + bonus_points)

    return {
        "metric_points": metric_points,
        "judge_points": judge_points,
        "judge_multiplier": judge_multiplier,
        "bonus_points": bonus_points,
        "execution_ok": execution_ok,
        "total": total,
    }
