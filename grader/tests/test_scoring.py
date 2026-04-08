"""Тесты для combine_scores — единственного кусочка с числовой логикой."""

from __future__ import annotations

from ml3_grade.scoring import combine_scores


META = {
    "weight": 100,
    "leaderboard": {
        "metric_weight": 60,
        "baseline_score": 0.25,
        "full_credit_score": 0.65,
    },
    "judge": {"weight": 40},
}


def test_metric_below_baseline_zero():
    lb = {"ok": True, "score": 0.10}
    jg = {"ok": True, "aggregated": {"total": 30}}
    out = combine_scores(META, lb, jg)
    assert out["metric_points"] == 0.0
    assert out["judge_points"] == 30.0


def test_metric_at_full_credit():
    lb = {"ok": True, "score": 0.65}
    jg = {"ok": True, "aggregated": {"total": 40}}
    out = combine_scores(META, lb, jg)
    assert out["metric_points"] == 60.0
    assert out["total"] == 100.0


def test_metric_above_full_credit_caps():
    lb = {"ok": True, "score": 0.99}
    jg = {"ok": True, "aggregated": {"total": 40}}
    out = combine_scores(META, lb, jg)
    assert out["metric_points"] == 60.0  # capped


def test_red_flags_zero_judge():
    lb = {"ok": True, "score": 0.65}
    jg = {
        "ok": True,
        "aggregated": {"total": 40, "red_flags_triggered": ["leakage"]},
    }
    out = combine_scores(META, lb, jg)
    assert out["judge_points"] == 0.0
    assert out["total"] == 60.0


def test_judge_caps_to_weight():
    lb = {"ok": True, "score": 0.65}
    jg = {"ok": True, "aggregated": {"total": 999}}
    out = combine_scores(META, lb, jg)
    assert out["judge_points"] == 40.0


def test_metric_linear_midpoint():
    lb = {"ok": True, "score": 0.45}  # midpoint между 0.25 и 0.65
    jg = {"ok": True, "aggregated": {"total": 0}}
    out = combine_scores(META, lb, jg)
    assert abs(out["metric_points"] - 30.0) < 1e-6


def test_execution_failure_zeroes_metric():
    """Если ноутбук не воспроизводится → metric_points = 0 (анти-чит)."""
    lb = {"ok": True, "score": 0.65}
    jg = {"ok": True, "aggregated": {"total": 40}}
    execution = {"ok": False, "reason": "agreement 0.42 < 0.98"}
    out = combine_scores(META, lb, jg, execution=execution)
    assert out["metric_points"] == 0.0
    assert out["execution_ok"] is False
    assert out["judge_points"] == 40.0  # judge не тронут
    assert out["total"] == 40.0


def test_execution_ok_keeps_metric():
    lb = {"ok": True, "score": 0.65}
    jg = {"ok": True, "aggregated": {"total": 40}}
    execution = {"ok": True, "reason": "reproducible"}
    out = combine_scores(META, lb, jg, execution=execution)
    assert out["metric_points"] == 60.0
    assert out["execution_ok"] is True
    assert out["total"] == 100.0


def test_awaiting_answers_halves_judge():
    """Probing-вопросы без ANSWERS.md → judge × 0.5."""
    lb = {"ok": True, "score": 0.65}
    jg = {
        "ok": True,
        "status": "awaiting_answers",
        "aggregated": {"total": 40},
    }
    out = combine_scores(META, lb, jg)
    assert out["judge_points"] == 20.0
    assert out["judge_multiplier"] == 0.5
    assert out["total"] == 80.0


def test_final_status_full_judge():
    lb = {"ok": True, "score": 0.65}
    jg = {
        "ok": True,
        "status": "final",
        "aggregated": {"total": 40},
    }
    out = combine_scores(META, lb, jg)
    assert out["judge_points"] == 40.0
    assert out["judge_multiplier"] == 1.0
