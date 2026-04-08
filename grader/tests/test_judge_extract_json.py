"""Тесты для _extract_json — он часто оказывается источником багов."""

from ml3_grade.judge import _extract_json


def test_pure_json():
    out = _extract_json('{"total": 30}')
    assert out == {"total": 30}


def test_fenced_json():
    text = """Вот моя оценка:

```json
{"scores": {"a": 5}, "total": 5}
```
"""
    out = _extract_json(text)
    assert out == {"scores": {"a": 5}, "total": 5}


def test_with_preamble_and_postamble():
    text = (
        "Размышления... \n"
        '{"total": 12, "scores": {"x": 3}}\n'
        "Дополнительный текст после."
    )
    out = _extract_json(text)
    assert out["total"] == 12


def test_picks_largest_balanced_block():
    # Сначала маленький, потом большой — должен взять большой.
    text = '{"a":1} ... {"a":1, "b":{"c":2}, "d": 3}'
    out = _extract_json(text)
    assert out == {"a": 1, "b": {"c": 2}, "d": 3}
