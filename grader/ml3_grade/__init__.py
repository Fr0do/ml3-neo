"""ml3-grade — локальный CLI грейдер курса ml3-neo.

Два режима:
- leaderboard: запускает eval.py на скрытом тесте, обновляет leaderboards.json.
- judge: спавнит Swarm-агентов (Codex/Gemini) с rubric, ensemble-медиана.

Запускается ТОЛЬКО локально на машине инструктора. В CI не идёт, потому
что Swarm требует подписочные CLI.
"""

__version__ = "0.1.0"
