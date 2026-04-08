"""ml3-grade CLI entrypoint.

Команды:
    ml3 grade leaderboard <hw> [--submission PATH] [--student NAME]
    ml3 grade judge       <hw> [--submission PATH] [--student NAME]
    ml3 grade both        <hw> [--submission PATH] [--student NAME]
    ml3 render-leaderboards
    ml3 fetch-data <hw>

Все пути относительно корня курса (где лежит _quarto.yml).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import find_course_root, load_meta
from .execute import verify_submission
from .instances import Instance, make_instance
from .judge import run_judge
from .leaderboard import run_leaderboard
from .leaderboard_render import render_state
from .scoring import combine_scores

app = typer.Typer(
    help="ml3-grade — локальный грейдер курса ml3-neo",
    no_args_is_help=True,
    add_completion=False,
)
grade_app = typer.Typer(help="Запуск грейдинга для одного ДЗ")
app.add_typer(grade_app, name="grade")

console = Console()


def _maybe_make_instance(root: Path, hw: str, student: str) -> Optional[Instance]:
    """Пытается сгенерировать персональный инстанс.

    Если для hw не зарегистрирован генератор — возвращает None, грейдер
    откатывается на глобальный hidden_data из meta.yml.
    """
    try:
        return make_instance(root, hw, student)
    except SystemExit:
        return None
    except FileNotFoundError as e:
        console.print(f"[yellow]instance skipped[/yellow]: {e}")
        return None


@grade_app.command("leaderboard")
def cmd_leaderboard(
    hw: str = typer.Argument(..., help="ID домашки, например 04-cnn-zoo"),
    submission: Optional[Path] = typer.Option(
        None, "--submission", "-s",
        help="Путь к каталогу с submission/. По умолчанию — homework/<hw>/",
    ),
    student: str = typer.Option("local", "--student", help="Имя для leaderboard"),
    skip_execute: bool = typer.Option(
        False, "--skip-execute", help="Пропустить execution verifier (debug)",
    ),
) -> None:
    root = find_course_root()
    meta = load_meta(root, hw)
    sub = submission or (root / "homework" / hw)

    instance = _maybe_make_instance(root, hw, student)
    hidden = instance.hidden_path if instance else None

    execution = None
    if not skip_execute and hidden is not None:
        execution = verify_submission(root, hw, sub, hidden).to_dict()

    lb = run_leaderboard(root, hw, meta, sub, student, hidden_override=hidden)
    console.print_json(data={"leaderboard": lb, "execution": execution})


@grade_app.command("judge")
def cmd_judge(
    hw: str = typer.Argument(..., help="ID домашки"),
    submission: Optional[Path] = typer.Option(None, "--submission", "-s"),
    student: str = typer.Option("local", "--student"),
) -> None:
    root = find_course_root()
    meta = load_meta(root, hw)
    sub = submission or (root / "homework" / hw)

    instance = _maybe_make_instance(root, hw, student)
    instance_description = instance.description if instance else None

    result = run_judge(
        root, hw, meta, sub, student,
        instance_description=instance_description,
    )
    console.print_json(data=result)


@grade_app.command("both")
def cmd_both(
    hw: str = typer.Argument(...),
    submission: Optional[Path] = typer.Option(None, "--submission", "-s"),
    student: str = typer.Option("local", "--student"),
    skip_execute: bool = typer.Option(False, "--skip-execute"),
) -> None:
    root = find_course_root()
    meta = load_meta(root, hw)
    sub = submission or (root / "homework" / hw)

    instance = _maybe_make_instance(root, hw, student)
    hidden = instance.hidden_path if instance else None
    instance_description = instance.description if instance else None

    execution = None
    if not skip_execute and hidden is not None:
        execution = verify_submission(root, hw, sub, hidden).to_dict()

    lb = run_leaderboard(root, hw, meta, sub, student, hidden_override=hidden)
    jg = run_judge(
        root, hw, meta, sub, student,
        instance_description=instance_description,
    )
    combined = combine_scores(meta, lb, jg, execution=execution)

    table = Table(title=f"ml3 grade both — {hw} — {student}", show_lines=True)
    table.add_column("Component")
    table.add_column("Score", justify="right")
    table.add_row("metric (capped)", f"{combined['metric_points']:.1f}")
    table.add_row(
        "judge",
        f"{combined['judge_points']:.1f}"
        + (f" ×{combined['judge_multiplier']}" if combined['judge_multiplier'] != 1.0 else ""),
    )
    table.add_row("bonus",           f"{combined['bonus_points']:.1f}")
    if execution is not None:
        table.add_row(
            "execution",
            "[green]ok[/green]" if combined["execution_ok"] else "[red]fail[/red]",
        )
    if jg.get("status") == "awaiting_answers":
        table.add_row("status", "[yellow]awaiting ANSWERS.md[/yellow]")
    table.add_row("[bold]TOTAL",     f"[bold]{combined['total']:.1f} / {meta['weight']}")
    console.print(table)
    console.print_json(data={
        "leaderboard": lb,
        "judge": jg,
        "execution": execution,
        "combined": combined,
    })


@app.command("render-leaderboards")
def cmd_render() -> None:
    root = find_course_root()
    out = render_state(root)
    console.print(f"[green]rendered[/green] → {out.relative_to(root)}")


@app.command("fetch-data")
def cmd_fetch(hw: str = typer.Argument(...)) -> None:
    """Заглушка для разворачивания публичных данных ДЗ."""
    root = find_course_root()
    target = root / "homework" / hw / "data" / "public"
    target.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[yellow]TODO[/yellow] реализовать скачивание данных для {hw} в {target}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
