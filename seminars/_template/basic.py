"""Seminar NN — <тема> (basic level).

Marimo notebook. Запуск:
    marimo edit seminars/<id>/basic.py

Шаблон. Скопируйте в новый модуль и заполните клетки.
Скаффолдинг: высокий, заполняемые TODO-куски, фиксированный стек.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Семинар NN — <тема> (basic)

        Перед запуском прочитайте [лекцию](../../lectures/<id>/lecture.qmd).
        """
    )
    return


@app.cell
def __():
    # TODO: импорты
    return


@app.cell
def __(mo):
    mo.md(r"## Шаг 1 — TODO")
    return


@app.cell
def __():
    # TODO: код шага 1
    return


@app.cell
def __(mo):
    mo.md(r"## TODO для студента (basic)")
    return


if __name__ == "__main__":
    app.run()
