"""Seminar NN — <тема> (advanced level).

Шаблон. Скаффолдинга минимум, открытая постановка, ссылка на ДЗ как на
естественное продолжение.
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
        # Семинар NN — <тема> (advanced)

        ## Задача

        TODO: одна-две короткие, открытые постановки.

        ## Чек-лист реализации

        - [ ] TODO
        - [ ] TODO
        - [ ] TODO
        """
    )
    return


@app.cell
def __():
    # === SUBMISSION (start) ===
    # Здесь — ваш код. Грейдер исполняет всё между маркерами.
    pass
    # === SUBMISSION (end) ===
    return


if __name__ == "__main__":
    app.run()
