"""Starter Marimo notebook for hw 01-backprop.

Скопируйте этот файл в submission/notebook.py и работайте там. Грейдер
исполняет именно `submission/notebook.py`.

Маркеры === SUBMISSION (start) === / === SUBMISSION (end) === говорят
грейдеру, какие ячейки оценивать.
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
        # ДЗ 01 — Backprop (стартер)

        Цель — написать `Value` или `Tensor` autograd-движок с нуля 
        и обучить с его помощью MLP.

        Прочитайте [README](README.qmd) и [`rubric.yml`](rubric.yml) перед началом работы.
        """
    )
    return


@app.cell
def __():
    # === SUBMISSION (start) ===
    # 1. Ваш класс Value (или Tensor)
    # 2. Ваш код численной проверки градиентов (numerical check)
    # 3. Ваша архитектура сети (Module, Linear, MLP)
    # 4. Training loop
    
    import json
    import random
    
    # Сохраните финальные результаты и loss curve в results.json 
    # (опционально, но полезно для визуализации)
    results = {
        "loss_curve": [],
        "final_accuracy": 0.0
    }
    with open("submission/results.json", "w") as f:
        json.dump(results, f)
        
    pass
    # === SUBMISSION (end) ===
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Чек-лист перед сабмитом

        - [ ] `submission/notebook.py` запускается чисто на свежем окружении.
        - [ ] Написана численная проверка градиентов.
        - [ ] `submission/MODEL.md` написан и объясняет вашу реализацию.
        - [ ] Локально прогнан `pytest homework/01-backprop/tests/`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
