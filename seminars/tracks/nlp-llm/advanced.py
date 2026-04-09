"""Seminar NLP — LLM (advanced level).

Optional track: nlp-llm.
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
        # Семинар NLP: LLMs (advanced)

        ## Задача

        1. Реализовать BPE токенизатор с нуля на корпусе (например, на стихах Пушкина), 
           построить vocab, encode/decode.
        2. Показать KV-cache trick: сравнить время генерации с кешем и без на GPT-2 
           через библиотеку `transformers`.

        ## Чек-лист реализации

        - [ ] Скачать или определить текстовый корпус.
        - [ ] Написать цикл слияния самых частых пар символов (BPE).
        - [ ] Реализовать функции `encode` и `decode` с полученным словарем.
        - [ ] Загрузить `gpt2` через `transformers`.
        - [ ] Написать цикл авторегрессионной генерации, включить таймер с `use_cache=True` и `use_cache=False`.
        """
    )
    return

@app.cell
def __():
    # === SUBMISSION (start) ===
    import time
    from collections import Counter
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # TODO: Реализовать BPE (train, encode, decode)
    
    # TODO: Загрузить gpt2
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelForCausalLM.from_pretrained("gpt2")

    # TODO: Сравнить скорость генерации с KV-cache и без
    pass
    # === SUBMISSION (end) ===
    return

if __name__ == "__main__":
    app.run()
