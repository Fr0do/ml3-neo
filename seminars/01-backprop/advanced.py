"""Seminar 01 — Backprop (advanced level).

Marimo notebook. Запуск:
    marimo edit seminars/01-backprop/advanced.py

Цель (advanced):
    Расширить autograd до работы с numpy массивами (тензорами), 
    добавить mini-batches, реализовать SGD и Adam, cross_entropy loss.
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
        # Семинар 01 — Backprop (advanced)

        Перед запуском прочитайте [лекцию](../../lectures/01-backprop/lecture.qmd).
        
        В `basic` версии мы работали со скалярами. В глубоком обучении всё работает с батчами
        и матрицами. В этом ноутбуке вам предстоит написать `Tensor` класс поверх `numpy`,
        реализовать базовые тензорные операции и оптимизатор Adam.
        """
    )
    return


@app.cell
def __():
    import numpy as np
    return (np,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Шаг 1: Класс Tensor
        
        Реализуйте класс `Tensor`, который оборачивает `np.ndarray` и вычисляет градиенты 
        по правилам матричного дифференцирования.
        """
    )
    return


@app.cell
def __(np):
    class Tensor:
        def __init__(self, data, _children=(), _op=''):
            self.data = np.array(data, dtype=np.float32)
            self.grad = np.zeros_like(self.data)
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op

        def __repr__(self):
            return f"Tensor(shape={self.data.shape})"

        # TODO: Добавить +, *, @ (matmul), sum
        # Не забудьте про broadcast градиентов при сложении!
        
        def backward(self):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)

            self.grad = np.ones_like(self.data)
            for node in reversed(topo):
                node._backward()
    return (Tensor,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Шаг 2: Adam Optimizer
        
        Реализуйте оптимизатор Adam вручную. Он должен хранить скользящие средние 
        градиентов и их квадратов (моменты).
        """
    )
    return


@app.cell
def __(np):
    # TODO: реализация Adam
    class Adam:
        def __init__(self, parameters, lr=0.001, b1=0.9, b2=0.999):
            self.parameters = parameters
            self.lr = lr
            self.b1 = b1
            self.b2 = b2
            self.t = 0
            self.m = [np.zeros_like(p.data) for p in parameters]
            self.v = [np.zeros_like(p.data) for p in parameters]

        def zero_grad(self):
            for p in self.parameters:
                p.grad = np.zeros_like(p.data)

        def step(self):
            self.t += 1
            # TODO: обновить веса
            pass
    return (Adam,)


if __name__ == "__main__":
    app.run()
