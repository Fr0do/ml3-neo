"""Seminar 01 — Backprop (basic level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/01-backprop/basic.py

Цель семинара (basic):
    1. Написать свой scalar Value класс (в стиле micrograd).
    2. Реализовать базовые операции (+, *, tanh, exp, **).
    3. Обучить MLP на простой задаче (XOR) с помощью написанного движка.
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
        # Семинар 01 — Backprop (basic)

        Перед запуском прочитайте [лекцию](../../lectures/01-backprop/lecture.qmd).
        
        Сегодня мы построим движок автоматического дифференцирования (autograd) 
        для скаляров с нуля, без использования PyTorch.
        """
    )
    return


@app.cell
def __():
    import math
    import random
    return math, random


@app.cell
def __(math):
    class Value:
        """Скалярное значение, отслеживающее свой градиент."""
        def __init__(self, data, _children=(), _op='', label=''):
            self.data = float(data)
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __repr__(self):
            return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), '+')

            def _backward():
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
            out._backward = _backward
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), '*')

            def _backward():
                # TODO: Чему равны локальные градиенты умножения?
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward
            return out

        def __pow__(self, other):
            assert isinstance(other, (int, float)), "only supporting int/float powers"
            out = Value(self.data**other, (self,), f'**{other}')

            def _backward():
                # TODO: локальный градиент для степени
                self.grad += other * (self.data ** (other - 1)) * out.grad
            out._backward = _backward
            return out

        def __rmul__(self, other):
            return self * other

        def __radd__(self, other):
            return self + other

        def __sub__(self, other):
            return self + (-other)

        def __neg__(self):
            return self * -1
            
        def exp(self):
            x = self.data
            out = Value(math.exp(x), (self, ), 'exp')
            
            def _backward():
                # TODO: производная экспоненты
                self.grad += out.data * out.grad
            out._backward = _backward
            return out

        def tanh(self):
            x = self.data
            t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
            out = Value(t, (self, ), 'tanh')

            def _backward():
                # TODO: производная tanh
                self.grad += (1 - t**2) * out.grad
            out._backward = _backward
            return out

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

            self.grad = 1.0
            for node in reversed(topo):
                node._backward()
                
    return (Value,)


@app.cell
def __(Value, mo):
    mo.md(
        r"""
        ## TODO для студента: Реализация недостающих узлов
        Выше приведен каркас `Value`. Убедитесь, что вы дописали правильные
        формулы для `_backward` в `__mul__`, `__pow__`, `exp`, `tanh`.
        """
    )
    return


@app.cell
def __(Value, random):
    class Neuron:
        def __init__(self, nin):
            self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
            self.b = Value(random.uniform(-1,1))

        def __call__(self, x):
            # w * x + b
            act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
            out = act.tanh()
            return out

        def parameters(self):
            return self.w + [self.b]

    class Layer:
        def __init__(self, nin, nout):
            self.neurons = [Neuron(nin) for _ in range(nout)]

        def __call__(self, x):
            outs = [n(x) for n in self.neurons]
            return outs[0] if len(outs) == 1 else outs

        def parameters(self):
            return [p for neuron in self.neurons for p in neuron.parameters()]

    class MLP:
        def __init__(self, nin, nouts):
            sz = [nin] + nouts
            self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def parameters(self):
            return [p for layer in self.layers for p in layer.parameters()]
            
    return Layer, MLP, Neuron


@app.cell
def __(mo):
    lr = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="learning rate")
    epochs = mo.ui.slider(10, 500, value=100, step=10, label="epochs")
    mo.hstack([lr, epochs])
    return epochs, lr


@app.cell
def __(MLP, epochs, lr, mo):
    import matplotlib.pyplot as plt
    
    # Обучение XOR
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])

    losses = []
    for k in range(epochs.value):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
        
        # update
        for p in n.parameters():
            p.data += -lr.value * p.grad
            
        losses.append(loss.data)
        
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_title("XOR Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    
    mo.vstack([
        mo.md(f"Финальный loss: **{losses[-1]:.4f}**"),
        mo.as_html(fig)
    ])
    return ax, fig, k, loss, losses, n, p, plt, xs, ypred, ys


if __name__ == "__main__":
    app.run()
