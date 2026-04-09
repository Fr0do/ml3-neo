"""Seminar Efficient — Efficient ML (advanced level)."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torch.quantization import quantize_dynamic
    import time
    return mo, torch, nn, models, quantize_dynamic, time


@app.cell
def __(mo):
    mo.md(
        r"""
        # Семинар Efficient — Efficient ML (advanced)

        ## Задача

        1. Квантизовать ResNet-18 в INT8 (torch.quantization.quantize_dynamic).
        2. Измерить размер модели до и после.
        3. Измерить throughput на CPU и accuracy degradation.
        4. Сравнить с knowledge distillation (ResNet-18 → MobileNetV2).

        ## Чек-лист реализации

        - [ ] Загрузить ResNet-18
        - [ ] Применить quantize_dynamic
        - [ ] Сравнить размеры и throughput
        - [ ] Написать цикл KD
        """
    )
    return


@app.cell
def __(torch, models, quantize_dynamic, time):
    # === SUBMISSION (start) ===
    def measure_model_size(model):
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024

    def test_throughput(model, device="cpu", batches=50):
        dummy_input = torch.randn(16, 3, 224, 224).to(device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(10): # warmup
                _ = model(dummy_input)
            start = time.time()
            for _ in range(batches):
                _ = model(dummy_input)
            end = time.time()
        return (16 * batches) / (end - start)

    # model = models.resnet18(pretrained=True)
    # quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # === SUBMISSION (end) ===
    return measure_model_size, test_throughput


if __name__ == "__main__":
    app.run()
