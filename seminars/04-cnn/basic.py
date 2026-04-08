"""Seminar 04 — CNN (basic level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/04-cnn/basic.py

Цель семинара (basic):
    1. Собрать ResNet-18 из BasicBlock'ов руками.
    2. Обучить на CIFAR-10 и понять, что меняет каждый кусок.
    3. Посмотреть на receptive field руками через фильтр Дирака.

Скаффолдинг здесь высокий: большая часть кода написана, студент заполняет
TODO-ячейки и эксперименты.
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
        # Семинар 04 — CNN (basic)

        Перед запуском убедитесь, что прочитали [лекцию](../../lectures/04-cnn/lecture.qmd).

        Ниже мы:

        1. Загрузим CIFAR-10 через `torchvision`.
        2. Соберём `BasicBlock` и из него — мини-ResNet.
        3. Обучим на 5 эпох и посмотрим на learning curves.
        4. Визуализируем receptive field через single-impulse трюк.

        Все ячейки реактивны: меняешь параметр — пересчитываются зависимости.
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return F, nn, torch


@app.cell
def __(mo):
    batch_size = mo.ui.slider(32, 512, value=128, step=32, label="batch size")
    lr = mo.ui.slider(1e-4, 3e-2, value=1e-3, step=1e-4, label="learning rate")
    epochs = mo.ui.slider(1, 10, value=3, step=1, label="epochs")
    mo.hstack([batch_size, lr, epochs])
    return batch_size, epochs, lr


@app.cell
def __(F, nn):
    class BasicBlock(nn.Module):
        """ResNet BasicBlock: Conv-BN-ReLU-Conv-BN + skip + ReLU."""

        expansion = 1

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=1, bias=False,
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            return F.relu(out)
    return (BasicBlock,)


@app.cell
def __(BasicBlock, F, nn):
    class MiniResNet(nn.Module):
        """4-stage ResNet с настраиваемой шириной."""

        def __init__(self, num_classes: int = 10, width: int = 32):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
            self.stage1 = self._make_stage(width, width, n=2, stride=1)
            self.stage2 = self._make_stage(width, width * 2, n=2, stride=2)
            self.stage3 = self._make_stage(width * 2, width * 4, n=2, stride=2)
            self.stage4 = self._make_stage(width * 4, width * 8, n=2, stride=2)
            self.fc = nn.Linear(width * 8, num_classes)

        @staticmethod
        def _make_stage(in_c, out_c, n, stride):
            blocks = [BasicBlock(in_c, out_c, stride=stride)]
            for _ in range(n - 1):
                blocks.append(BasicBlock(out_c, out_c, stride=1))
            return nn.Sequential(*blocks)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)
    return (MiniResNet,)


@app.cell
def __(MiniResNet, mo, torch):
    model = MiniResNet(num_classes=10, width=32)
    n_params = sum(p.numel() for p in model.parameters())
    mo.md(f"**MiniResNet**: {n_params/1e6:.2f}M параметров.")

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    mo.md(f"Sanity check: вход `{tuple(x.shape)}` → выход `{tuple(y.shape)}`")
    return model, n_params, x, y


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Receptive field руками

        Трюк: подаём на вход тензор с одной единичкой в центре (impulse) и
        смотрим, какие пиксели на промежуточных feature maps стали ненулевыми.
        Это и есть фактическое RF данного выходного нейрона.

        ⚠️ Важно: импульс должен пройти **до** BN/ReLU, иначе ReLU «съест»
        половину сигнала. Поэтому ниже мы переключаем модель в `eval()` и
        используем линеаризованную версию через `torch.no_grad`.
        """
    )
    return


@app.cell
def __(model, torch):
    @torch.no_grad()
    def receptive_field(model, layer_name: str, input_size: int = 65) -> torch.Tensor:
        """Вернёт map ненулевых пикселей входа, влияющих на центр layer_name."""
        impulse = torch.zeros(1, 3, input_size, input_size)
        impulse[0, :, input_size // 2, input_size // 2] = 1.0

        feats = {}

        def hook(name):
            def _h(module, inp, out):
                feats[name] = out.detach()
            return _h

        handle = dict(model.named_modules())[layer_name].register_forward_hook(
            hook(layer_name)
        )
        model.eval()
        _ = model(impulse)
        handle.remove()
        return (feats[layer_name].abs().sum(dim=1).squeeze() > 0).float()


    rf = receptive_field(model, "stage2")
    return receptive_field, rf


@app.cell
def __(mo, rf):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rf.numpy(), cmap="viridis")
    ax.set_title(f"RF центра stage2 (сторона ~{int(rf.sum().sqrt().item())})")
    ax.axis("off")
    mo.center(fig)
    return ax, fig, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        ## TODO для студента

        1. **Receptive field растёт.** Замени `"stage2"` на `"stage3"` и
           `"stage4"`. Запиши размер RF для каждой стадии. Совпадает ли с
           формулой $1 + L(k-1)$ с учётом stride?
        2. **Заменить BN на GN.** Перепиши `BasicBlock`, заменив `nn.BatchNorm2d`
           на `nn.GroupNorm(num_groups=8, num_channels=...)`. Как меняется
           sanity-check на батче размера 2?
        3. **Аугментации.** Подключи `albumentations` (см. лекцию) и обучи на
           CIFAR-10 одну модель с аугментациями и одну без. Сравни test
           accuracy после 5 эпох.

        В **advanced** версии этого же семинара (`advanced.py`) скаффолдинга
        нет — там нужно собрать всё с нуля и сравнить с torchvision-эталоном.
        """
    )
    return


if __name__ == "__main__":
    app.run()
