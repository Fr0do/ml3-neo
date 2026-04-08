"""Execution verifier: проверяет, что submission/notebook.py реально
производит предъявленный `predictions.npy`.

Зачем: убивает «submit-without-code» и хардкод ответов. Студент не может
просто принести чужой `predictions.npy` — грейдер прогоняет его код на
*его* инстансе и сравнивает.

Как это работает:
1. Копируем submission/notebook.py в tmp-рабочую директорию.
2. Подменяем переменные окружения `ML3_HIDDEN_TEST` → путь к инстансу
   студента и `ML3_OUTPUT_PREDICTIONS` → куда писать свежие predictions.
3. Вызываем `marimo run submission/notebook.py` или `python -m marimo run ...`
   с жёстким timeout и capped memory (если есть `ulimit`/`systemd-run`,
   иначе — плоский timeout).
4. Сравниваем новый `predictions.npy` с тем, что студент положил в сабмит:
   - tolerance 0 для классификации (идеальное совпадение меток),
   - agreement_rate ≥ 0.98 — ok, иначе mismatch.
5. Возвращаем отчёт; судья получит его как дополнительный сигнал.

Песочница: в reference-реализации — только timeout + cwd isolation. Для
production инструктор может обернуть в firejail/gVisor/модульный runner.
Контракт один — возвращаем `ExecutionReport`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class ExecutionReport:
    ok: bool
    reason: str
    agreement_rate: float         # доля совпавших меток с submitted predictions
    submitted_hash: str
    recomputed_hash: str
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict:
        return asdict(self)


# Лимиты по умолчанию. Инструктор может переопределить через meta.yml
# (см. use в judge.py, если захотите).
DEFAULT_TIMEOUT_SEC = 600
AGREEMENT_THRESHOLD = 0.98


def verify_submission(
    root: Path,
    hw_id: str,
    submission: Path,
    hidden_test: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> ExecutionReport:
    nb = submission / "submission" / "notebook.py"
    submitted_preds = submission / "submission" / "predictions.npy"

    if not nb.exists():
        return _fail("missing submission/notebook.py")
    if not submitted_preds.exists():
        return _fail("missing submission/predictions.npy")
    if not hidden_test.exists():
        return _fail(f"hidden test not available: {hidden_test}")

    # Готовим рабочий каталог. Копируем ноутбук сюда; код не должен
    # трогать ничего за пределами workdir.
    workdir = root / "grader" / ".cache" / "exec" / hw_id / submission.name
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(nb, workdir / "notebook.py")

    out_preds = workdir / "predictions.npy"
    env = os.environ.copy()
    env["ML3_HIDDEN_TEST"] = str(hidden_test.resolve())
    env["ML3_OUTPUT_PREDICTIONS"] = str(out_preds.resolve())
    env["PYTHONHASHSEED"] = "0"

    # Как запускать ноутбук:
    # - если это marimo-файл (имеет `marimo.App(`) — предпочитаем marimo run,
    #   headless; это не поднимет GUI и выполнит все ячейки по графу.
    # - иначе (plain python для юнит-тестов execute.py) — просто python.
    body = (workdir / "notebook.py").read_text(errors="replace")
    is_marimo_notebook = "marimo.App(" in body
    cmd: list[str]
    if is_marimo_notebook and shutil.which("marimo"):
        cmd = ["marimo", "run", "notebook.py",
               "--headless", "--no-token"]
    else:
        cmd = [sys.executable, "notebook.py"]

    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=env,
            capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as e:
        return _fail(
            f"execution timed out after {timeout}s",
            stdout=e.stdout or "", stderr=e.stderr or "",
        )

    stdout_tail = (proc.stdout or "")[-2000:]
    stderr_tail = (proc.stderr or "")[-2000:]

    if proc.returncode != 0:
        return _fail(
            f"notebook.py exited with code {proc.returncode}",
            stdout=stdout_tail, stderr=stderr_tail,
        )

    if not out_preds.exists():
        return _fail(
            "notebook.py did not write ML3_OUTPUT_PREDICTIONS",
            stdout=stdout_tail, stderr=stderr_tail,
        )

    submitted = np.load(submitted_preds)
    recomputed = np.load(out_preds)

    if submitted.shape != recomputed.shape:
        return ExecutionReport(
            ok=False,
            reason=f"shape mismatch: submitted {submitted.shape} vs recomputed {recomputed.shape}",
            agreement_rate=0.0,
            submitted_hash=_hash_array(submitted),
            recomputed_hash=_hash_array(recomputed),
            stdout_tail=stdout_tail, stderr_tail=stderr_tail,
        )

    agreement = float((submitted == recomputed).mean())
    ok = agreement >= AGREEMENT_THRESHOLD
    return ExecutionReport(
        ok=ok,
        reason=(
            "predictions reproducible"
            if ok
            else f"agreement {agreement:.3f} < {AGREEMENT_THRESHOLD}"
        ),
        agreement_rate=agreement,
        submitted_hash=_hash_array(submitted),
        recomputed_hash=_hash_array(recomputed),
        stdout_tail=stdout_tail, stderr_tail=stderr_tail,
    )


def _fail(reason: str, stdout: str = "", stderr: str = "") -> ExecutionReport:
    return ExecutionReport(
        ok=False,
        reason=reason,
        agreement_rate=0.0,
        submitted_hash="",
        recomputed_hash="",
        stdout_tail=stdout[-2000:] if stdout else "",
        stderr_tail=stderr[-2000:] if stderr else "",
    )


def _hash_array(arr: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
