"""Render TikZ standalone files to SVG (and PDF) via Tectonic + dvisvgm.

Usage:
    python figures/render.py figures/tikz/normalization.tex
    python figures/render.py --all          # rebuild every .tex in figures/tikz/

Output goes to figures/tikz/_build/<name>.svg and .pdf. The build dir is
gitignored — TikZ source is the canonical artifact.

Why Tectonic: zero-conf LaTeX, fetches packages on demand, deterministic
across linux/macos/win. Why dvisvgm: produces clean vector SVG with embedded
fonts that render the same in browser and PDF.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIKZ_DIR = ROOT / "tikz"
BUILD_DIR = TIKZ_DIR / "_build"


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def render_one(tex_path: Path) -> Path:
    """Compile one standalone .tex into _build/<name>.svg + .pdf."""
    if not tex_path.exists():
        raise FileNotFoundError(tex_path)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    name = tex_path.stem
    work = BUILD_DIR / f"_work_{name}"
    work.mkdir(exist_ok=True)

    # Tectonic resolves \input relative to the source file's directory.
    # We pass the source via stdin? No — easier: copy the file and the
    # _preamble next to it inside the work dir so \input{_preamble.tex} works.
    shutil.copy(tex_path, work / tex_path.name)
    preamble = TIKZ_DIR / "_preamble.tex"
    if preamble.exists():
        shutil.copy(preamble, work / "_preamble.tex")

    if not have("tectonic"):
        sys.exit("error: tectonic not found in PATH (pixi install)")

    cmd = [
        "tectonic",
        "--keep-intermediates",
        "--outdir", str(work),
        str(work / tex_path.name),
    ]
    subprocess.run(cmd, check=True)

    pdf = work / f"{name}.pdf"
    if not pdf.exists():
        sys.exit(f"error: tectonic produced no PDF for {tex_path}")

    out_pdf = BUILD_DIR / f"{name}.pdf"
    out_svg = BUILD_DIR / f"{name}.svg"
    shutil.copy(pdf, out_pdf)

    # Предпочитаем pdftocairo из poppler: надёжно читает PDF от xdvipdfmx
    # и есть на всех платформах через conda-forge. Fallback-ы — pdf2svg,
    # затем dvisvgm.
    if have("pdftocairo"):
        # pdftocairo требует путь без .svg (он сам добавит)
        out_base = out_svg.with_suffix("")
        subprocess.run(
            ["pdftocairo", "-svg", str(out_pdf), str(out_svg)],
            check=True,
        )
        del out_base  # не используется, но сохраняем намерение
    elif have("pdf2svg"):
        subprocess.run(["pdf2svg", str(out_pdf), str(out_svg)], check=True)
    elif have("dvisvgm"):
        subprocess.run(
            ["dvisvgm", "--pdf", "--no-fonts=0", "--exact-bbox",
             f"--output={out_svg}", str(out_pdf)],
            check=True,
        )
    else:
        print("warning: no PDF→SVG converter installed — SVG skipped")

    shutil.rmtree(work, ignore_errors=True)
    print(f"  ✓ {name}: {out_pdf.name}, {out_svg.name}")
    return out_svg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", help="path to .tex file")
    p.add_argument("--all", action="store_true", help="render everything in figures/tikz/")
    args = p.parse_args()

    if args.all:
        sources = sorted(
            f for f in TIKZ_DIR.glob("*.tex") if not f.name.startswith("_")
        )
        if not sources:
            sys.exit("no .tex files found")
        print(f"rendering {len(sources)} TikZ file(s)…")
        for s in sources:
            render_one(s)
        return

    if not args.source:
        p.error("provide a .tex path or use --all")
    render_one(Path(args.source))


if __name__ == "__main__":
    main()
