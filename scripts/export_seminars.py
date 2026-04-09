"""Export all Marimo seminar notebooks to WASM HTML.

Usage:
    pixi run export-seminars          # all seminars
    pixi run export-seminars 04-cnn   # one module

Output: seminars/<module>/{basic,advanced}.html
        seminars/tracks/<track>/advanced.html

These HTML files are embedded in lecture pages via .level-toggle buttons.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def export_notebook(py_path: Path) -> bool:
    out_html = py_path.with_suffix(".html")
    print(f"  exporting {py_path.relative_to(ROOT)} ...", end=" ", flush=True)
    result = subprocess.run(
        ["marimo", "export", "html-wasm", str(py_path), "-o", str(out_html), "--mode", "run"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("ok")
        return True
    else:
        print(f"FAILED\n{result.stderr[:300]}")
        return False


def main():
    filter_module = sys.argv[1] if len(sys.argv) > 1 else None

    notebooks = sorted(ROOT.glob("seminars/**/*.py"))
    if filter_module:
        notebooks = [p for p in notebooks if filter_module in str(p)]

    if not notebooks:
        print(f"No notebooks found (filter={filter_module!r})")
        sys.exit(1)

    print(f"Exporting {len(notebooks)} notebooks to WASM HTML...")
    failed = 0
    for nb in notebooks:
        if not export_notebook(nb):
            failed += 1

    if failed:
        print(f"\n{failed}/{len(notebooks)} failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(notebooks)} exported successfully.")
        print("Run 'pixi run render' to rebuild the site with embedded seminars.")


if __name__ == "__main__":
    main()
