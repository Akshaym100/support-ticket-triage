"""Convert a `# %%`-style Python script into a Jupyter notebook.

Run:  uv run python scripts/py_to_ipynb.py notebooks/finetune_distilbert_lora.py
"""

import json
import sys
from pathlib import Path


def convert(src: Path) -> Path:
    cells, kind, lines = [], None, []

    def flush():
        body = "\n".join(lines).strip("\n")
        if kind and body:
            if kind == "markdown":
                body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            source = [l + "\n" for l in body.splitlines()]
            source[-1] = source[-1].rstrip("\n")
            cell = {"cell_type": kind, "metadata": {}, "source": source}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()

    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    print("wrote", convert(Path(sys.argv[1])))
