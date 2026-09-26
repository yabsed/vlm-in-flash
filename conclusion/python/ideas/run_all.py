"""Execute every notebook from a fresh kernel; never resume from saved results."""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", nargs="*", help="Optional experiment numbers, e.g. 01 02")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per cell")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-conclusion-matplotlib")
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/vlm-conclusion-jupyter")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    paths = [p for p in sorted(root.glob("[0-9][0-9]_*.ipynb"))
             if not args.numbers or p.name[:2] in args.numbers]
    if not paths:
        parser.error("No matching notebooks")
    for path in paths:
        print(f"Executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        try:
            NotebookClient(notebook, timeout=args.timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(root)}}).execute()
        finally:
            nbformat.write(notebook, path)
        print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
