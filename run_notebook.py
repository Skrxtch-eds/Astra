"""Execute the notebook with this Python environment; no global kernel setup."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main() -> None:
    root = Path(__file__).resolve().parent
    path = root / "hockey_analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Explicit temporary specification avoids accidentally using a different Python.
    with tempfile.TemporaryDirectory(prefix="hockey-kernel-") as temporary:
        kernel = Path(temporary) / "hockey"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (hockey)", "language": "python",
        }), encoding="utf-8")
        specs = KernelSpecManager(kernel_dirs=[temporary], ensure_native_kernel=False)
        manager = KernelManager(kernel_name="hockey", kernel_spec_manager=specs)
        client = NotebookClient(notebook, km=manager, timeout=180, startup_timeout=60,
                                allow_errors=False, resources={"metadata": {"path": str(root)}})
        print("Starting notebook execution...", flush=True)
        client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Notebook executed successfully.", flush=True)


if __name__ == "__main__":
    main()
