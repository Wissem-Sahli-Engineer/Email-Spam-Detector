"""
Robust cell-by-cell notebook executor with real-time logging and auto-save.
"""

import time
import nbformat
from nbclient import NotebookClient

print("--- Step-by-Step Notebook Execution ---")
nb = nbformat.read("diabetes_risk_ml_pipeline.ipynb", as_version=4)

client = NotebookClient(nb, timeout=300, kernel_name="python3")

with client.setup_kernel():
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            t0 = time.time()
            preview = cell.source.strip().split("\n")[0][:40]
            print(f"[{idx+1}/{len(nb.cells)}] Running Code Cell: {preview} ...", end="", flush=True)
            client.execute_cell(cell, idx)
            print(f" done in {time.time()-t0:.2f}s")
        else:
            print(f"[{idx+1}/{len(nb.cells)}] Markdown Cell")

        # Save progress
        with open("diabetes_risk_ml_pipeline.ipynb", "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

print("\n✓ SUCCESS: All notebook cells executed and saved with outputs!")
