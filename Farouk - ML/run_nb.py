"""
Execute notebook cell-by-cell and save outputs cleanly in-place.
"""

import nbformat
from nbclient import NotebookClient

print("Reading notebook...")
nb = nbformat.read("diabetes_risk_ml_pipeline.ipynb", as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")

print("Executing notebook cells...")
client.execute()

print("Saving executed notebook...")
with open("diabetes_risk_ml_pipeline.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("✓ All notebook cells executed and saved successfully!")
