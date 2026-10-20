"""Dump the de novo tutorial notebook cells (code + markdown headers)."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"benchmark\tutorial_kernel\casmi-denovo-tutorial-notebook.ipynb",
                    encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    print(f"--- cell {i} ({c['cell_type']}, {len(src)} chars) ---")
    print(src[:3000])
    print()
