"""Dump interesting cells of the V17 kernel notebook."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"benchmark\v17_kernel\enveda-casmi-2026-fast-spectral-cosine-baseline.ipynb",
                    encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if any(w in src.lower() for w in ("coconut", "input", "dataset", "csv", "parquet",
                                      "pool", "candidate")):
        print(f"--- cell {i} ({c['cell_type']}) ---")
        print(src[:2500])
        print()
