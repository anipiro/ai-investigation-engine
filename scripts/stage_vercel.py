"""Stage a minimal Vercel deployment in build/vercel/ (code only — no data, tests or local caches).

  uv run python scripts/stage_vercel.py && cd build/vercel && vercel deploy --prod

Vercel's Python runtime builds the FastAPI app in main.py into one Fluid Compute function.
The ONNX models are downloaded to /tmp on a cold start (MODEL_CACHE_DIR=/tmp/models).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "vercel"


def main() -> None:
    vercel_dir = OUT / ".vercel"
    keep = None
    if vercel_dir.exists():
        keep = OUT.parent / ".vercel-keep"
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(vercel_dir, keep)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "engine", OUT / "engine", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "main.py", OUT / "main.py")
    req = subprocess.run(["uv", "export", "--no-dev", "--no-hashes", "--format", "requirements-txt"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    (OUT / "requirements.txt").write_text(req, encoding="utf-8")
    (OUT / ".python-version").write_text("3.12\n", encoding="utf-8")
    (OUT / "vercel.json").write_text(json.dumps({
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "regions": ["iad1"],
        "functions": {"main.py": {"maxDuration": 300}},
    }, indent=2), encoding="utf-8")
    if keep is not None:
        shutil.copytree(keep, vercel_dir)
        shutil.rmtree(keep, ignore_errors=True)
    elif (ROOT / ".vercel").exists():
        shutil.copytree(ROOT / ".vercel", vercel_dir)
    print(f"staged {OUT}")


if __name__ == "__main__":
    main()
