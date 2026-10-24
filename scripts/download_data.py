"""Download the raw public datasets used by build_corpus.py.

  data/raw/hotpotqa/distractor_validation.parquet  <- huggingface.co/datasets/hotpotqa/hotpot_qa (distractor/validation)
  data/raw/postmortems/README.md                   <- github.com/danluu/post-mortems

Usage: uv run python scripts/download_data.py
"""

from __future__ import annotations

from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import httpx

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    RAW / "hotpotqa" / "distractor_validation.parquet":
        "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet",
    RAW / "postmortems" / "README.md": "https://raw.githubusercontent.com/danluu/post-mortems/master/README.md",
}


def main() -> None:
    for path, url in FILES.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(url, follow_redirects=True, timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"{path.relative_to(RAW.parent.parent)}: {len(r.content):,} bytes")


if __name__ == "__main__":
    main()
