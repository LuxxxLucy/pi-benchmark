"""Pre-fetch all candidate weights + tokenizers into the HF cache.

Idempotent: already-cached models are a no-op. Separates download time
from measurement time.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from huggingface_hub import snapshot_download  # noqa: E402
from models import CANDIDATES  # noqa: E402


def main() -> int:
    for spec in CANDIDATES:
        print(f"[download] {spec.hf_id}  ({spec.name})")
        try:
            snapshot_download(repo_id=spec.hf_id)
        except Exception as e:
            print(f"  FAIL: {e!r}")
    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
