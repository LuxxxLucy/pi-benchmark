"""Pre-fetch all candidate weights + tokenizers into the HF cache.

Idempotent: already-cached models are a no-op. Separates download time
from measurement time.

Uses a tight `allow_patterns` filter so we only pull the tokenizer +
one weight format (safetensors preferred, pytorch_model.bin fallback)
— not the full repo snapshot. Bare `snapshot_download` would double-pull
bin+safetensors on repos that ship both and also grab READMEs / images /
tf_model.h5 / flax_model.msgpack that the bench never touches.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from models import CANDIDATES  # noqa: E402


_TOKENIZER_PATTERNS = [
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
    "spm.model",
    "sentencepiece.bpe.model",
]


def _weight_pattern(files: list[str]) -> str:
    """Pick one weight pattern per repo. Prefer safetensors."""
    if any(f.endswith(".safetensors") for f in files):
        return "*.safetensors"
    return "pytorch_model*.bin"  # sharded or single


def main() -> int:
    api = HfApi()
    for spec in CANDIDATES:
        print(f"[download] {spec.hf_id}  ({spec.name})")
        try:
            files = api.list_repo_files(spec.hf_id)
            allow = _TOKENIZER_PATTERNS + [_weight_pattern(files)]
            snapshot_download(repo_id=spec.hf_id, allow_patterns=allow)
        except Exception as e:
            print(f"  FAIL: {e!r}")
    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
