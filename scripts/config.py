"""Load and validate benchmark configuration from YAML."""
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load config.yaml and return the parsed dict."""
    p = path or CONFIG_PATH
    with open(p) as f:
        cfg = yaml.safe_load(f)

    # Merge pipelines into models with type=pipeline
    for name, pipe_cfg in cfg.get("pipelines", {}).items():
        cfg["models"][name] = {
            "type": "pipeline",
            "stage1": pipe_cfg["stage1"],
            "stage2": pipe_cfg["stage2"],
            "params": pipe_cfg.get("params", "?"),
        }

    return cfg
