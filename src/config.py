"""Config loading. All paths in the YAML are relative to the repository root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def resolve(cfg: dict, relative: str) -> Path:
    """Resolve a repo-relative path from the config and create parent dirs."""
    p = Path(cfg["_root"]) / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
