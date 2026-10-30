"""Configuration: YAML files with an optional `base` file, merged recursively, plus command-line overrides."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _set(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=None) -> dict:
    """Read a YAML configuration. `base: other.yaml` (relative to the file) is read first and updated with this file.
    `overrides`: list of "key.sub=value" strings; values are parsed as YAML (numbers, lists, true/false, null)."""
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = REPO_ROOT / path
    cfg = yaml.safe_load(path.read_text()) or {}
    if cfg.get("base"):
        cfg = _merge(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides or []:
        key, _, val = item.partition("=")
        if not _:
            raise ValueError(f"override '{item}' must be key=value")
        _set(cfg, key.strip(), yaml.safe_load(val))
    cfg["_config_file"] = str(path)
    return cfg


def resolve(path) -> Path | None:
    """Absolute path; relative paths are taken from the repository root."""
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]
