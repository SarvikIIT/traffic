import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

class ConfigManager:

    def __init__(self, config_path: str = "config/config.yaml"):
        self._path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with open(self._path, "r") as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._config)

    def section(self, name: str) -> Dict[str, Any]:
        return self._config.get(name, {})

def load_config(path: Optional[str] = None) -> ConfigManager:
    if path is None:
        path = os.environ.get("TRAFFIC_CONFIG", "config/config.yaml")
    return ConfigManager(path)
