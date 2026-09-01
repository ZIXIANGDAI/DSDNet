from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section, key in (("data", "root"), ("training", "stage1_init"), ("training", "save_dir")):
        value = config.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            config[section][key] = str((path.parent / value).resolve())
    return config
