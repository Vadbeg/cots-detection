"""Module with utils for whole project"""

from pathlib import Path
from typing import Dict, Union

import yaml
from yaml import Loader


def load_config(config_path: Union[Path, str]) -> Dict:
    config_path = Path(config_path)

    with config_path.open(mode='r') as file:
        config = yaml.load(file.read(), Loader=Loader)

    return config
