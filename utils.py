from datetime import datetime
from os import PathLike as os_PathLike
from pathlib import Path
from typing import Union

PathLike = Union[str, os_PathLike]


def create_run_log_dir(name: str) -> Path:
    date_string = datetime.now().strftime("%Y-%m-%dT%H:%M")
    p = Path(".") / "experiments" / f"run - {date_string} - {name}/"
    p.mkdir(parents=False, exist_ok=False)
    return p
