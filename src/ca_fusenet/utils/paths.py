'''
Docstring for ca_fusenet.utils.paths

Is used to ensure that the folder where the model will save the results (weights, logs, graphs) 
actually exists before attempting to write to it, and if not creates it and all necessary parent directories.
'''
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
