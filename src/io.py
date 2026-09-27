import hashlib
import json
import os
from pathlib import Path
import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_npz(path, **arrays):
    path = Path(path)
    with path.with_name(path.name + ".part").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(path.with_name(path.name + ".part"), path)
