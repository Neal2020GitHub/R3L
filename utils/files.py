import os


def expand_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def ensure_dir(dir_path: str) -> str:
    resolved = expand_path(dir_path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def ensure_parent_dir(path: str) -> str:
    resolved = expand_path(path)
    parent = os.path.dirname(resolved) or "."
    os.makedirs(parent, exist_ok=True)
    return resolved


