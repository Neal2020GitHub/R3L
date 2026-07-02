"""
Leveled status / debug lines for library internals.

Terse, bracket-prefixed one-liners ([INFO]/[OK]/[WARN]/[ERR]) plus a colored
nested-structure dumper, for code deep in the pipeline (solver, asset, renderer)
that wants to report status without owning the curated entry-point experience —
that polished chrome lives in `utils.console`. This module is colorama-only.
"""

from typing import Any
from colorama import Fore


def print_dict(d: dict, indent: int = 4):
    def is_branch(x):
        return isinstance(x, (dict, list, tuple))

    def is_empty(x) -> bool:
        return (isinstance(x, dict) and not x) or (isinstance(x, (list, tuple)) and len(x) == 0)

    def brackets_for(x):
        if isinstance(x, dict):
            return "{", "}"
        return ("[", "]") if isinstance(x, list) else ("(", ")")

    def scalar_str(x: Any) -> str:
        try:
            if x.__class__.__module__.startswith("numpy") and hasattr(x, "item"):
                x = x.item()
        except Exception:
            pass
        if isinstance(x, str):
            return f'"{x}"'
        return str(x)

    def walk(obj: Any, level: int) -> None:
        pad = " " * (level * indent)
        if isinstance(obj, dict):
            if not obj:
                print(pad + f"{Fore.MAGENTA}{{}}{Fore.RESET}")
                return
            print(pad + f"{Fore.MAGENTA}{{{Fore.RESET}")
            for k, v in obj.items():
                line_pad = " " * ((level + 1) * indent)
                if is_branch(v):
                    if is_empty(v):
                        ob, cb = brackets_for(v)
                        print(f"{line_pad}{Fore.BLUE}{k}{Fore.RESET}: {Fore.MAGENTA}{ob}{cb}{Fore.RESET}")
                    else:
                        print(f"{line_pad}{Fore.BLUE}{k}{Fore.RESET}: {Fore.MAGENTA}→{Fore.RESET}")
                        walk(v, level + 1)
                else:
                    print(f"{line_pad}{Fore.BLUE}{k}{Fore.RESET}: {Fore.CYAN}{scalar_str(v)}{Fore.RESET}")
            print(pad + f"{Fore.MAGENTA}}}{Fore.RESET}")
        elif isinstance(obj, (list, tuple)):
            open_b, close_b = ("[", "]") if isinstance(obj, list) else ("(", ")")
            if len(obj) == 0:
                print(pad + f"{Fore.MAGENTA}{open_b}{close_b}{Fore.RESET}")
                return
            print(pad + f"{Fore.MAGENTA}{open_b}{Fore.RESET}")
            for i, v in enumerate(obj):
                line_pad = " " * ((level + 1) * indent)
                tag = f"{Fore.MAGENTA}[{i}]{Fore.RESET}"
                if is_branch(v):
                    if is_empty(v):
                        ob, cb = brackets_for(v)
                        print(f"{line_pad}{tag}: {Fore.MAGENTA}{ob}{cb}{Fore.RESET}")
                    else:
                        print(f"{line_pad}{tag}: {Fore.MAGENTA}→{Fore.RESET}")
                        walk(v, level + 1)
                else:
                    print(f"{line_pad}{tag}: {Fore.CYAN}{scalar_str(v)}{Fore.RESET}")
            print(pad + f"{Fore.MAGENTA}{close_b}{Fore.RESET}")
        else:
            print(pad + f"{Fore.CYAN}{scalar_str(obj)}{Fore.RESET}")

    walk(d, 0)


def print_warn(msg: str, header: str = "WARN", end: str = "\n"):
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.YELLOW}{prefix}{msg}{Fore.RESET}", end=end)


def print_info(msg: str, header: str = "INFO", end: str = "\n"):
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.BLUE}{prefix}{msg}{Fore.RESET}", end=end)


def print_good(msg: str, header: str = "OK", end: str = "\n"):
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.GREEN}{prefix}{msg}{Fore.RESET}", end=end)


def print_error(msg: str, header: str = "ERR", end: str = "\n"):
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.RED}{prefix}{msg}{Fore.RESET}", end=end)
