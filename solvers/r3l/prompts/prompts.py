"""
prompts.py - Prompt store with Jinja2 template rendering.

Classes:
- JinjaRenderer: renders .j2/.jinja2 templates to YAML
- PromptStore: manages cache with stamp-based invalidation
"""

import os
import tempfile
import yaml
from string import Template
from typing import Dict, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class TemplateNotFound(FileNotFoundError):
    pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _template_category(fname: str) -> Optional[str]:
    """Return category if fname ends with .j2 or .jinja2, else None."""
    for ext in ('.jinja2', '.j2'):
        if fname.endswith(ext):
            return fname[:-len(ext)]
    return None


def _validate_prompt_dict(doc, source: str) -> None:
    """Raise ValueError if doc is not dict[str, str]."""
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: expected dict")
    for k, v in doc.items():
        if not isinstance(v, str):
            raise ValueError(f"{source}[{k}]: expected string")


def _atomic_write(path: str, content: str) -> None:
    """Atomically write content to path via tempfile + rename."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# JinjaRenderer
# ---------------------------------------------------------------------------

class JinjaRenderer:
    """Render Jinja2 templates to validated YAML dicts."""

    def __init__(self, templates_dir: str, include_paths: list):
        self.dir = templates_dir
        self.env = Environment(
            loader=FileSystemLoader(include_paths),
            undefined=StrictUndefined,
        )

    def render(self, ctx: dict) -> Dict[str, Tuple[str, dict]]:
        """
        Render all templates in directory.
        Returns {category: (raw_text, parsed_dict)}.
        """
        if not os.path.isdir(self.dir):
            return {}

        out = {}
        for fname in os.listdir(self.dir):
            cat = _template_category(fname)
            if cat is None or fname.startswith('.'):
                continue

            raw = self.env.get_template(fname).render(ctx)
            doc = yaml.safe_load(raw)
            _validate_prompt_dict(doc, fname)
            out[cat] = (raw, doc)

        return out


# ---------------------------------------------------------------------------
# PromptStore
# ---------------------------------------------------------------------------

class PromptStore:
    """
    Prompt cache with Jinja2 rendering and stamp-based invalidation.

    Access via attribute (prompts.constraints_spatial). Cache rebuilds
    automatically when the module/prompt config affecting template rendering
    changes.
    """

    def __init__(self):
        self._root = os.path.dirname(__file__)
        tpl_dir = os.path.join(self._root, 'templates')
        self._renderer = JinjaRenderer(tpl_dir, [tpl_dir, self._root])
        self._cache: Dict[str, Dict[str, Template]] = {}
        self._stamp: Optional[tuple] = None
        self._cfg = None

    @property
    def cfg(self):
        """Lazy cfg import to avoid circular dependency."""
        if self._cfg is None:
            from solvers.r3l.config import cfg
            self._cfg = cfg
        return self._cfg

    def _stamp_now(self) -> tuple:
        return (
            self.cfg.modules.decomposition,
            self.cfg.modules.imagination_pose,
            self.cfg.prompt.hv_absolute,
            self.cfg.modules.imagination_footprint,
        )

    def _ensure_fresh(self) -> None:
        stamp = self._stamp_now()
        if stamp != self._stamp:
            self._rebuild()
            self._stamp = stamp

    def _rebuild(self) -> None:
        ctx = {
            'clustered': self.cfg.modules.decomposition != 'none',
            'cognitive_map': self.cfg.modules.imagination_pose,
            'volumetric_grounding': self.cfg.modules.imagination_footprint,
            'hv_absolute': self.cfg.prompt.hv_absolute,
        }
        new = {}

        # Rendered templates -> cache + disk
        for cat, (raw, doc) in self._renderer.render(ctx).items():
            new[cat] = {k: Template(v) for k, v in doc.items()}
            _atomic_write(os.path.join(self._root, f'{cat}.yaml'), raw)

        self._cache = new

    def __getattr__(self, name: str) -> Dict[str, Template]:
        if name.startswith('_'): raise AttributeError(name)
        self._ensure_fresh()
        try:
            return self._cache[name]
        except KeyError:
            raise TemplateNotFound(f"Template file {name} not found")


prompts = PromptStore()

# Example:
#   prompts.constraints['key'].substitute(...)
