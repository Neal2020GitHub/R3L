"""
Lists the CLIP and SBERT model files the retriever needs, and checks that they
are present in the local cache.

To load a model, the program needs several files: the model weights, plus the
config and tokenizer files that control how those weights are used. The README
is not needed, because it does not change the result. A model is "present" when
all of these files are in the local cache, and each one is non-empty and
readable.

The program checks for presence because a model can be broken in two ways that
are easy to miss:
- A download was interrupted. Each file is written all-or-nothing, so an
  interrupted download leaves some files completely missing rather than
  half-written. When the weights file is missing, the loader tries to fetch it
  over the network, and can hang there if the connection is unreliable.
- A config file is missing. sentence-transformers does not raise an error in
  this case. It loads the weights into a default model structure instead of the
  correct one, and then returns wrong vectors without any warning.

One function, resolve(), performs this check. Three parts of the program call
it, so they always agree on whether a model is present:
- The preflight step runs before the pipeline starts. It calls resolve()
  (through missing()) to warn the user about missing models before any slow
  work begins.
- The model loaders call resolve() (and clip_weights()) to get the local path
  of a model before loading it.
- The download script calls resolve() (through present()) to decide which
  models still need to be downloaded.

resolve() checks only that the files exist, are non-empty, and are readable, for
one fixed version of each model. It does not check whether the contents of the
files are correct. Hugging Face verifies each file as it downloads, so a file
downloaded by this program cannot be corrupt. A file damaged later by something
outside the program, such as a manual edit or a disk error, is not detected, by
design.

This module imports no heavy libraries (no torch, open_clip, or
sentence_transformers), so the preflight check stays fast. Only download() uses
the network.
"""

import os
from dataclasses import dataclass

from huggingface_hub import hf_hub_download, try_to_load_from_cache

# The one place the remediation command lives, shared by the Missing message and preflight.
DOWNLOAD_CMD = "uv run python scripts/download_models.py"


@dataclass(frozen=True)
class Model:
    repo: str
    revision: str
    files: tuple[str, ...]

    def __post_init__(self):
        if "/" in self.files[0]:
            raise ValueError(f"{self.repo}: files[0] must be root-level, got {self.files[0]!r}")


@dataclass(frozen=True)
class ClipModel(Model):
    arch: str


CLIP = ClipModel(
    repo="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    revision="1627032197142fbe2a7cfec626f4ced3ae60d07a",
    files=("open_clip_pytorch_model.bin",),
    arch="ViT-L-14",
)
SBERT = Model(
    repo="sentence-transformers/all-mpnet-base-v2",
    revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
    # Everything that defines or weights the model. README.md is omitted: it is model-card
    # text whose content does not shape the loaded model. The rest each do — and omitting a
    # config file makes sentence-transformers silently fall back to a wrong model, so the
    # gate (not the permissive loader) must require them.
    files=(
        "model.safetensors", "modules.json", "config.json",
        "config_sentence_transformers.json", "sentence_bert_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.txt", "1_Pooling/config.json",
    ),
)

MODELS: tuple[Model, ...] = (CLIP, SBERT)


class Missing(RuntimeError):
    def __init__(self, model: Model):
        self.model = model
        super().__init__(f"model not fully cached: {model.repo}. "
                         f"Download it once with: {DOWNLOAD_CMD}")


def _cached_path(repo: str, name: str, revision: str) -> str | None:
    try:
        path = try_to_load_from_cache(repo, name, revision=revision)
        if isinstance(path, str) and os.path.getsize(path) > 0 and os.access(path, os.R_OK):
            return path
    except OSError:
        pass
    return None


def resolve(model: Model) -> str:
    """
    Returns the local directory that holds the model's files.

    Before returning, it confirms that every file the model needs is in the cache
    and readable. If any file is missing, it raises Missing instead, so the caller
    never loads an incomplete model or quietly starts downloading over the network.
    """
    root = _cached_path(model.repo, model.files[0], model.revision)
    if root is None or not all(_cached_path(model.repo, f, model.revision) for f in model.files[1:]):
        raise Missing(model)
    return os.path.dirname(root)  # files[0] is root-level (enforced in __post_init__)


def present(model: Model) -> bool:
    """
    Reports whether the model is fully present in the local cache.

    The download script calls this to decide whether a model still needs
    downloading, and calls it again afterward to confirm that every file arrived.
    """
    try:
        resolve(model)
        return True
    except Missing:
        return False


def missing() -> list[Model]:
    """
    Returns the models that are not fully present in the local cache.

    present() checks one model; this checks every model in the manifest and returns
    the ones that are not present, so the preflight step can list them for the user.
    """
    return [model for model in MODELS if not present(model)]


def clip_weights(model: ClipModel) -> str:
    """The local CLIP weight file open_clip's `pretrained=` wants (resolve() gives the dir)."""
    return os.path.join(resolve(model), model.files[0])


def download(model: Model) -> None:
    """Fetch every required file into the local cache (NETWORK). Idempotent and revision-pinned:
    hf_hub_download fetches only what is absent, so this also REPAIRS a missing-file partial.
    Called only by scripts/download_models.py."""
    for name in model.files:
        hf_hub_download(model.repo, name, revision=model.revision)
