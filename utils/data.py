"""
Lists the objathor data files the retriever needs, and checks that they are
present in the local assets directory.

The retriever loads two kinds of objathor data to build its asset pool:

- **Features.** Pre-computed CLIP image/text and SBERT text vectors for every
  objaverse object, one pickle per encoder. The retriever reads these straight
  into a single tensor and cannot run without them.
- **Annotations.** The metadata (category, onFloor/onWall flags, bounding box,
  descriptions) that the layout selectors consult when placing each object.

A file is "present" when it exists, is non-empty, and is readable. This is the
same scope `utils.models` uses for model checkpoints: presence, not contents.
Unlike Hugging Face, objathor ships its data as a tar that `tar -xf` unpacks
without verifying, so a truncated download could leave a half-written pickle.
That gap is closed not here but in the download path: `scripts/download_data.py`
re-runs this presence check after fetching (a self-test), so a green preflight
after a download means the files actually landed.

One function, `present()`, performs this check. Three parts of the program call
it, so they always agree on whether the data is present:
- The preflight step runs before the pipeline starts. It calls `missing()` to
  warn the user about absent data before any slow work begins.
- The retriever loads the data directly (the paths it reads come from
  `utils.holodeck_v2.constants`, which resolves them from the same env var as
  `base_dir()` here).
- The download script calls `present()` to decide what still needs downloading,
  and calls it again afterward to confirm every file arrived.

This module imports no heavy libraries (no torch, objathor, or compress_pickle),
so the preflight check stays fast. Only `download()` uses the network, and it
imports objathor lazily for the same reason.

The base directory and version are read from the environment, exactly as
`utils.holodeck_v2.constants` does, so preflight and the retriever target the
same tree. There is one version (the objathor assets version); no second source
of truth exists.
"""

import os
from dataclasses import dataclass

# The one place the remediation command lives, shared by the preflight screen.
DOWNLOAD_CMD = "uv run python scripts/download_data.py"

# The single objathor assets version this release uses. Mirrors the default in
# utils.holodeck_v2.constants; read from the same env var so both agree.
ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2024_08_16")


def base_dir() -> str:
    """The objathor assets root, resolved exactly as
    utils.holodeck_v2.constants does, but absolute. Reading it fresh on every
    call (not at import) means a user who sources env.sh after launching Python
    sees the updated path; the abspath removes any dependence on the process's
    working directory."""
    return os.path.abspath(
        os.environ.get("OBJATHOR_ASSETS_BASE_DIR", "data/objathor-assets")
    )


@dataclass(frozen=True)
class Dataset:
    name: str
    # Paths relative to the base directory, so the record stays env-independent.
    files: tuple[str, ...]

    def __post_init__(self):
        # Every file lives under the versioned assets dir; a path not starting
        # with the version segment is a manifest typo.
        if self.files and not all(f.startswith(ASSETS_VERSION) for f in self.files):
            raise ValueError(
                f"{self.name}: every file must sit under {ASSETS_VERSION}/, "
                f"got {self.files}"
            )


# The objathor data the retriever hard-loads: the two feature pickles (CLIP +
# SBERT) and the annotations archive. All three live under the versioned assets
# directory. Annotations are required (the layout selectors key off their
# fields), not soft.
OBJATHOR = Dataset(
    name="objathor",
    files=(
        f"{ASSETS_VERSION}/features/clip_features.pkl",
        f"{ASSETS_VERSION}/features/sbert_features.pkl",
        f"{ASSETS_VERSION}/annotations.json.gz",
    ),
)

DATA: tuple[Dataset, ...] = (OBJATHOR,)


def _path(rel: str) -> str:
    return os.path.join(base_dir(), rel)


def _is_present(rel: str) -> bool:
    path = _path(rel)
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0 and os.access(path, os.R_OK)
    except OSError:
        return False


def missing() -> list[Dataset]:
    """
    Returns the datasets that are not fully present in the local assets directory.

    The preflight step calls this to list absent data for the user before the
    pipeline starts.
    """
    result = []
    for dataset in DATA:
        absent = [f for f in dataset.files if not _is_present(f)]
        if absent:
            result.append(dataset)
    return result


def paths(dataset: Dataset) -> list[str]:
    """Absolute paths of a dataset's files. Used by preflight to print the
    specific files that are missing."""
    return [_path(f) for f in dataset.files]


def present(dataset: Dataset = OBJATHOR) -> bool:
    """
    Reports whether the dataset is fully present in the local assets directory.

    The download script calls this to decide whether the data still needs
    downloading, and calls it again afterward to confirm that every file arrived.
    """
    return all(_is_present(f) for f in dataset.files)


def download(dataset: Dataset = OBJATHOR) -> None:
    """Fetch the dataset into the local assets directory (NETWORK). Idempotent:
    objathor's loaders fetch only what is absent, so this also repairs a
    missing-file partial. Called only by scripts/download_data.py.

    Lazily imports objathor to keep this module import-light for preflight.

    A subtlety of objathor's `load_features_dir`: it short-circuits and skips the
    download whenever the features directory already holds ANY .pkl. So a half-
    unpacked tar (one pickle present, another truncated away) looks "done" and
    never gets refetched. When the features pickles are missing we therefore
    clear the directory first, forcing a clean re-fetch.
    """
    import shutil
    from objathor.dataset import DatasetSaveConfig, load_annotations_path, load_features_dir

    dsc = DatasetSaveConfig(VERSION=ASSETS_VERSION, BASE_PATH=base_dir())

    # Clear a partial features directory before refetching (see docstring).
    features_dir = os.path.join(base_dir(), ASSETS_VERSION, "features")
    features_present = all(
        _is_present(f)
        for f in dataset.files
        if f.startswith(f"{ASSETS_VERSION}/features/")
    )
    if not features_present and os.path.isdir(features_dir):
        shutil.rmtree(features_dir, ignore_errors=True)

    load_features_dir(dsc)
    load_annotations_path(dsc)
