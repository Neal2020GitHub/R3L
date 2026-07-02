# Copyright 2023 Allen Institute for Artificial Intelligence
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from AllenAI Holodeck (https://github.com/allenai/Holodeck;
# Yang et al., CVPR 2024) and Holodeck 2.0 (Bian et al., 2025,
# arXiv:2508.05899). Adapted for the R3L pipeline. See the LICENSE file
# in this directory for the full Apache 2.0 terms.
#
# Modifications (where applicable) are Copyright (c) 2026 Yuqi Wang and
# Zhifeng Gu and licensed under the MIT License (see the repository root
# LICENSE).

import os
from pathlib import Path

import objathor.dataset

ABS_PATH_OF_HOLODECK = os.path.abspath(os.path.dirname(Path(__file__)))

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2024_08_16")

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser(f"data/objathor-assets")  # TODO: change this to your own path
)

ASSET_BASE_DIR = "./data/assets"  # TODO: change this to your own path

OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")

BASE_URL = objathor.dataset.DatasetSaveConfig(  # pyright: ignore[reportCallIssue]  # objathor stub marks BASE_BUCKET_URL required; it has a runtime default
    VERSION=ASSETS_VERSION
).VERSIONED_BUCKET_URL


if ASSETS_VERSION == "2024_08_16":
    os.makedirs(OBJATHOR_ASSETS_DIR, exist_ok=True)

# LLM_MODEL_NAME = "gpt-4o-2024-08-06"
# SMALL_LLM_MODEL_NAME = "gpt-4o-mini-2024-07-18"
LLM_MODEL_NAME = "gpt-5"
SMALL_LLM_MODEL_NAME = "gpt-5-mini"

DEBUGGING = os.environ.get("DEBUGGING", "0").lower() in ["1", "true", "True", "t", "T"]

# MULTIPROCESSING = os.environ.get("MULTIPROCESSING", "1").lower() in [
#     "1",
#     "true",
#     "t",
# ]
MULTIPROCESSING = False
