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

from .retriever import HolodeckRetrieverV2, SlimRetriever
from sentence_transformers import SentenceTransformer
from utils.holodeck_v2.types import ClipModelsDict

__all__ = [
    "create_holodeck_retriever",
    "create_slim_retriever",
]


def create_slim_retriever():
    """Create a lightweight retriever with only the database (no CLIP/SBERT)."""
    return SlimRetriever()


def create_holodeck_retriever(
        clip_models: ClipModelsDict,
        sbert_model: SentenceTransformer,
        retrieval_threshold: int = 28,
    ): 

    return HolodeckRetrieverV2(
        clip_model=clip_models["clip_model"],
        clip_preprocess=clip_models["clip_preprocess"],
        clip_tokenizer=clip_models["clip_tokenizer"],
        sbert_model=sbert_model,
        retrieval_threshold=retrieval_threshold,
    )
