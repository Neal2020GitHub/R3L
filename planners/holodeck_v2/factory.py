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

from typing import List
import open_clip
from sentence_transformers import SentenceTransformer

from .rooms import FloorPlanGeneratorV2
from .walls import WallGeneratorV2
from .selector import HolodeckSelectorV2, HolodeckSmallSelectorV2

from retrievers.holodeck_v2.retriever import HolodeckRetrieverV2

from utils.holodeck_v2.types import ClipModelsDict
from utils.holodeck_v2.llm import OpenAIWithTracking
from utils.holodeck_v2.types import RoomDict, WallsDict, validate_room_dict, validate_walls_dict
from utils.models import CLIP, SBERT, clip_weights, resolve

__all__ = [
    "create_clip_models",
    "create_sbert_model",
    "create_holodeck_rooms",
    "create_holodeck_walls",
    "create_holodeck_selector",
    "create_holodeck_small_selector",
]

def create_clip_models() -> ClipModelsDict:
    weights = clip_weights(CLIP)  # local weight file, zero network
    clip_tokenizer = open_clip.get_tokenizer(CLIP.arch)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP.arch, pretrained=weights  # a file path ⇒ open_clip loads it locally, no hub call
    )
    return {
        "clip_tokenizer": clip_tokenizer,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
    }

def create_sbert_model() -> SentenceTransformer:
    return SentenceTransformer(resolve(SBERT), device="cpu", local_files_only=True)


def create_holodeck_rooms(
    *,
    llm: OpenAIWithTracking,
    query: str,
    used_assets: List[str] = [],
    visualize: bool = False,
) -> RoomDict:
    if not isinstance(llm, OpenAIWithTracking):
        raise TypeError(f"Invalid LLM: {type(llm)}")

    gen = FloorPlanGeneratorV2(llm, used_assets)
    room: RoomDict = gen.generate_rooms(query, visualize)
    validate_room_dict(room)
    return room


def create_holodeck_walls(query: str, llm: OpenAIWithTracking, room: RoomDict) -> WallsDict:
    if not isinstance(llm, OpenAIWithTracking):
        raise TypeError(f"Invalid LLM: {type(llm)}")
    validate_room_dict(room)

    gen = WallGeneratorV2(llm)
    walls: WallsDict = gen.generate_walls(query, room)
    validate_walls_dict(walls)
    return walls


def create_holodeck_selector(
    object_retriever: HolodeckRetrieverV2,
    llm: OpenAIWithTracking,
    *,
    floor_capacity_ratio: float = 0.4,
    wall_capacity_ratio: float = 0.5,
    object_size_tolerance: float = 0.8,
    similarity_threshold_floor: float = 31,
    similarity_threshold_wall: float = 31,
    thin_threshold: float = 3,
    consider_size: bool = True,
    used_assets: List[str] = [],
    random_selection: bool = False,
) -> HolodeckSelectorV2: 

    return HolodeckSelectorV2(
        object_retriever=object_retriever,
        llm=llm,
        floor_capacity_ratio=floor_capacity_ratio,
        wall_capacity_ratio=wall_capacity_ratio,
        object_size_tolerance=object_size_tolerance,
        similarity_threshold_floor=similarity_threshold_floor,
        similarity_threshold_wall=similarity_threshold_wall,
        thin_threshold=thin_threshold,
        consider_size=consider_size,
        used_assets=used_assets,
        random_selection=random_selection,
    )


def create_holodeck_small_selector(
    retriever: HolodeckRetrieverV2,
    llm: OpenAIWithTracking,
    *,
    clip_threshold: float = 30,
    size_threshold: float = 0.9,
    used_assets: List[str] = [],
) -> HolodeckSmallSelectorV2:

    return HolodeckSmallSelectorV2(
        retriever=retriever,
        llm=llm,
        clip_threshold=clip_threshold,
        size_threshold=size_threshold,
        used_assets=used_assets,
    )
    
