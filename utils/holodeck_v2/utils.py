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

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from utils.asset import get_asset_metadata

import numpy as np



def get_bbox_dims(obj_data: Dict[str, Any]) -> Dict[str, float]:
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}


def get_bbox_dims_vec(obj_data: Dict[str, Any]) -> np.ndarray:
    bbox_info = get_bbox_dims(obj_data)
    return np.array([bbox_info["x"], bbox_info["y"], bbox_info["z"]])


# def get_secondary_properties(obj_data: Dict[str, Any]):
    # am = get_asset_metadata(obj_data)
    # return am["secondaryProperties"]


def unpack_kwargs(func):
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], Dict):
            kwargs.update(args[0])
            args = args[1:]
        return func(*args, **kwargs)

    return wrapper


def random_select(candidates: List[Tuple[str, float]]) -> Tuple[str, float]:
    scores = [candidate[1] for candidate in candidates]
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    mean = scores_tensor.mean()
    std = scores_tensor.std(unbiased=False)
    if std > 0:
        logits = (scores_tensor - mean) / (std + 1e-8)
    else:
        logits = torch.zeros_like(scores_tensor)
    probas = F.softmax(logits, dim=0)
    selected_index = int(torch.multinomial(probas, 1).item())
    selected_candidate = candidates[selected_index]
    return selected_candidate

