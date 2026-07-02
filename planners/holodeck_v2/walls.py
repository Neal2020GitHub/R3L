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

import copy
import random

import numpy as np
from colorama import Fore
from langchain_core.prompts import PromptTemplate
from shapely.geometry import Polygon, Point
from typing import List

from utils.holodeck_v2 import prompts
from utils.holodeck_v2.llm import OpenAIWithTracking
from utils.holodeck_v2.types import RoomDict, WallsDict, WallSegment, validate_room_dict, validate_walls_dict


class WallGeneratorV2:
    def __init__(self, llm: OpenAIWithTracking):
        self.json_template = {
            "id": None,
            "roomId": None,
            "polygon": [],
        }
        self.llm = llm
        self.wall_height_template = PromptTemplate(
            input_variables=["input"], template=prompts.WALL_HEIGHT_PROMPT
        )
        self.used_assets = []

    def generate_walls(self, query: str, room: RoomDict) -> WallsDict:
        validate_room_dict(room)
        wall_height = self.get_wall_height(query)

        room_id = room["id"]
        full_vertices = room.get("full_vertices") or room["vertices"]

        walls = []
        for j in range(len(full_vertices)):
            a = full_vertices[j]
            b = full_vertices[(j + 1) % len(full_vertices)]

            wall = copy.deepcopy(self.json_template)
            wall["roomId"] = room_id
            wall["polygon"] = self.generate_wall_polygon(a, b, wall_height)

            width, direction = self.get_wall_direction(a, b, full_vertices)
            wall["width"] = width
            wall["height"] = wall_height
            wall["direction"] = direction
            wall["segment"] = [a, b]
            wall["id"] = f"wall|{room_id}|{direction}|{j}"
            wall["connected_rooms"] = []
            walls.append(wall)

        # Unconditionally add exterior walls (legacy interface)
        updated_walls: List[WallSegment] = []
        for wall in walls:
            exterior_wall = copy.deepcopy(wall)
            exterior_wall["id"] = wall["id"] + "|exterior"
            exterior_wall["polygon"] = wall["polygon"][::-1]
            exterior_wall["segment"] = wall["segment"][::-1]
            wall["connect_exterior"] = exterior_wall["id"]
            updated_walls.append(exterior_wall)
            updated_walls.append(wall)
        walls = updated_walls

        result: WallsDict = {
            "wall_height": wall_height, 
            "walls": walls
        }
        validate_walls_dict(result)
        return result

    def get_wall_height(self, query: str):
        # get wall height
        wall_height_prompt = self.wall_height_template.format(input=query)

        response_text = self.llm(wall_height_prompt).split("\n")[0].strip()

        try:
            wall_height = float(response_text)
        except:
            print(f"{Fore.YELLOW}Warning: LLM wall_height failed, using random height{Fore.RESET}")
            wall_height = round(
                random.uniform(2.5, 4.5), 1
            )  # if failed, random height between 2.5 and 4.5

        wall_height = min(
            max(wall_height, 2.0), 4.5
        )  # limit the wall height between 2.0 and 4.5

        print(f"\nUser: {wall_height_prompt}\n")
        print(f"{Fore.GREEN}AI: The wall height is {wall_height}{Fore.RESET}")

        return wall_height

    def generate_wall_polygon(self, point, next_point, wall_height):
        wall_polygon = []
        # add the base point
        wall_polygon.append({"x": point[0], "y": 0, "z": point[1]})
        # add the top point (with the same x and z, but y = wall_height)
        wall_polygon.append({"x": point[0], "y": wall_height, "z": point[1]})
        # add the top point of the next base point
        wall_polygon.append({"x": next_point[0], "y": wall_height, "z": next_point[1]})
        # add the next base point
        wall_polygon.append({"x": next_point[0], "y": 0, "z": next_point[1]})
        return wall_polygon


    def get_wall_direction(self, wall_endpoint1, wall_endpoint2, room_vertices):
        wall_width = np.linalg.norm(np.array(wall_endpoint1) - np.array(wall_endpoint2))

        wall_direction = None
        room_polygon = Polygon(room_vertices)
        wall_center = [
            (wall_endpoint1[0] + wall_endpoint2[0]) / 2,
            (wall_endpoint1[1] + wall_endpoint2[1]) / 2,
        ]

        if wall_endpoint1[1] == wall_endpoint2[1]:
            extend_point_1 = [wall_center[0], wall_center[1] + 0.01]
            extend_point_2 = [wall_center[0], wall_center[1] - 0.01]
            # check which point is in room polygon
            if room_polygon.contains(Point(extend_point_1)):
                wall_direction = "south"
            elif room_polygon.contains(Point(extend_point_2)):
                wall_direction = "north"

        elif wall_endpoint1[0] == wall_endpoint2[0]:
            extend_point_1 = [wall_center[0] + 0.01, wall_center[1]]
            extend_point_2 = [wall_center[0] - 0.01, wall_center[1]]
            # check which point is in room polygon
            if room_polygon.contains(Point(extend_point_1)):
                wall_direction = "west"
            elif room_polygon.contains(Point(extend_point_2)):
                wall_direction = "east"

        return wall_width, wall_direction

