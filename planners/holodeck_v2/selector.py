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

import ast
import copy
import json
import re
from typing import Dict, List, Optional, Sequence, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from colorama import Fore
from objathor.utils.queries import Text
from shapely import Polygon

from retrievers.holodeck_v2.retriever import HolodeckRetrieverV2

from utils.holodeck_v2 import prompts
from utils.holodeck_v2.constants import SMALL_LLM_MODEL_NAME
from utils.holodeck_v2.llm import OpenAIWithTracking
from utils.holodeck_v2.types import (
    object_plan_from_dict,
    RoomDict,
    WallsDict,
    ObjectPlan,
    ObjectInfo,
    AssetList,
    SmallObjectInfo,
    validate_room_dict,
    validate_walls_dict,
)
from utils.holodeck_v2.utils import (
    get_bbox_dims,
    get_bbox_dims_vec,
    random_select,
)
from utils.asset import get_annotations

EXPECTED_OBJECT_ATTRIBUTES = [
    "description",
    "location",
    "size",
    "quantity",
    "variance_type",
    "objects_on_top",
]

MAX_SINGLE_OBJ_QUANTITY = 10


COMPARISON_TEMPLATE = """\
Below I have {NUM} descriptions of objects. Formatted as a list:
1. <DESCRIPTION 1>
2. <DESCRIPTION 2>
... 

For each description, I want you to compare how similar the object is to a {OBJECT_NAME} and answer with a number between 0 (very dissimilar) and 10 (very similar) and no other text.
I.e. respond with
1. <NUMBER 1>
2. <NUMBER 2>
...

Object descriptions:
{ALL_DESCRIPTIONS}
"""

_print_lock = threading.Lock()

def _report_candidates_status(candidates: Sequence, obj_name: str, selector_name: str): 
    with _print_lock:
        if len(candidates) == 0:
            print(f"{Fore.RED}{selector_name}: No candidates found for {obj_name}{Fore.RESET}", flush=True)
        else: 
            print(f"{Fore.GREEN}{selector_name}: Found {len(candidates)} candidates for {obj_name}{Fore.RESET}", flush=True)


def _compare_object_name_to_descriptions( # internal helper function
    object_name: str,
    asset_ids: Sequence[str],
    database: Dict[str, Dict[str, Any]],
    llm: OpenAIWithTracking,
):
    descriptions = []
    for asset_id in asset_ids:
        description = get_annotations(database[asset_id])["description"]
        if description is None:
            description = "a " + get_annotations(database[asset_id])["description_auto"]

        descriptions.append(description.replace("\n", " "))

    descriptions = [
        f"{index+1}. {description}" for index, description in enumerate(descriptions)
    ]
    all_descriptions = "\n".join(descriptions)

    output = llm.get_answer(
        model=SMALL_LLM_MODEL_NAME,
        messages=[
            Text(
                "You are a helpful AI assistant that expertly helps people compare objects.",
                role="system",
            ),
            Text(
                COMPARISON_TEMPLATE.format(
                    NUM=len(descriptions),
                    OBJECT_NAME=object_name,
                    ALL_DESCRIPTIONS=all_descriptions,
                ),
                role="user",
            ),
        ],
    )

    line_num_to_score = {}
    for line in output.split("\n"):
        try:
            line_num, score = line.split(".")
            line_num = int(line_num)
            score = float(score)
            line_num_to_score[line_num] = score
        except ValueError:
            pass

    return [line_num_to_score.get(index + 1, -1) for index in range(len(descriptions))]


class HolodeckSelectorV2:
    def __init__(
        self, 
        object_retriever: HolodeckRetrieverV2, 
        llm: OpenAIWithTracking,

        # std objs
        floor_capacity_ratio: float = 0.4,
        wall_capacity_ratio: float = 0.5,
        object_size_tolerance: float = 0.8,
        similarity_threshold_floor: float = 31, # needs tuning
        similarity_threshold_wall: float = 31, # needs tuning
        thin_threshold: float = 3,
        consider_size: bool = True,
        size_buffer: int = 10,
        used_assets: Optional[List[str]] = None,
        random_selection: bool = False,
    ) -> None:
        assert isinstance(object_retriever, HolodeckRetrieverV2)
        assert isinstance(llm, OpenAIWithTracking)
        
        # object retriever
        self.object_retriever = object_retriever
        self.database = object_retriever.database

        # language model and prompt templates
        self.llm = llm

        # hyperparameters
        self.floor_capacity_ratio = floor_capacity_ratio
        self.wall_capacity_ratio = wall_capacity_ratio
        self.object_size_tolerance = object_size_tolerance
        self.similarity_threshold_floor = similarity_threshold_floor
        self.similarity_threshold_wall = similarity_threshold_wall
        self.thin_threshold = thin_threshold
        self.used_assets = list(used_assets) if used_assets else []
        self.consider_size = consider_size
        self.size_buffer = size_buffer

        self.random_selection = random_selection

    def select_objects(
        self, 
        query: str,
        room: RoomDict,
        walls: WallsDict,
        get_floor_objects: bool = True,
        get_wall_objects: bool = True,
        additional_requirements="N/A"
    ) -> Tuple[AssetList, AssetList, str, ObjectPlan]:
        validate_room_dict(room)
        validate_walls_dict(walls)
        if not (get_floor_objects or get_wall_objects):
            raise ValueError("Need get_floor_objects or get_wall_objects")

        rooms_type = room["roomType"]
        room_area = self.get_room_area(room)
        room_size = self.get_room_size(room, walls["wall_height"])  # [length, height, width]
        room_perimeter = self.get_room_perimeter(room)
        # room_vertices = [(x * 100, y * 100) for (x, y) in room["vertices"]]
        room_floor_capacity_init = [room_area * self.floor_capacity_ratio, 0.0]
        room_wall_capacity_init = [room_perimeter * self.wall_capacity_ratio, 0.0]  # only consider wall width

        results = self.plan_room(
            query=query,
            room_type=rooms_type,
            additional_requirements=additional_requirements,
            room_size=room_size,
            room_floor_capacity_init=room_floor_capacity_init,
            room_wall_capacity_init=room_wall_capacity_init,
            get_floor_objects=get_floor_objects,
            get_wall_objects=get_wall_objects,
        )
        floor_objects, wall_objects, design, selection_plan = results


        print(
            f"\n{Fore.GREEN}AI: Here is the object selection plan:\n{selection_plan}{Fore.RESET}",
            end="\n\n\n",
        )
        return floor_objects, wall_objects, design, selection_plan

    def plan_room(
        self,
        query: str,
        room_type: str,
        additional_requirements,
        room_size: Tuple[float, float, float],
        room_floor_capacity_init: Tuple[float, float],
        room_wall_capacity_init: Tuple[float, float],
        *,
        get_floor_objects: bool = True,
        get_wall_objects: bool = True,
    ) -> Tuple[AssetList, AssetList, str, ObjectPlan]:

        def print_plan_summary(plan: ObjectPlan): 
            names = list(plan.keys())
            width = max((len(f"[{i}]{name}:") for i, name in enumerate(names, 1)), default=0)
            for i, (object_name, object_info) in enumerate(plan.items(), 1):
                prefix = f"[{i}]{object_name}:"
                print(f"{Fore.BLUE}{prefix}{Fore.RESET}{' ' * (width - len(prefix) + 1)}{object_info['description']}")
            print('\n')

        print(f"\n{Fore.GREEN}AI: Selecting objects for {room_type}...{Fore.RESET}\n")

        room_size_str = (
            f"{int(room_size[0])*100}cm in length,"
            f" {int(room_size[2])*100}cm in width,"  # room_size: [length, height, width]
            f" {int(room_size[1])*100}cm in height"
        )
        print("Room size: ", room_size_str)

        location_instruction = self._location_instruction(
            get_floor_objects, get_wall_objects
        )

        messages = [
            Text(
                content=(
                    prompts.WALL_FLOOR_AND_SMALL_OBJECT_SELECTION_SYSTEM_PROMPT.replace(
                        "REQUIREMENTS", additional_requirements
                    )
                ),
                role="system",
            ),
        ]

        user_message = (
            prompts.WALL_FLOOR_AND_SMALL_OBJECT_SELECTION_USER_PROMPT.replace(
                "INPUT", query
            )
            .replace("ROOM_TYPE", room_type)
            .replace("ROOM_SIZE", room_size_str)
        )
        if additional_requirements.strip().lower() != "n/a":
            user_message += (
                " "
                + prompts.WALL_FLOOR_AND_SMALL_OBJECT_SELECTION_USER_EXTRA_REQUIREMENTS_PROMPT.replace(
                    "REQUIREMENTS", additional_requirements
                )
            )
        if location_instruction:
            user_message += " " + location_instruction

        messages.append(Text(content=user_message, role="user"))
        design, plan_1 = self._chat_for_plan(
            messages, get_floor_objects, get_wall_objects, room_type
        )
        new_plan = plan_1
        
        print(f"\n\n{Fore.GREEN}AI: Initial selection plan:\n{Fore.RESET}")
        print_plan_summary(plan_1)

        (
            floor_objects,
            floor_capacity,
            wall_objects,
            wall_capacity,
        ) = self.get_objects_by_room(
            parsed_plan=plan_1,
            room_size=room_size,
            floor_capacity=room_floor_capacity_init,
            wall_capacity=room_wall_capacity_init,
            get_wall_objects=get_wall_objects,
            get_floor_objects=get_floor_objects,
        )

        # if utilization is less than 80%, add more objects
        required_floor_capacity_percentage = 0.8
        if floor_capacity[1] / floor_capacity[0] < required_floor_capacity_percentage:
            print(
                f"{Fore.RED}USER: The used floor capacity of {room_type} is {floor_capacity[1]:.2g}m^2,"
                f" which is less than {100*required_floor_capacity_percentage:.0f}% of the total floor capacity"
                f" {floor_capacity[0]:.2g}m^2. Asking the LLM to add additional objects."
                f"{Fore.RESET}"
            )
            followup_prompt = prompts.WALL_FLOOR_AND_SMALL_OBJECT_SELECTION_PROMPT_ADD_MORE_OBJECTS_FOLLOWUP.format(
                room=room_type,
            )
            if location_instruction:
                followup_prompt += " " + location_instruction

            messages.append(Text(content=followup_prompt, role="user"))
            _, plan_2 = self._chat_for_plan(
                messages,
                get_floor_objects,
                get_wall_objects,
                f"{room_type} follow-up",
            )

            new_plan = copy.deepcopy(new_plan)
            if plan_2 is not None:
                for object in plan_2:
                    new_plan[object] = plan_2[object]
            else: 
                print(
                    f"{Fore.RED}AI: Replanning failed, will use 1st plan.{Fore.RESET}"
                )
            
            print(f"\n\n{Fore.GREEN}AI: Updated selection plan:\n{Fore.RESET}")
            print_plan_summary(new_plan)

            floor_objects, floor_capacity, wall_objects, wall_capacity = self.get_objects_by_room(
                parsed_plan=new_plan,
                room_size=room_size,
                floor_capacity=room_floor_capacity_init,
                wall_capacity=room_wall_capacity_init,
                get_wall_objects=get_wall_objects,
                get_floor_objects=get_floor_objects,
            )

            if floor_capacity[1] / floor_capacity[0] < required_floor_capacity_percentage:
                print(
                    f"{Fore.RED}WARNING: The used floor capacity of {room_type} is {floor_capacity[1]:.2g}m^2,"
                    f" still less than {100*required_floor_capacity_percentage:.0f}% of the total floor capacity"
                    f" {floor_capacity[0]:.2g}m^2."
                    f"{Fore.RESET}"
                )
            else: 
                print(
                    f"{Fore.GREEN}INFO: The used floor capacity of {room_type} is {floor_capacity[1]:.2g}m^2,"
                    f" which is greater than {100*required_floor_capacity_percentage:.0f}% of the total floor capacity"
                    f" {floor_capacity[0]:.2g}m^2."
                    f"{Fore.RESET}"
                )

        return floor_objects, wall_objects, design, new_plan

    def extract_json(self, input_string: str) -> Optional[ObjectPlan]:
        # Using regex to identify the JSON structure in the string
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            extracted_json = json_match.group(0)

            # Convert the extracted JSON string into a Python dictionary
            json_dict = None
            try:
                json_dict = json.loads(extracted_json)
            except:
                try:
                    json_dict = ast.literal_eval(extracted_json)
                except:
                    pass

            if json_dict is None:
                print(
                    f"{Fore.RED}[ERROR] while parsing the JSON for:\n{input_string}{Fore.RESET}",
                    flush=True,
                )
                return None

            json_dict = object_plan_from_dict(json_dict)

            return json_dict

        else:
            print(f"No valid JSON found in:\n{input_string}", flush=True)
            return None

    def _location_instruction(
        self, allow_floor: bool, allow_wall: bool
    ) -> str:
        if allow_floor and not allow_wall:
            return "Include only floor objects; omit wall objects."
        if allow_wall and not allow_floor:
            return "Include only wall objects; omit floor objects."
        if allow_floor and allow_wall:
            return "You are free to include both floor and wall objects."
        assert False # cant happen

    def _chat_for_plan(
        self,
        messages: List[Text],
        allow_floor: bool,
        allow_wall: bool,
        context: str,
    ) -> Tuple[str, ObjectPlan]:
        for attempt in range(3):
            response = self.llm.get_answer(messages)
            messages.append(Text(response, role="assistant"))

            design = response.split('{')[0]
            design = design.split("```")[0].strip()

            print(design)
            plan = self.extract_json(response.lower())
            assert plan is not None, f"Failed to extract plan for {context}."
            try:
                return design, self._filter_plan_locations(plan, allow_floor, allow_wall, context)
            except ValueError as err:
                print(f"{Fore.RED}{err}. Retrying...{Fore.RESET}")
                if attempt == 2:
                    raise
                retry_msg = self._retry_instruction(allow_floor, allow_wall)
                if retry_msg:
                    messages.append(Text(retry_msg, role="user"))
        raise RuntimeError("Unreachable")

    def _filter_plan_locations(
        self,
        plan: ObjectPlan,
        allow_floor: bool,
        allow_wall: bool,
        context: str,
    ) -> ObjectPlan:
        allowed = set()
        if allow_floor:
            allowed.add("floor")
        if allow_wall:
            allowed.add("wall")

        if not allowed:
            if plan:
                raise ValueError(
                    f"{context} returned objects but both floor and wall objects are disabled."
                )
            return plan

        disallowed = [
            name
            for name, info in plan.items()
            if str(info.get("location", "")).lower() not in allowed
        ]

        if disallowed:
            raise ValueError(
                f"{context} contains objects with disallowed locations: {disallowed}"
            )

        return plan

    def _retry_instruction(self, allow_floor: bool, allow_wall: bool) -> str:
        if allow_floor and not allow_wall:
            return "Revise JSON to include only floor objects; remove wall entries."
        if allow_wall and not allow_floor:
            return "Revise JSON to include only wall objects; remove floor entries."
        assert False

    def get_objects_by_room(
        self,
        parsed_plan: ObjectPlan,
        room_size: Tuple[float, float, float],
        floor_capacity: Tuple[float, float],
        wall_capacity: Tuple[float, float],
        *,
        get_wall_objects: bool,
        get_floor_objects: bool,
    ) -> Tuple[Optional[AssetList], Tuple[float, float], Optional[AssetList], Tuple[float, float]]:
        # get the floor and wall objects
        floor_object_list: List[ObjectInfo] = []
        wall_object_list: List[ObjectInfo] = []
        for object_name, object_info in parsed_plan.items():
            object_info["object_name"] = object_name
            if object_info["location"] == "floor":
                floor_object_list.append(object_info)
            else:
                wall_object_list.append(object_info)

        floor_objects, floor_capacity = self.get_floor_objects(
            floor_object_list, floor_capacity, room_size
        ) if get_floor_objects else ([], floor_capacity)
        wall_objects, wall_capacity = self.get_wall_objects(
            wall_object_list, wall_capacity, room_size
        ) if get_wall_objects else ([], wall_capacity)

        return floor_objects, floor_capacity, wall_objects, wall_capacity

    def get_room_size(self, room, wall_height):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        x_dim = max(x_values) - min(x_values)
        z_dim = max(z_values) - min(z_values)

        # Ensure length >= width
        if x_dim > z_dim:
            return (x_dim, wall_height, z_dim)
        else:
            return (z_dim, wall_height, x_dim)

    def get_room_area(self, room):
        room_vertices = room["vertices"]
        room_polygon = Polygon(room_vertices)
        return room_polygon.area

    def get_room_perimeter(self, room):
        room_vertices = room["vertices"]
        room_polygon = Polygon(room_vertices)
        return room_polygon.length

    def get_floor_objects(
        self,
        floor_object_list: List[ObjectInfo],
        floor_capacity: Tuple[float, float],
        room_size: Tuple[float, float, float],
    ) -> Tuple[AssetList, Tuple[float, float]]:
        selected_floor_objects_all = []
        for floor_object in sorted(
            floor_object_list, key=lambda fo: -1 * fo["importance"]
        ):
            object_name = floor_object["object_name"]
            object_description = floor_object["description"]
            object_size = floor_object["size"]
            importance = floor_object["importance"]
            quantity = min(floor_object["quantity"], MAX_SINGLE_OBJ_QUANTITY)
            variance_type = floor_object["variance_type"]

            candidates = self.object_retriever.retrieve_with_name_and_desc(
                object_names=[object_name],
                object_descriptions=[object_description],
                threshold=self.similarity_threshold_floor,
            )

            candidates = [
                candidate
                for candidate, annotation in zip(
                    candidates,
                    [
                        get_annotations(self.database[candidate[0]])
                        for candidate in candidates
                    ],
                )
                if annotation["onFloor"]  # only select objects on the floor
                and (
                    not annotation["onCeiling"]
                )  # only select objects not on the ceiling
                and all(  # ignore doors and windows and frames
                    k not in annotation["category"].lower()
                    for k in ["door", "window", "frame"]
                )
            ]

            # check if the object is too big
            candidates = self.check_object_size(candidates, room_size)

            # Check candidates actually match the object name
            candidates = candidates[:15]
            candidate_scores = _compare_object_name_to_descriptions(
                object_name=object_name.replace("_", " "),
                asset_ids=[candidate[0] for candidate in candidates],
                database=self.database,
                llm=self.llm,
            )
            score_and_val_candidate_list = list(zip(candidate_scores, candidates))
            score_and_val_candidate_list.sort(key=lambda x: x[0], reverse=True)

            candidates = [
                candidate
                for candidate_scores, candidate in score_and_val_candidate_list
                if candidate_scores >= 5
            ]

            # NOTE: check_floor_placement removed here.

            _report_candidates_status(candidates, object_name, "Floor Selector")
            if len(candidates) == 0:
                continue

            # remove used assets
            # if all candidates are used, use the top one as fallback
            top_one_candidate = candidates[0]
            if len(candidates) > 1:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(candidates) == 0:
                candidates = [top_one_candidate]

            # consider object size difference
            if object_size is not None and self.consider_size:
                candidates = self.object_retriever.compute_size_difference(
                    object_size, candidates
                )

            candidates = candidates[
                :MAX_SINGLE_OBJ_QUANTITY
            ]  # only select top 10 candidates

            selected_asset_ids = []

            if variance_type == "same":
                selected_candidate = random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = random_select(candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(candidates) > 1:
                        candidates.remove(selected_candidate)
            else:
                raise NotImplementedError(
                    f"Variance type {variance_type} is not supported."
                )

            for i in range(quantity):
                selected_asset_id = selected_asset_ids[i]
                object_id = f"{object_name}-{i}"
                selected_floor_objects_all.append(
                    (object_id, selected_asset_id, importance)
                )

        # reselect objects if they exceed floor capacity
        selected_floor_objects_filtered = []
        for object_name, selected_asset_id, importance in selected_floor_objects_all:
            x_size, _, z_size = get_bbox_dims_vec(self.database[selected_asset_id])
            selected_asset_area = x_size * z_size
            if (
                floor_capacity[1] + selected_asset_area > floor_capacity[0]
                and len(selected_floor_objects_filtered) > 0
            ):
                print(f"{object_name} {selected_asset_id} exceeds floor capacity")
            else:
                selected_floor_objects_filtered.append(
                    (object_name, selected_asset_id, importance)
                )
                floor_capacity = (
                    floor_capacity[0],
                    floor_capacity[1] + selected_asset_area,
                )

        floor_objects = [
            (on, aid)
            for (on, aid, _) in sorted(
                selected_floor_objects_filtered, key=lambda x: -x[-1]
            )
        ] # sort by importance (descending order)
        return floor_objects, floor_capacity

    def get_wall_objects(
        self,
        wall_object_list: List[ObjectInfo],
        wall_capacity: Tuple[float, float],
        room_size: Tuple[float, float, float],
    ) -> Tuple[AssetList, Tuple[float, float]]:
        selected_wall_objects_all = []
        for wall_object in wall_object_list:
            object_name = wall_object["object_name"]
            object_description = wall_object["description"]
            object_size = wall_object["size"]
            importance = wall_object["importance"]
            quantity = min(wall_object["quantity"], 10)
            variance_type = wall_object["variance_type"]

            candidates = self.object_retriever.retrieve_with_name_and_desc(
                object_names=[object_name],
                object_descriptions=[object_description],
                threshold=self.similarity_threshold_wall,
            )

            # check on wall objects
            candidates = [
                candidate
                for candidate in candidates
                if get_annotations(self.database[candidate[0]])["onWall"] == True
            ]  # only select objects on the wall

            # ignore doors and windows
            candidates = [
                candidate
                for candidate in candidates
                if "door"
                not in get_annotations(self.database[candidate[0]])["category"].lower()
            ]
            candidates = [
                candidate
                for candidate in candidates
                if "window"
                not in get_annotations(self.database[candidate[0]])["category"].lower()
            ]

            # check if the object is too big
            candidates = self.check_object_size(candidates, room_size)

            # check thin objects
            candidates = self.check_thin_object(candidates)

            # NOTE: check_wall_placement removed here.

            _report_candidates_status(candidates, object_name, "Wall Selector")
            if len(candidates) == 0:
                continue

            # remove used assets
            top_one_candidate = candidates[0]
            if len(candidates) > 1:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(candidates) == 0:
                candidates = [top_one_candidate]

            # consider object size difference
            if object_size is not None and self.consider_size:
                candidates = self.object_retriever.compute_size_difference(
                    object_size, candidates
                )

            candidates = candidates[:10]  # only select top 10 candidates

            selected_asset_ids = []
            if variance_type == "same":
                selected_candidate = random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = random_select(candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(candidates) > 1:
                        candidates.remove(selected_candidate)
            else:
                raise NotImplementedError(
                    f"Variance type {variance_type} is not supported."
                )

            for i in range(quantity):
                selected_asset_id = selected_asset_ids[i]
                object_id = f"{object_name}-{i}"
                selected_wall_objects_all.append(
                    (object_id, selected_asset_id, importance)
                )

        # reselect objects if they exceed wall capacity, consider the diversity of objects
        selected_wall_objects_filtered = []
        for object_name, selected_asset_id, importance in selected_wall_objects_all:
            selected_asset_capacity, _, _ = get_bbox_dims_vec(
                self.database[selected_asset_id]
            )
            if (
                wall_capacity[1] + selected_asset_capacity > wall_capacity[0]
                and len(selected_wall_objects_filtered) > 0
            ):
                print(f"{object_name} {selected_asset_id} exceeds wall capacity")
            else:
                selected_wall_objects_filtered.append(
                    (object_name, selected_asset_id, importance)
                )
                wall_capacity = (
                    wall_capacity[0],
                    wall_capacity[1] + selected_asset_capacity,
                )

        wall_objects = [
            (on, aid)
            for (on, aid, _) in sorted(
                selected_wall_objects_filtered, key=lambda x: -x[-1]
            )
        ] # sort by importance (descending order)
        return wall_objects, wall_capacity

    def check_object_size(self, candidates, room_size: Tuple[float, float, float]):
        valid_candidates = []
        for candidate in candidates:
            dimension = get_bbox_dims(self.database[candidate[0]])
            size = [dimension["x"], dimension["y"], dimension["z"]]
            if size[2] > size[0]:
                size = [size[2], size[1], size[0]]  # make sure that x > z

            if size[0] > room_size[0] * self.object_size_tolerance:
                continue
            if size[1] > room_size[1] * self.object_size_tolerance:
                continue
            if size[2] > room_size[2] * self.object_size_tolerance:
                continue
            if size[0] * size[2] > room_size[0] * room_size[2] * 0.5:
                continue  # TODO: consider using the floor area instead of the room area

            valid_candidates.append(candidate)

        return valid_candidates

    def check_thin_object(self, candidates):
        valid_candidates = []
        for candidate in candidates:
            dimension = get_bbox_dims(self.database[candidate[0]])
            size = [dimension["x"], dimension["y"], dimension["z"]]
            if size[2] > min(size[0], size[1]) * self.thin_threshold:
                continue
            valid_candidates.append(candidate)
        return valid_candidates


class HolodeckSmallSelectorV2: 
    def __init__(
        self, 
        retriever: HolodeckRetrieverV2,
        llm: OpenAIWithTracking,
        clip_threshold: float = 30,
        size_threshold: float = 0.9,
        used_assets: Optional[List[str]] = None,
    ):
        assert isinstance(retriever, HolodeckRetrieverV2)
        assert isinstance(llm, OpenAIWithTracking)

        self.retriever = retriever
        self.llm = llm
        self.db = retriever.database
        self.clip_threshold = clip_threshold
        self.size_threshold = size_threshold
        self.used_assets = list(used_assets) if used_assets else []
    
    def select_small_objects(
        self, 
        plan: ObjectPlan,
        floor_objs: AssetList,
        wall_objs: AssetList,
    ) -> Dict[str, AssetList]: 
        
        parent_list = floor_objs + wall_objs
        # Parent: refer to large floor / wall objects selected in `HolodeckSelectorV2.plan_room`
        # Child: refer to small objects placed on top of parent surfaces

        results = {}

        # object id (pa_obj_id) is unique 
        # object name (pa_obj_name) not necessarily unique
        # example: 
        # - object id: "couch-0", "couch-1", "couch-2"
        # - object name: "couch", "couch", "couch"

        # Submit retrieval tasks in parallel and aggregate results in the main thread
        futures_to_parent: Dict[Any, str] = {}
        tasks = []
        for parent_obj_id, parent_asset_id in parent_list:
            parent_obj_name = parent_obj_id.split("-")[:-1]
            parent_obj_name = "-".join(parent_obj_name)
            child_obj_plan = plan[parent_obj_name]["objects_on_top"]  # child plan corresponding to the current parent obj
            if child_obj_plan:
                tasks.append((parent_obj_id, parent_asset_id, child_obj_plan))

        if not tasks:
            return results

        max_workers = min(8, len(tasks))  # default=8  TODO: May raise APIConnectionError
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for parent_obj_id, parent_asset_id, child_obj_plan in tasks:
                fut = executor.submit(
                    self._retrieve_small_objects,
                    parent_obj_id,
                    parent_asset_id,
                    child_obj_plan,
                )
                futures_to_parent[fut] = parent_obj_id

            for fut in as_completed(futures_to_parent):
                parent_id = futures_to_parent[fut]
                results[parent_id] = fut.result()

        return results


    def _retrieve_small_objects(
        self, 
        parent_name: str, 
        parent_asset_id: str,
        child_objs: List[SmallObjectInfo],
    ) -> AssetList:
        results: AssetList = []
        assert parent_asset_id in self.db
        parent_dims = get_bbox_dims(self.db[parent_asset_id])
        parent_size = [float(parent_dims["x"]), float(parent_dims["z"])]
        # parent_area = parent_size[0] * parent_size[1]
        parent_size.sort()

        child_objs = sorted(child_objs, key=lambda x: -x["importance"]) # desc order by importance
        for child_obj in child_objs:
            child_name, quantity, variance_type, importance = (
                child_obj["object_name"],
                child_obj["quantity"],
                child_obj["variance_type"],
                child_obj["importance"],
            )
            quantity = min(quantity, 5)
            # print(f"Selecting {quantity} {child_name} w/ importance {importance} for placing on {parent_name}")
            
            candidates: Sequence[Tuple[str, float]] = (
                self.retriever.retrieve_with_name_and_desc(
                    object_names=[child_name],
                    object_descriptions=None,
                    threshold=self.clip_threshold,
                )
            )
            candidates = [
                candidate
                for candidate in candidates
                if get_annotations(self.db[candidate[0]])["onObject"]
            ] # filter out objects without "onObject"

            valid_candidates = []
            for cand in candidates:
                cand_dims = get_bbox_dims(self.db[cand[0]])
                cand_size = [float(cand_dims["x"]), float(cand_dims["z"])]
                cand_size.sort()
                if (
                    cand_size[0] < parent_size[0] * self.size_threshold and 
                    cand_size[1] < parent_size[1] * self.size_threshold
                ): 
                    valid_candidates.append(cand)
            
            valid_candidates = valid_candidates[:25]
            candidate_scores = _compare_object_name_to_descriptions(
                object_name=child_name,
                asset_ids=[cand[0] for cand in valid_candidates],
                database=self.db,
                llm=self.llm,
            )
            score_and_val_candidate_list = list(zip(candidate_scores, valid_candidates))
            valid_candidates = [
                cand
                for cand_scores, cand in score_and_val_candidate_list
                if cand_scores >= 5
            ]
            _report_candidates_status(valid_candidates, f"{child_name} on {parent_name}", "Small Selector")
            if len(valid_candidates) == 0:
                continue

            # remove used assets
            top_candidate = valid_candidates[0]
            valid_candidates = [
                cand
                for cand in valid_candidates
                if cand[0] not in self.used_assets
            ]
            if len(valid_candidates) == 0:
                valid_candidates = [top_candidate]

            valid_candidates = valid_candidates[:5]

            selected_asset_ids = []
            if variance_type == "same":
                selected_candidate = random_select(valid_candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = random_select(valid_candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(valid_candidates) > 1:
                        valid_candidates.remove(selected_candidate)
            else:
                raise ValueError(
                    f"Variance type {variance_type} is not supported."
                )
            
            for i in range(quantity):
                selected_asset_id = selected_asset_ids[i]
                child_id = f"{child_name}-{i}"
                results.append((child_id, selected_asset_id))

        return results







    

