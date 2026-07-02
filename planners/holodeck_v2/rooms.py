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
import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from colorama import Fore
from langchain_core.prompts import PromptTemplate
from shapely.geometry import LineString, Point, Polygon

from utils.holodeck_v2 import prompts
from utils.holodeck_v2.constants import DEBUGGING
from utils.holodeck_v2.llm import OpenAIWithTracking
from utils.holodeck_v2.types import RoomDict, validate_room_dict


class FloorPlanGeneratorV2:
    def __init__(self, llm: OpenAIWithTracking, used_assets=[]):
        self.json_template = {
            "ceilings": [],
            "children": [],
            "vertices": None,
            "floorPolygon": [],
            "id": None,
            "roomType": None,
        }
        self.floor_plan_template = PromptTemplate(
            input_variables=["input", "additional_requirements"],
            template=prompts.FLOOR_PLAN_PROMPT,
        )
        self.llm = llm
        self.used_assets = used_assets

    def generate_rooms(self, query, visualize=False) -> RoomDict:
        # force single-room
        additional_requirements = "I only need one room"
        floor_plan_prompt = self.floor_plan_template.format(
            input=query, additional_requirements=additional_requirements
        )

        raw_floor_plan = self.llm(floor_plan_prompt)

        print(f"User: {floor_plan_prompt}\n")
        print(f"{Fore.GREEN}AI: Here is the floor plan:\n{raw_floor_plan}{Fore.RESET}")

        parsed_rooms = self.get_plan(query, raw_floor_plan, visualize)
        if len(parsed_rooms) > 1:
            print(
                f"{Fore.YELLOW}Warning: AI generated >1 rooms. Selecting only the first room{Fore.RESET}"
            )
        room0 = parsed_rooms[0]
        validate_room_dict(room0)
        return room0

    def get_plan(self, query, raw_plan, visualize=False):
        parsed_plan = self.parse_raw_plan(raw_plan)

        if visualize:
            self.visualize_floor_plan(query, parsed_plan)

        return parsed_plan

    def parse_raw_plan(self, raw_plan):
        parsed_plan = []
        room_types = []
        plans = [plan.lower() for plan in raw_plan.split("\n") if "|" in plan]
        for i, plan in enumerate(plans):
            room_type, vertices = plan.split("|")
            room_type = room_type.strip().replace("'", "")  # remove single quote

            if room_type in room_types:
                room_type += f"-{i}"
            room_types.append(room_type)

            vertices = ast.literal_eval(vertices.strip())
            # change to float
            vertices = [(float(vertex[0]), float(vertex[1])) for vertex in vertices]

            current_plan = copy.deepcopy(self.json_template)
            current_plan["id"] = room_type
            current_plan["roomType"] = room_type
            current_plan["vertices"], current_plan["floorPolygon"] = self.vertices2xyz(
                vertices
            )
            parsed_plan.append(current_plan)

        # get full vertices: consider the intersection with other rooms
        all_vertices = []
        for room in parsed_plan:
            all_vertices += room["vertices"]
        all_vertices = list(set(map(tuple, all_vertices)))

        for room in parsed_plan:
            full_vertices = self.get_full_vertices(room["vertices"], all_vertices)
            full_vertices = list(set(map(tuple, full_vertices)))
            room["full_vertices"], room["floorPolygon"] = self.vertices2xyz(
                full_vertices
            )

        valid, msg = self.check_validity(parsed_plan)

        if not valid:
            print(f"{Fore.RED}AI: {msg}{Fore.RESET}")

            if DEBUGGING:
                import matplotlib.pyplot as plt
                import numpy as np

                colors = plt.cm.rainbow(np.linspace(0, 1, len(parsed_plan)))
                for room in parsed_plan:
                    for i in range(len(room["vertices"])):
                        a = room["vertices"][i]
                        b = room["vertices"][(i + 1) % len(room["vertices"])]
                        plt.plot([a[0], b[0]], [a[1], b[1]], color=colors[i])
                plt.show()

            raise ValueError(msg)
        else:
            print(f"{Fore.GREEN}AI: {msg}{Fore.RESET}")
            return parsed_plan

    def vertices2xyz(self, vertices):
        sort_vertices = self.sort_vertices(vertices)
        xyz_vertices = [
            {"x": vertex[0], "y": 0, "z": vertex[1]} for vertex in sort_vertices
        ]
        return sort_vertices, xyz_vertices

    def xyz2vertices(self, xyz_vertices):
        vertices = [(vertex["x"], vertex["z"]) for vertex in xyz_vertices]
        return vertices

    def sort_vertices(self, vertices):
        # Calculate the centroid of the polygon
        cx = sum(x for x, y in vertices) / max(len(vertices), 1)
        cy = sum(y for x, y in vertices) / max(len(vertices), 1)

        # Sort the vertices in clockwise order
        vertices_clockwise = sorted(
            vertices, key=lambda v: (-math.atan2(v[1] - cy, v[0] - cx)) % (2 * math.pi)
        )

        # Find the vertex with the smallest x value
        min_vertex = min(vertices_clockwise, key=lambda v: v[0])

        # Rotate the vertices so the vertex with the smallest x value is first
        min_index = vertices_clockwise.index(min_vertex)
        vertices_clockwise = (
            vertices_clockwise[min_index:] + vertices_clockwise[:min_index]
        )

        return vertices_clockwise

    def get_full_vertices(self, original_vertices, all_vertices):
        # Create line segments from the original vertices
        lines = [
            LineString(
                [
                    original_vertices[i],
                    original_vertices[(i + 1) % len(original_vertices)],
                ]
            )
            for i in range(len(original_vertices))
        ]

        # Check each vertex against each line segment
        full_vertices = []
        for vertex in all_vertices:
            point = Point(vertex)
            for line in lines:
                if line.intersects(point):
                    full_vertices.append(vertex)

        return full_vertices

    def parsed2raw(self, rooms):
        raw_plan = ""
        for room in rooms:
            raw_plan += " | ".join([room["roomType"], str(room["vertices"])])
            raw_plan += "\n"
        return raw_plan

    def check_interior_angles(self, vertices):
        n = len(vertices)
        for i in range(n):
            a, b, c = vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n]
            angle = abs(
                math.degrees(
                    math.atan2(c[1] - b[1], c[0] - b[0])
                    - math.atan2(a[1] - b[1], a[0] - b[0])
                )
            )
            if angle < 90 or angle > 270:
                return False
        return True

    def check_validity(self, rooms):
        room_polygons = [Polygon(room["vertices"]) for room in rooms]

        # check interior angles
        for room in rooms:
            if not self.check_interior_angles(room["vertices"]):
                return (
                    False,
                    "All interior angles of the room must be greater than or equal to 90 degrees.",
                )

        if len(room_polygons) == 1:
            return True, "The floor plan is valid. (Only one room)"

        # check overlap, connectivity and vertex inside another room
        for i in range(len(room_polygons)):
            has_neighbor = False
            for j in range(len(room_polygons)):
                if i != j:
                    if (
                        room_polygons[i].equals(room_polygons[j])
                        or room_polygons[i].contains(room_polygons[j])
                        or room_polygons[j].contains(room_polygons[i])
                    ):
                        return False, "Room polygons must not overlap."
                    intersection = room_polygons[i].intersection(room_polygons[j])
                    if isinstance(intersection, LineString):
                        has_neighbor = True
                    for vertex in rooms[j]["vertices"]:
                        if Polygon(rooms[i]["vertices"]).contains(Point(vertex)):
                            return (
                                False,
                                "No vertex of a room can be inside another room.",
                            )
            if not has_neighbor:
                return (
                    False,
                    "Each room polygon must share an edge with at least one other room polygon.",
                )

        return True, "The floor plan is valid."

    def visualize_floor_plan(self, query, parsed_plan):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 22
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = [
            (0.53, 0.81, 0.98, 0.5),
            (0.56, 0.93, 0.56, 0.5),
            (0.94, 0.5, 0.5, 0.5),
            (1.0, 1.0, 0.88, 0.5),
        ]

        def midpoint(p1, p2):
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        for i, room in enumerate(parsed_plan):
            coordinates = room["vertices"]
            polygon = patches.Polygon(
                coordinates, closed=True, edgecolor="black", linewidth=2
            )
            polygon.set_facecolor(colors[i % len(colors)])
            ax.add_patch(polygon)

        for i, room in enumerate(parsed_plan):
            coordinates = room["vertices"]
            # Label the rooms
            x, y = zip(*coordinates)
            room_x = sum(x) / len(coordinates)
            room_y = sum(y) / len(coordinates)
            # ax.text(room_x, room_y, room["roomType"], ha='center', va='center')

            # Add points to the corners
            ax.scatter(x, y, s=100, color="black")  # s is the size of the point

            # # Display width and length
            # for i in range(len(coordinates)):
            #     p1, p2 = coordinates[i], coordinates[(i + 1) % len(coordinates)]
            #     label = f"{np.round(np.linalg.norm(np.array(p1) - np.array(p2)), 2)} m"
            #     ax.text(*midpoint(p1, p2), label, ha='center', va='center', fontsize=20, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round4'))

        # Set aspect of the plot to be equal, so squares appear as squares
        ax.set_aspect("equal")
        ax.autoscale_view()

        # Turn off the axis
        ax.axis("off")

        folder_name = query.replace(" ", "_")
        plt.savefig(f"{folder_name}.pdf", bbox_inches="tight", dpi=300)
        plt.show()
