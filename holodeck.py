# R3L entry point that aggregates the vendored Holodeck / Holodeck 2.0
# subsystem (planners/holodeck_v2, retrievers/holodeck_v2, utils/holodeck_v2).
# That subsystem is derived from AllenAI Holodeck (Apache License 2.0); see
# THIRDPARTY_LICENSES.md and the LICENSE file in each holodeck_v2 directory.
# This file is R3L's own glue code, released under the project's MIT License
# (see the repository root LICENSE).

from planners.holodeck_v2.factory import *
from retrievers.holodeck_v2.factory import create_holodeck_retriever, create_slim_retriever
from utils.holodeck_v2.llm import OpenAIWithTracking
from utils.holodeck_v2.constants import LLM_MODEL_NAME
from utils.holodeck_v2.types import *
from adapters.protocols import HolodeckBlueprint

from colorama import Fore


_clip_models = None
_sbert_model = None
_retriever = None
_llm = None


def init_holodeck(openai_key: str, slim: bool = False):
    """
    Initialize Holodeck components.

    Args:
        openai_key: OpenAI API key for LLM calls
        slim: If True, skip loading CLIP/SBERT models (use when assets are pre-specified)
    """
    global _clip_models, _sbert_model, _retriever, _llm

    if _clip_models or _sbert_model or _retriever or _llm:
        print(f"{Fore.RED}[WARNING] Holodeck already initialized. Overwriting.{Fore.RESET}")

    if slim:
        # Lightweight mode: only load database, skip CLIP/SBERT (~2GB+ memory saved)
        _retriever = create_slim_retriever()
    else:
        # Full mode: load everything for asset retrieval
        _clip_models = create_clip_models()
        _sbert_model = create_sbert_model()
        _retriever = create_holodeck_retriever(_clip_models, _sbert_model)

    _llm = OpenAIWithTracking(
        model=LLM_MODEL_NAME,
        max_tokens=16000,
        openai_api_key=openai_key,
        verbose=False,
    )


def create_floor_plan(query: str) -> tuple[RoomDict, WallsDict]:
    """Planning phase 1: the LLM lays out the room rectangle and picks a wall height.
    Split from object selection so the CLI can present the two as distinct stages."""
    room: RoomDict = create_holodeck_rooms(
        llm=_llm,
        query=query,
        used_assets=[],
        visualize=False,
    )
    walls: WallsDict = create_holodeck_walls(query, _llm, room)
    return room, walls


def create_object_selection(
    query: str,
    room: RoomDict,
    walls: WallsDict,
    include_floor: bool = True,
    include_wall: bool = False,
    include_small: bool = False,
) -> HolodeckBlueprint:
    """Planning phase 2: the LLM selects furniture and CLIP/SBERT retrieves assets,
    assembled with the phase-1 room/walls into the HolodeckBlueprint."""
    selector = create_holodeck_selector(_retriever, _llm)
    floor_objs, wall_objs, design, plan = selector.select_objects(
        query=query,
        room=room,
        walls=walls,
        get_floor_objects=include_floor,
        get_wall_objects=include_wall,
    )

    if include_small:
        s_selector = create_holodeck_small_selector(_retriever, _llm)
        small_objs = s_selector.select_small_objects(plan, floor_objs, wall_objs)
    else:
        small_objs = {}

    return HolodeckBlueprint(
        query=query,
        objects={
            'floor': floor_objs,
            'wall': wall_objs,
            'small': small_objs
        },
        walls=walls,
        room=room,
        design=design,
        plan=plan,
    )
