"""
LLM frontend: requirement -> constraint JSON.

This is the first stage of the pipeline (generate -> compile -> solve). It owns
everything between a natural-language requirement and a parsed constraint dict:
build the asset list, render the generation prompt, drive the LLM, parse the
returned DSL (with LLM fix retries), and persist the artifacts to disk.

It is a thin frontend over `dsl` (parse_program/code_to_json/parse_cogmap), the
`prompts` singleton, and an `llm.ChatSession`. It holds no scene/solver state and
emits a plain `(constr_json, code, cogmap_json)` tuple for `compile` to consume.
The semantic-vs-spatial prompt set is read at call time so the prompt cfg-stamp
matches the run that actually invokes `generate`.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from utils.console import ReasoningWindow, reasoning_unavailable
from utils.log import print_info, print_warn, print_error
from utils.r3l.llm import ChatSession, extract_code_block
from utils.r3l.types import AssetInfo, get_uid
from utils.r3l.plot import visualize_scene_graph, visualize_cogmap
from solvers.r3l.config import cfg
from solvers.r3l.prompts.prompts import prompts
from solvers.r3l.dsl.parse_constraints import parse_program, code_to_json
from solvers.r3l.dsl.parse_cogmap import parse_cogmap, CogMapParseError


def _prompt_set() -> Dict:
    """Route to the semantic or spatial prompt set, read at call time so the
    prompt cfg-stamp tracks the active run: semantic uses constraints_semantic;
    spatial and flat (decomposition==none) use constraints_spatial."""
    return (
        prompts.constraints_semantic
        if cfg.modules.decomposition == "semantic"
        else prompts.constraints_spatial
    )


def _new_session(sys_prompt: str) -> ChatSession:
    """Create the conversation session used for generation and fix retries."""
    return ChatSession(
        model=cfg.llm.heavy,
        mode="conversation",
        sys_prompt=sys_prompt,
        verbose=cfg.runtime.verbose,
    )


def _build_asset_list_str(
    asset_ids: List[str],
    asset_info: Dict[str, AssetInfo],
    asset_to_object: Dict[str, str],
) -> Tuple[str, List[str]]:
    """Build asset list string and object ID list for prompts."""
    asset_list: List[str] = []
    object_id_list: List[str] = []
    for aid in asset_ids:
        ainfo = asset_info[get_uid(aid)]
        object_id = asset_to_object[aid]
        object_id_list.append(object_id)

        match cfg.prompt.desc_type:
            case "short": desc = ainfo.desc_short
            case "long": desc = ainfo.desc_long
            case "none": desc = ainfo.name
            case _: raise ValueError(f"Unrecognized: {cfg.prompt.desc_type}")

        bbox = ainfo.bbox
        item = f"{object_id.replace('-', '_')} = Asset(id=\"{object_id}\", description=\"{desc}\", size=({bbox['x']:.1f}, {bbox['y']:.1f}))"
        asset_list.append(item)
    return "\n".join(asset_list), object_id_list


def _build_main_prompt(
    pc: Dict,
    requirements: str,
    room_size: Tuple[float, float],
    asset_list_str: str,
) -> str:
    """Build main generation prompt."""
    example_str = pc['output_example'].substitute() if cfg.prompt.include_example else ""
    return pc['main'].substitute(
        output_example=example_str,
        requirement=requirements,
        length=f"{room_size[0]:.1f}",
        width=f"{room_size[1]:.1f}",
        asset_list_str=asset_list_str,
    )


def _parse_dsl(code: str, var_to_obj_id: Dict[str, str]) -> Tuple[dict, Optional[dict]]:
    """
    Parse DSL code into constraint JSON (and optional cogmap JSON).

    The single DSL->JSON parse sequence shared by `_parse_with_retry` (LLM path)
    and the factory reparse branch (load-from-disk path).

    Returns:
        (constraints_json, cogmap_json or None)
    """
    program = parse_program(code, var_to_obj_id, hv_absolute=cfg.prompt.hv_absolute)
    cogmap_json = parse_cogmap(code, program) if cfg.modules.imagination_pose else None
    return code_to_json(program), cogmap_json


def _parse_with_retry(
    llm: ChatSession,
    pc: Dict,
    code_block: str,
    object_id_list: List[str],
    max_retries: int = 3,
) -> Tuple[dict, str, Optional[dict]]:
    """
    Parse DSL code with LLM fix retries.

    Returns:
        (constraints_json, final_code, cogmap_json or None)

    Raises:
        RuntimeError: If parse fails after max_retries
    """
    var_to_obj_id = {oid.replace("-", "_"): oid for oid in object_id_list}
    fix_template = pc['fix_error']

    for i in range(max_retries):
        a = "a" if i == 0 else "another"
        try:
            constr_json, cogmap_json = _parse_dsl(code_block, var_to_obj_id)
            return constr_json, code_block, cogmap_json
        except (Exception, CogMapParseError) as e:
            print_warn(f"Caught DSL error: {e}. Prompting for fix.")
            fix_prompt = fix_template.substitute(a=a, e=str(e))
            code_block = extract_code_block(llm.ask(fix_prompt))

    raise RuntimeError("[constr] Failed to parse DSL after retries")


def generate(
    requirements: str,
    room_size: Tuple[float, float],
    asset_info: Dict[str, AssetInfo],
    asset_ids: List[str],
    asset_to_object: Dict[str, str],
) -> Tuple[dict, str, Optional[dict]]:
    """
    Generate initial constraints (and optionally cogmap) for the group using LLM.
    This is called once at the beginning of the pipeline.

    Returns:
        (constraints_json, dsl_code, cogmap_json or None)

    Does NOT save artifacts - caller must call save().
    """
    pc = _prompt_set()
    llm = _new_session(pc['sys'].substitute())
    asset_list_str, object_id_list = _build_asset_list_str(asset_ids, asset_info, asset_to_object)
    prompt = _build_main_prompt(pc, requirements, room_size, asset_list_str)

    # Verbose mode dumps the raw prompt/response; the in-place reasoning window would
    # fight that output, so the two displays are mutually exclusive.
    window = ReasoningWindow()
    on_reasoning = None if cfg.runtime.verbose else window.feed
    t0 = time.time()
    try:
        raw = llm.ask(prompt, on_reasoning=on_reasoning)
    finally:
        window.close()  # collapse the window on any outcome, before printing below it
    if on_reasoning is not None and not window.saw_any:
        reasoning_unavailable()
    print_info(f"[constr] LLM generation took {(time.time() - t0):.0f}s")

    try:
        code_block = extract_code_block(raw)
    except ValueError:
        print_error("LLM output format error. Retrying...")
        code_block = extract_code_block(llm.retry())

    return _parse_with_retry(llm, pc, code_block, object_id_list)


def _save_program(path: str, code: str) -> None:
    """Save program code to file."""
    with open(path, "w") as f:
        f.write(code)


def _save_json(path: str, data: dict, indent: int = 4) -> None:
    """Save JSON data to file with specified indentation."""
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def save(
    save_dir: str,
    code: str,
    constr_json: dict,
    cogmap_json: Optional[dict],
    asset_ids: List[str],
    asset_info: Dict[str, AssetInfo],
    asset_to_object: Dict[str, str],
) -> None:
    """
    Save all generation artifacts.

    Saves:
    - llm_output.py
    - constraints.json
    - cogmap.json (if cogmap enabled)
    - cogmap.png (if cogmap enabled)
    - scene_graph.mmd
    - scene_graph.png
    """
    os.makedirs(save_dir, exist_ok=True)

    _save_program(os.path.join(save_dir, "llm_output.py"), code)
    _save_json(os.path.join(save_dir, "constraints.json"), constr_json)

    if cogmap_json:
        _save_json(os.path.join(save_dir, "cogmap.json"), cogmap_json)
        visualize_cogmap(
            cogmap_json,
            save_dir,
            assets=asset_ids,
            asset_info=asset_info,
            asset_to_object=asset_to_object,
            constraints_json=constr_json,
            out_name="cogmap.png",
        )

    visualize_scene_graph(constr_json, save_dir, format="png", base_name="scene_graph")


def generate_spec(
    save_dir: str,
    requirements: str,
    room_size: Tuple[float, float],
    asset_info: Dict[str, AssetInfo],
    asset_ids: List[str],
    asset_to_object: Dict[str, str],
) -> dict:
    exist = os.path.exists(os.path.join(save_dir, "llm_output.py"))
    print_info(f"[constr] start: {'reparse' if exist else 'gen+parse'}")

    if exist:
        with open(os.path.join(save_dir, "llm_output.py")) as f:
            code = f.read()
        var_to_obj_id = {oid.replace("-", "_"): oid for oid in asset_to_object.values()}
        constr_json, cogmap_json = _parse_dsl(code, var_to_obj_id)
    else:
        constr_json, code, cogmap_json = generate(
            requirements=requirements,
            room_size=room_size,
            asset_info=asset_info,
            asset_ids=asset_ids,
            asset_to_object=asset_to_object,
        )

    save(save_dir, code, constr_json, cogmap_json, asset_ids, asset_info, asset_to_object)
    return constr_json
