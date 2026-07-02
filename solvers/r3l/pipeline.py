import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .constraints import CompiledConstraints

from .optimize import Optimizer
from .config import cfg, BaseStage
from .layout import init_layout
from . import render as render_mod
from .generate import generate_spec
from .compile import compile as compile_spec
from .assets import create_asset_mapping, create_asset_info, flatten_assets

from utils.console import inter_stage_rule
from utils.r3l.types import Pose, PoseVec, BBoxVec, get_uid
from adapters.protocols import R3LBlueprint, R3LSolution


class Stage(Enum):
    GENERATE = auto()
    OPTIMIZE = auto()
    RENDER = auto()


@dataclass
class OptResult:
    tag: str # stage of the optimization ("base" or "finetune")
    pose_vec: PoseVec # final pose vector for the stage
    pose_map: Dict[str, Pose] # mapping from asset_id to Pose for the stage
    frames: List[PoseVec] # intermediate frames for animation


class Pipeline:
    def __init__(self, blueprint: R3LBlueprint, save_dir: str, asset_dir: str):
        self.save_dir, self.asset_dir = save_dir, asset_dir
        self.room_size = blueprint.room_size
        self.requirements = _requirements(blueprint)
        self.asset_ids = flatten_assets(blueprint.assets)
        self.asset_info = create_asset_info(asset_ids=self.asset_ids, asset_dir=self.asset_dir)
        self.obj_to_asset, self.asset_to_obj = create_asset_mapping(
            asset_info=self.asset_info, asset_ids=self.asset_ids)
        self.bbox_vec = BBoxVec.build(self.asset_info, self.asset_ids, cfg.runtime.device)
        self.optimizer = Optimizer(device=cfg.runtime.device)

    def run(self, on_stage: Callable[[Stage], None] = lambda _stage: None) -> R3LSolution:
        """Run the pipeline end-to-end and return the layout solution.

            requirement
                │ generate
                ▼
            constraint JSON
                │ compile
                ▼
            CompiledConstraints
                │ optimize
                ▼
            poses
                │ assemble
                ▼
            R3LSolution

        render emits each stage's animation and the final layout along the way.
        """

        on_stage(Stage.GENERATE)
        constr_json = self.generate()
        constraints = self.compile(constr_json)

        on_stage(Stage.OPTIMIZE)
        runs        = self.optimize(constraints)
        solution    = self.assemble(runs[-1])

        on_stage(Stage.RENDER)
        self.render(runs, constraints)
        return solution

    def generate(self) -> dict:
        """Generate the constraint JSON via LLM, or reparse a cached llm_output.py."""
        return generate_spec(
            save_dir=self.save_dir, requirements=self.requirements, room_size=self.room_size,
            asset_info=self.asset_info, asset_ids=self.asset_ids, asset_to_object=self.asset_to_obj)

    def compile(self, constr_json: dict) -> "CompiledConstraints":
        """Compile the constraint JSON into the differentiable objective."""
        object_to_index = {oid: i for i, oid in enumerate(self.obj_to_asset.keys())}
        return compile_spec(constr_json, object_to_index, self.bbox_vec, self.room_size, self.save_dir)

    def optimize(self, constraints: "CompiledConstraints") -> List[OptResult]:
        """Run the two-stage optimization strategy and return their results (see Paper's Section E.1)"""
        def run_stage(poses: PoseVec, stage: BaseStage, *, train_var: bool, tag: str) -> OptResult:
            vec, frames = self.optimizer.optimize(
                constraints=constraints, poses_vec=poses,
                stage=stage, train_var=train_var, tag=tag,
            )
            return OptResult(tag=tag, pose_vec=vec, pose_map=vec.to_pose_dict(self.asset_ids), frames=frames)

        init = PoseVec.from_pose_dict(init_layout(self.asset_ids, self.room_size), cfg.runtime.device)
        base     = run_stage(init,          cfg.solver.base,     train_var=False, tag="base"); inter_stage_rule()
        finetune = run_stage(base.pose_vec, cfg.solver.finetune, train_var=True,  tag="finetune")
        return [base, finetune]

    def render(self, runs: List[OptResult], constraints: "CompiledConstraints") -> None:
        """Animate every stage; render the final layout once."""
        render_mod.render(
            runs=runs, constraints=constraints, save_dir=self.save_dir,
            asset_ids=self.asset_ids, asset_info=self.asset_info,
            room_size=self.room_size, asset_to_object=self.asset_to_obj,
        )

    def assemble(self, run: OptResult) -> R3LSolution:
        """Build the R3LSolution from the main stage's poses."""
        data: Dict = {"layout": {}, "room_size": self.room_size}
        for aid, pose in run.pose_map.items():
            z = self.asset_info[get_uid(aid)].bbox["z"] * 0.5
            data["layout"][self.asset_to_obj[aid]] = {
                "asset_id": get_uid(aid),
                "position": (pose.x, pose.y, z),
                "rotation": (0, 0, pose.rz / math.pi * 180),
            }
        return R3LSolution.build(data)


def _requirements(blueprint: R3LBlueprint) -> str:
    return (
        f"- User Requirement: {blueprint.prompt}\n- Detailed Design Intent: {blueprint.design}"
        if blueprint.design else blueprint.prompt
    )
