import os
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")  # immutable + loud on unknown/mistyped key


# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------
class LLM(_Base):
    heavy: str
    max_tokens: int
    temperature: float
    reasoning_effort: Literal["minimal", "low", "medium", "high"]


# -----------------------------------------------------------------------------
# MODULES
# -----------------------------------------------------------------------------
class Modules(_Base):
    decomposition: Literal["spatial", "semantic", "none"]
    imagination_pose: bool
    imagination_footprint: bool
    optimization_reparam: bool


# -----------------------------------------------------------------------------
# PROMPT
# -----------------------------------------------------------------------------
class Prompt(_Base):
    desc_type: Literal["short", "long", "none"]
    include_example: bool
    hv_absolute: bool


# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
class LR(_Base):
    position: float
    rotation: float
    param: Optional[float] = None


class GradClipNorm(_Base):
    position: Optional[float] = None
    rotation: Optional[float] = None
    param: Optional[float] = None


class LRAnneal(_Base):
    name: Literal["cosine", "onecycle", "step", "constant"]
    eta_min: float
    pct_start: float
    step_size: int
    gamma: float


class PhysicsSchedule(_Base):
    shape: Literal["none", "linear", "cosine", "quadratic"]
    delay_steps: int
    min: float
    max: float


class BaseStage(_Base):
    iterations: int
    optimizer: Literal["adam", "adamw", "sgd", "momentum"]
    lr: LR
    grad_clip_norm: GradClipNorm
    lr_anneal: LRAnneal
    physics_schedule: PhysicsSchedule


class Solver(_Base):
    base: BaseStage
    finetune: BaseStage


# -----------------------------------------------------------------------------
# CONSTRAINTS
# -----------------------------------------------------------------------------
class Weights(_Base):
    collision: float
    wall: float
    gap: float
    distance: float
    directional: float
    against_wall: float
    corner: float
    facing_radial: float
    facing_ortho: float
    facing_wall: float
    around: float
    angle: float
    aesthetics: float
    horizontal: float
    vertical: float


class WallShape(_Base):
    act: Literal["relu", "relu2", "softplus", "softplus2"]


class CollisionShape(_Base):
    method: Literal["iou", "diou", "giou"]


class GapShape(_Base):
    method: Literal["sdf", "mtv", "l2"]


class DirectionalShape(_Base):
    act: Literal["relu", "relu2", "softplus", "softplus2"]
    w_dir: float
    w_align: float


class CornerShape(_Base):
    w_pos: float
    w_rot: float
    loss_fn: Literal["l2", "l1", "huber"]
    huber_delta: float


class AgainstWallShape(_Base):
    w_pos: float
    w_rot: float
    loss_fn: Literal["l2", "l1", "huber"]
    huber_delta: float


class AroundShape(_Base):
    stable: bool
    eps: float
    w_spacing: float
    w_center: float


class HorizontalShape(_Base):
    loss_fn: Literal["l2", "l1", "huber"]
    huber_delta: float


class VerticalShape(_Base):
    loss_fn: Literal["l2", "l1", "huber"]
    huber_delta: float


class AestheticsShape(_Base):
    enabled: bool
    angles_deg: list[float]
    eps_deg: float
    soft_landing: Literal["tukey", "cosine", "smoothstep"]


class Shapes(_Base):
    wall: WallShape
    collision: CollisionShape
    gap: GapShape
    directional: DirectionalShape
    corner: CornerShape
    against_wall: AgainstWallShape
    around: AroundShape
    horizontal: HorizontalShape
    vertical: VerticalShape
    aesthetics: AestheticsShape


class LearnablePrior(_Base):
    loss: Literal["mse", "l1", "smooth_l1", "circular_deg"]
    weight: float


class LearnablePriors(_Base):
    scale_prior: float
    unit: LearnablePrior
    nonneg: LearnablePrior
    angle_deg: LearnablePrior


class Constraints(_Base):
    weights: Weights
    shapes: Shapes
    learnable_priors: LearnablePriors


# -----------------------------------------------------------------------------
# RENDER  —  all render/output config (static image + process animation)
# -----------------------------------------------------------------------------
class View2D(_Base):
    """2D matplotlib top-down process animation (fixed top camera)."""
    enabled: bool = True
    palette: list[str]
    fill_alpha: float
    collision_alpha: float
    existing_color: str


class View3D(_Base):
    """3D Blender-Cycles process animation. Single camera (top or side, not
    multi). Off by default — per-frame Cycles is expensive."""
    enabled: bool = False
    stages: list[Literal["base", "finetune"]] = Field(
        default_factory=lambda: ["base", "finetune"])
    camera: Literal["top", "side"] = "side"
    format: Literal["gif", "mp4"] = "mp4"


class Animation(_Base):
    """Optimization replay: 2D top-down + 3D perspective, shared fps."""
    fps: float = Field(24.0, gt=0)
    view_2d: View2D = Field(default_factory=View2D)
    view_3d: View3D = Field(default_factory=View3D)


class Image(_Base):
    """Static final layout stills: image_top.png + image_side.png + scene.blend."""
    enabled: bool = True


class Render(_Base):
    """All render output: static image + process animation."""
    image: Image = Field(default_factory=Image)
    animation: Animation = Field(default_factory=Animation)


# -----------------------------------------------------------------------------
# RUNTIME  —  system & I/O (render config lives under `render`, not here)
# -----------------------------------------------------------------------------
class Runtime(_Base):
    device: Literal["cuda", "cpu", "mps"]
    seed: Optional[int] = None
    frame_every: int = Field(gt=0)  # optimization-loop pose sampling cadence; feeds animations
    loss_every: int = Field(gt=0)  # loss-dashboard refresh cadence; modulo divisor
    loss_rows: Optional[int] = None
    verbose: bool


# -----------------------------------------------------------------------------
# ROOT
# -----------------------------------------------------------------------------
class R3LConfig(_Base):
    llm: LLM
    modules: Modules
    prompt: Prompt
    solver: Solver
    constraints: Constraints
    runtime: Runtime
    render: Render


def _load() -> "R3LConfig":
    path = os.environ.get("R3L_CONFIG") or os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path) as f:
        return R3LConfig.model_validate(yaml.safe_load(f) or {})


cfg = _load()
