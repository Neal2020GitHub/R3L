import math
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from typing import Callable

from solvers.r3l.config import LRAnneal, PhysicsSchedule


def make_phy_scheduler(physics_cfg: PhysicsSchedule, iters: int) -> Callable[[int], float]:
    if physics_cfg.shape == "none":
        max_val = physics_cfg.max
        return lambda i: max_val
    delay = physics_cfg.delay_steps
    warmup = max(iters - delay, 0)
    match physics_cfg.shape:
        case "linear":    ramp = LinearScheduler(warmup, physics_cfg.min, physics_cfg.max)
        case "cosine":    ramp = CosineScheduler(warmup, physics_cfg.min, physics_cfg.max)
        case "quadratic": ramp = QuadraticScheduler(warmup, physics_cfg.min, physics_cfg.max)
        case _: raise ValueError(f"Invalid phy schedule shape: {physics_cfg.shape}")
    floor = physics_cfg.min
    return lambda i: floor if i < delay else ramp(i - delay)


def make_lr_scheduler(opt: optim.Optimizer, max_lrs: list[float], lr_cfg: LRAnneal, iters: int):
    match lr_cfg.name:
        case "onecycle": return lrs.OneCycleLR(opt, max_lr=max_lrs, total_steps=iters, pct_start=lr_cfg.pct_start)
        case "cosine":   return lrs.CosineAnnealingLR(opt, T_max=iters, eta_min=lr_cfg.eta_min)
        case "step":     return lrs.StepLR(opt, step_size=lr_cfg.step_size, gamma=lr_cfg.gamma)
        case "constant": return lrs.ConstantLR(opt, factor=1.0, total_iters=0)
        case _:
            raise ValueError(f"Unknown LR scheduler: {name}")
        

class LinearScheduler:
    """Linear ramp from min to max over warmup_steps."""

    def __init__(self, warmup_steps: int, min_val: float, max_val: float):
        self.warmup_steps = warmup_steps
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, iter: int) -> float:
        t = min(iter / self.warmup_steps, 1.0) if self.warmup_steps > 0 else 1.0
        return self.min_val + (self.max_val - self.min_val) * t


class CosineScheduler:
    """
    Cosine annealing warmup: S-curve with smooth start and end.

    Formula: min + (max - min) * 0.5 * (1 - cos(pi * t))

    - Starts slow (gentle gradient at t=0)
    - Accelerates through middle
    - Decelerates at end (gentle gradient at t=1)
    """

    def __init__(self, warmup_steps: int, min_val: float, max_val: float):
        self.warmup_steps = warmup_steps
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, iter: int) -> float:
        t = min(iter / self.warmup_steps, 1.0) if self.warmup_steps > 0 else 1.0
        return self.min_val + (self.max_val - self.min_val) * 0.5 * (1 - math.cos(math.pi * t))


class QuadraticScheduler:
    """
    Quadratic warmup: slow start, fast finish.

    Formula: min + (max - min) * t^2

    Useful when you want semantic constraints to dominate early,
    then physics constraints to kick in hard at the end.
    """

    def __init__(self, warmup_steps: int, min_val: float, max_val: float):
        self.warmup_steps = warmup_steps
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, iter: int) -> float:
        t = min(iter / self.warmup_steps, 1.0) if self.warmup_steps > 0 else 1.0
        return self.min_val + (self.max_val - self.min_val) * (t * t)