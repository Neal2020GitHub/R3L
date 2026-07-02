import numpy as np
from typing import Dict, List, Tuple
from utils.r3l.types import Pose
from .config import cfg

def init_layout(asset_ids: List[str], room_size: Tuple[float, float]) -> Dict[str, Pose]:
    if cfg.runtime.seed is not None:
        np.random.seed(cfg.runtime.seed)
    n, (W, H) = len(asset_ids), room_size
    x = 0.25 * W + np.random.rand(n) * (0.5 * W)      # centered half-region
    y = 0.25 * H + np.random.rand(n) * (0.5 * H)
    rz = np.deg2rad(np.random.choice([0, 90, 180, 270], size=n))
    return {aid: Pose(x=x[i], y=y[i], rz=rz[i]) for i, aid in enumerate(asset_ids)}
