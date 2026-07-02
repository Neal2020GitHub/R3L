from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import torch
import torch.optim as optim

from utils.r3l.types import PoseVec, ParamVec
from utils.log import print_info, print_dict
from solvers.r3l.report import LossDashboard, LossCurveRecorder, emit_constraint_param_report
from solvers.r3l.constraints import CompiledConstraints
from solvers.r3l.schedulers import make_lr_scheduler, make_phy_scheduler
from solvers.r3l.config import cfg, BaseStage, GradClipNorm, LR, LRAnneal


# Optimizer roles, in the order their param-groups are registered. The order is
# load-bearing: it drives the per-group `max_lr` list handed to the scheduler.
ROLES = ("position", "rotation", "param")


def _as_param(t: torch.Tensor, device: str) -> torch.nn.Parameter:
    return torch.nn.Parameter(t.detach().clone().to(device))  # requires_grad=True


@dataclass
class Trainables:
    """The optimizer's entire trainable state.

    `x`, `y`, `rz` (object poses) are always trained. `param` holds the optimizable
    constraint (Var) parameters; it exists only when the constraints define any, and
    is handed to the optimizer only when `train_var` is set. `groups()` is the single
    place that maps a tensor to its optimizer role.
    """

    x: torch.nn.Parameter
    y: torch.nn.Parameter
    rz: torch.nn.Parameter
    param: Optional[torch.nn.Parameter]

    def pose(self) -> PoseVec:
        return PoseVec(x=self.x, y=self.y, rz=self.rz)

    def param_vec(self) -> Optional[ParamVec]:
        return ParamVec(values=self.param) if self.param is not None else None

    def groups(self, train_var: bool) -> Dict[str, List[torch.nn.Parameter]]:
        groups: Dict[str, List[torch.nn.Parameter]] = {
            "position": [self.x, self.y],
            "rotation": [self.rz],
        }
        if self.param is not None and train_var:
            groups["param"] = [self.param]
        return groups


class Optimizer:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.frame_every = cfg.runtime.frame_every
        self.loss_every = cfg.runtime.loss_every

    def _make_opt(self, grouped: Dict[str, List[torch.nn.Parameter]], lr: LR,
                  opt_name: str, lr_cfg: LRAnneal, iters: int):
        param_groups: List[Dict] = []
        max_lrs: List[float] = []
        for r in ROLES:
            ps = grouped.get(r)
            if not ps:
                continue
            rate = getattr(lr, r)
            param_groups.append({"params": ps, "lr": rate})
            max_lrs.append(rate)

        match opt_name:
            case "adam": opt = optim.Adam(param_groups)
            case "adamw": opt = optim.AdamW(param_groups)
            case "sgd": opt = optim.SGD(param_groups)
            case "momentum": opt = optim.SGD(param_groups, momentum=0.9)
            case _: raise ValueError(opt_name)
        lr_scheduler = make_lr_scheduler(opt, max_lrs, lr_cfg, iters)
        return opt, lr_scheduler

    def _clip_gradients(self, grouped: Dict[str, List[torch.nn.Parameter]], clip_cfg: GradClipNorm) -> None:
        """Apply per-group gradient norm clipping based on config."""
        for role, params in grouped.items():
            max_norm = getattr(clip_cfg, role, None)
            if max_norm is None:
                continue
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                continue
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    def optimize(
        self,
        constraints: CompiledConstraints,
        poses_vec: PoseVec,
        stage: BaseStage,
        train_var: bool = True,
        tag: Optional[str] = None,
    ) -> Tuple[PoseVec, List[PoseVec]]:
        """
        Run one optimization stage. 

        Returns the final poses and the animation frames.

        Input and output poses are all in the global frame.
        Inside, the loop may use the local frame, but that
        is hidden to the caller. Callers are agnostic to the 
        internal re-parameterization mechanisms.
        """
        iters = stage.iterations
        lr = stage.lr
        phy_scheduler = make_phy_scheduler(stage.physics_schedule, iters)

        # `reparam` selects the optimization frame: when set, poses are optimized in
        # the hierarchical local/global frame (localize on the way in, globalize out);
        # when unset, in the raw global frame and localize/globalize are no-ops.
        reparam = cfg.modules.optimization_reparam

        # Localize poses (cluster members -> local coordinates), optimize in the local
        # frame, and globalize on the way out (see `snapshot`).
        local = constraints.scene.localize(poses_vec, reparam=reparam)

        # `param` (Var parameters) is built only when the constraints define any, so
        # that `evaluate` can read their current value; it is trained only when
        # `train_var` is set (see `Trainables.groups`).
        st = Trainables(
            x=_as_param(local.x, self.device),
            y=_as_param(local.y, self.device),
            rz=_as_param(local.rz, self.device),
            param=(
                _as_param(torch.tensor(constraints.params.priors, dtype=torch.float32), self.device)
                if constraints.params else None
            ),
        )

        grouped = st.groups(train_var)
        opt, lr_scheduler = self._make_opt(grouped, lr, stage.optimizer, stage.lr_anneal, iters)

        def snapshot(p: PoseVec) -> PoseVec:
            g = constraints.scene.globalize(p, reparam=reparam)
            return PoseVec(
                x=g.x.detach().clone(), 
                y=g.y.detach().clone(), 
                rz=g.rz.detach().clone()
            )

        frames: List[PoseVec] = [snapshot(st.pose())]
        dashboard = LossDashboard(
            stage_name=tag or "optimize", 
            total_iters=iters,
            max_rows=cfg.runtime.loss_rows
        )
        recorder = LossCurveRecorder(save_dir=constraints.save_dir)
        loss: torch.Tensor | None = None
        nominal: Dict[str, torch.Tensor] | None = None

        for i in range(iters):
            opt.zero_grad(set_to_none=True)
            loss, nominal = constraints.evaluate(
                poses=st.pose(), 
                alpha=phy_scheduler(i), 
                params=st.param_vec(), 
                reparam=reparam, 
                train_var=train_var
            )
            loss.backward()
            self._clip_gradients(grouped, stage.grad_clip_norm)
            opt.step()
            lr_scheduler.step()

            recorder.record(i + 1, loss.item(), nominal)
            if (i + 1) % self.loss_every == 0:
                dashboard.update(i + 1, loss.item(), nominal)
            if ((i + 1) % self.frame_every == 0) or (i + 1 == iters):
                frames.append(snapshot(st.pose()))

        assert loss is not None
        assert nominal is not None
        dashboard.finish()
        # The final scalar loss + full nominal-component dict are a debug detail; the
        # dashboard already showed the trend, so keep them out of the normal console.
        if cfg.runtime.verbose:
            print_info(f"Optimization finished, Loss: {loss.item():.4f}")
            print_dict(nominal)

        # Save loss curve (CSV + plot); tag is "base" or "finetune".
        if tag:
            recorder.save_csv(f"loss_curve_{tag}.csv")
            recorder.save_plot(f"loss_curve_{tag}.png")

        # Emit the constraint-param report only when actually training Var params.
        if st.param is not None and train_var:
            emit_constraint_param_report(
                names=constraints.params.names,
                kinds=constraints.params.kinds,
                prior_raw=constraints.params.priors,
                final_raw=st.param.detach().tolist(),
                save_dir=constraints.save_dir,
                report_tag=tag,
            )

        return snapshot(st.pose()), frames
