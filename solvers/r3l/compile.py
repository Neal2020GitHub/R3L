"""
Compile a constraint JSON into the objective the layout solver minimizes
(like re.compile -> re.Pattern: built once, evaluated every optimizer step).

The JSON is the contract produced by dsl.code_to_json. Compiling it yields a
CompiledConstraints with three parts:
  - SceneIndex: which objects stand alone and which form clusters;
  - ParamTable: the schema of the learnable constraint parameters;
  - the objective: a {loss_name: LossTerm} dict, one term per constraint type.

The same constraint type can be stated in several places at once — among the
scene-wide rules, and inside each cluster — and all of its instances become one
term in that dict. The original JSON is also kept verbatim as `spec`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from utils.r3l.types import LossTerm, ParamTable, BBoxVec
from solvers.r3l.cluster import ClusterMeta, SceneIndex, build_pairs
from solvers.r3l.constraints import CompiledConstraints
from solvers.r3l.config import cfg
from solvers.r3l.losses import CENTERLINE_RAD
from solvers.r3l import builders


def _to_long(v) -> torch.Tensor:
    return torch.tensor(v, device=cfg.runtime.device, dtype=torch.long)


def _to_f32(v) -> torch.Tensor:
    return torch.tensor(v, device=cfg.runtime.device, dtype=torch.float32)


WALL_MAP = {"L": 0, "R": 1, "T": 2, "B": 3} # Which room wall an entity references (against_wall / corner alignment / facing wall).
CORNER_MAP = {"BL": 0, "BR": 1, "TR": 2, "TL": 3} # Which room corner an entity is placed at.

LONG, F32 = "long", "f32"
_COL_CAST = {LONG: _to_long, F32: _to_f32}


@dataclass(frozen=True)
class _Spec:
    """The fixed description of how one constraint type becomes a loss term.

    `make` is the function that builds the term. `columns` names every value an
    instance of this type carries, keyed by the `make` argument it fills, because the
    term is built as make(**columns); a column key that is not an argument of `make`
    is a hard error, never a silent mismatch. Two specs may share one output `name`
    when that loss is produced by more than one `make` function.
    """
    key: str                          # registry id (unique)
    name: str                         # output loss name (may repeat across specs)
    make: Callable[..., LossTerm]     # builders.make_*
    columns: Dict[str, str]           # maker-kwarg -> LONG | F32
    param_kind_kw: Optional[str] = None  # the make_* kwarg taking the (homogeneous) param kind
    room_size: bool = False           # inject ctx.room_size at finalize


# The values common to the four directional constraints (left / right / in-front / behind).
_DIR_COLS = {"source_index": LONG, "target_index": LONG, "percentile": F32, "percentile_param_idx": LONG}

# One spec per constraint type whose instances are independent and so can share a
# single term. `around` is not here: the objects in one around-rule are placed
# relative to each other, so each rule becomes its own term during the walk.
_SPECS: Tuple[_Spec, ...] = (
    _Spec("against_wall", "against_wall_loss", builders.make_against_wall,
          {"index": LONG, "wall_index": LONG}, room_size=True),
    _Spec("corner", "corner_loss", builders.make_corner,
          {"index": LONG, "corner_index": LONG, "wall_index": LONG}, room_size=True),
    _Spec("horizontal_abs", "horizontal_abs_loss", builders.make_horizontal_abs,
          {"index": LONG, "x": F32, "x_param_idx": LONG}, "x_param_kind", room_size=True),
    _Spec("horizontal_rel", "horizontal_rel_loss", builders.make_horizontal_rel,
          {"index": LONG, "percentile": F32, "percentile_param_idx": LONG}, "percentile_param_kind", room_size=True),
    _Spec("vertical_abs", "vertical_abs_loss", builders.make_vertical_abs,
          {"index": LONG, "y": F32, "y_param_idx": LONG}, "y_param_kind", room_size=True),
    _Spec("vertical_rel", "vertical_rel_loss", builders.make_vertical_rel,
          {"index": LONG, "percentile": F32, "percentile_param_idx": LONG}, "percentile_param_kind", room_size=True),
    _Spec("fr", "fr_loss", builders.make_facing_radial, {"source_index": LONG, "target_index": LONG}),
    _Spec("fo", "fo_loss", builders.make_facing_ortho, {"source_index": LONG, "target_index": LONG}),
    _Spec("fw", "fw_loss", builders.make_facing_wall, {"source_index": LONG, "wall_index": LONG}),
    _Spec("gap", "gap_loss", builders.make_gap,
          {"source_index": LONG, "target_index": LONG, "gap": F32, "gap_param_idx": LONG}, "gap_param_kind"),
    _Spec("distance", "distance_loss", builders.make_distance,
          {"source_index": LONG, "target_index": LONG, "distance": F32, "distance_param_idx": LONG}, "distance_param_kind"),
    _Spec("left_of", "left_of_loss", builders.make_left_of, _DIR_COLS, "percentile_param_kind"),
    _Spec("right_of", "right_of_loss", builders.make_right_of, _DIR_COLS, "percentile_param_kind"),
    _Spec("in_front_of", "in_front_of_loss", builders.make_in_front_of, _DIR_COLS, "percentile_param_kind"),
    _Spec("behind_of", "behind_of_loss", builders.make_behind_of, _DIR_COLS, "percentile_param_kind"),
    _Spec("in_front_of_dir_only", "in_front_of_loss", builders.make_in_front_of_dir_only,
          {"source_index": LONG, "target_index": LONG}),
    _Spec("angle", "angle_loss", builders.make_angle,
          {"source_index": LONG, "target_index": LONG, "angle_deg": F32, "angle_param_idx": LONG}, "angle_param_kind"),
)


class _Accumulator:
    """Holds every instance of one constraint type, to be built into a single term.

    A constraint type — say "chair faces table" — can be stated among the scene-wide
    rules and inside individual furniture clusters. This gathers all of its instances
    as the constraint JSON is read, and the gathered instances become one loss term
    for that type. Every instance must share one learnable-parameter kind, which
    `append` checks.
    """

    def __init__(self, spec: _Spec) -> None:
        self.spec = spec
        self.cols: Dict[str, list] = {col: [] for col in spec.columns}
        self.kind: Optional[str] = None

    @property
    def empty(self) -> bool:
        return all(not col for col in self.cols.values())

    def append(self, row: Dict[str, object], kind: Optional[str]) -> None:
        assert row.keys() == self.cols.keys(), \
            f"{self.spec.key}: row keys {set(row)} != columns {set(self.cols)}"
        for col, value in row.items():
            self.cols[col].append(value)
        if self.spec.param_kind_kw is not None:
            assert kind is not None, f"{self.spec.key}: parametric spec needs a param kind"
            assert self.kind in (None, kind), \
                f"{self.spec.key}: non-homogeneous param kind {self.kind!r} vs {kind!r}"
            self.kind = kind

    def build(self, room_size: Tuple[float, float]) -> LossTerm:
        tens = {col: _COL_CAST[dt](self.cols[col]) for col, dt in self.spec.columns.items()}  # tensorized columns
        if self.spec.room_size:
            tens["room_size"] = room_size
        if self.spec.param_kind_kw is not None:
            tens[self.spec.param_kind_kw] = self.kind
        return self.spec.make(**tens)


def _sum_terms(terms: List[LossTerm]) -> LossTerm:
    """Add the terms that share one loss name into a single term.

    Most loss names have exactly one term. A name carries several only when its
    constraint type is built per-rule (around) or by more than one `make` function
    (in_front_of).
    """
    if len(terms) == 1:
        return terms[0]

    def evaluate_fn(aug, params=None):
        loss, nominal = terms[0].evaluate(aug, params)
        for term in terms[1:]:
            l, n = term.evaluate(aug, params)
            loss, nominal = loss + l, nominal + n
        return loss, nominal
    return LossTerm(evaluate_fn=evaluate_fn)


def compile(
    constr_json: dict,
    object_to_index: Dict[str, int],
    bbox_vec: BBoxVec,
    room_size: Tuple[float, float],
    save_dir: str = "",
) -> CompiledConstraints:
    """Compile a constraint JSON into a CompiledConstraints (see the module docstring).

    `object_to_index` fixes the 0..N-1 numbering that every constraint's object
    references resolve to. `constr_json` is stored unchanged on the result as `spec`.
    """
    blocks = constr_json.get("constraints", {})

    params = _walk_params(constr_json.get("constraint_params", {}))
    scene = _walk_entities(constr_json.get("scene_entities", {}), object_to_index)

    ctx = _Context(scene=scene, params=params, room_size=room_size)
    _walk_composition(ctx, blocks.get("composition", {}))
    for cid, rules in blocks.get("cluster_internal", {}).items():
        _walk_relational(ctx, rules, in_cluster=cid)
    _walk_relational(ctx, blocks.get("scene_relational", {}), in_cluster=None)

    return CompiledConstraints(
        scene=scene,
        constraints=ctx.finalize(),
        params=params,
        bbox_vec=bbox_vec,
        room_size=room_size,
        save_dir=save_dir,
        spec=constr_json,
    )


class _Context:
    """The working state carried through one walk of a constraint JSON.

    It keeps one `_Accumulator` per constraint type plus any terms emitted directly,
    and resolves entity names and parameter names to indices. `finalize` produces the
    {loss_name: LossTerm} objective from it.
    """

    def __init__(self, scene: SceneIndex, params: ParamTable, room_size: Tuple[float, float]):
        self.scene = scene
        self.params = params
        self.room_size = room_size
        self._name_to_idx = {name: i for i, name in enumerate(params.names)}
        self.acc: Dict[str, _Accumulator] = {spec.key: _Accumulator(spec) for spec in _SPECS}
        self.terms: Dict[str, List[LossTerm]] = {}

    def append(self, key: str, row: Dict[str, object], kind: Optional[str] = None) -> None:
        self.acc[key].append(row, kind)

    def emit(self, name: str, term: LossTerm) -> None:
        self.terms.setdefault(name, []).append(term)

    def finalize(self) -> Dict[str, LossTerm]:
        for acc in self.acc.values():
            if not acc.empty:
                self.emit(acc.spec.name, acc.build(self.room_size))
        return {name: _sum_terms(terms) for name, terms in self.terms.items()}

    def resolve(self, kind: str, id_str: str, cluster_context: Optional[str] = None) -> int:
        match kind:
            case "object": return self.scene.object_to_index[id_str]
            case "cluster": return self.scene.cluster_to_index[id_str]
            case "anchor":
                assert cluster_context, "Anchor resolution requires cluster context"
                return self.scene.clusters[cluster_context].anchor_index
            case _: raise ValueError(f"Invalid endpoint kind: {kind}")

    def param_idx(self, item: dict, field: str, constr_type: str) -> int:
        key = f"{field}_param"
        if key not in item:
            raise ValueError(
                f"[{constr_type}] variable mode requires '{key}' field, but it's missing. "
                f"Ensure DSL uses Var(...) for '{field}'."
            )
        name = item[key]
        if name not in self._name_to_idx:
            raise ValueError(
                f"[{constr_type}] '{key}'='{name}' not found in constraint_params. "
                f"Available: {list(self._name_to_idx.keys())}"
            )
        return self._name_to_idx[name]

    def param_kind(self, idx: int) -> str:
        return self.params.kind_of(idx)


def _walk_params(constraint_params: dict) -> ParamTable:
    """Build the param schema from the constraint_params JSON section."""
    names = list(constraint_params.get("names", []))
    priors = [float(p) for p in constraint_params.get("priors", [])]
    kinds = list(constraint_params.get("kinds", []))
    if not (len(names) == len(priors) == len(kinds)):
        raise ValueError(
            f"constraint_params arrays have inconsistent lengths: "
            f"names={len(names)}, priors={len(priors)}, kinds={len(kinds)}"
        )
    return ParamTable(names=names, priors=priors, kinds=kinds)


def _walk_entities(entities: dict, object_to_index: Dict[str, int]) -> SceneIndex:
    """Build the static scene topology (independent objects + clusters)."""
    N = len(object_to_index)

    indep: List[str] = entities.get("independent_objects", [])
    for obj_id in indep:
        assert obj_id in object_to_index, f"Unknown object: {obj_id}"

    clusters: Dict[str, ClusterMeta] = {}
    for clu in entities.get("clusters", []):
        cid = clu["cluster_id"]
        anchor = clu["anchor"]
        assert anchor["anchor_kind"] == "object", "virtual anchors unsupported"
        anchor_obj = anchor["anchor_object_id"]
        assert anchor_obj in object_to_index, f"Unknown anchor object: {anchor_obj}"
        anchor_idx = object_to_index[anchor_obj]

        mem_indices, non_anchor_indices = [], []
        for m in clu["members"]:
            assert m in object_to_index, f"Unknown member: {m}"
            idx = object_to_index[m]
            mem_indices.append(idx)
            if idx != anchor_idx:
                non_anchor_indices.append(idx)

        clusters[cid] = ClusterMeta(
            member_indices=torch.tensor(mem_indices, device=cfg.runtime.device, dtype=torch.long),
            non_anchor_indices=torch.tensor(non_anchor_indices, device=cfg.runtime.device, dtype=torch.long),
            anchor_index=anchor_idx,
        )

    sorted_cids = sorted(clusters.keys())
    cluster_to_index = {cid: N + k for k, cid in enumerate(sorted_cids)}

    indep_idx_list = [object_to_index[oid] for oid in indep]
    independent_indices = (
        torch.tensor(indep_idx_list, device=cfg.runtime.device, dtype=torch.long)
        if indep_idx_list
        else torch.tensor([], device=cfg.runtime.device, dtype=torch.long)
    )

    return SceneIndex(
        N=N,
        independent_indices=independent_indices,
        cluster_to_index=cluster_to_index,
        clusters=clusters,
        object_to_index=dict(object_to_index),
        local_pairs=build_pairs(clusters),
    )


def _walk_composition(ctx: _Context, composition: dict) -> None:
    """Append the composition constraints (room-anchored placements) to their accumulators."""
    for it in composition.get("against_wall", []):
        ctx.append("against_wall", {
            "index": ctx.resolve(it.get("src_kind", "object"), it["src_id"]),
            "wall_index": WALL_MAP[it["wall"]],
        })
    for it in composition.get("corner", []):
        ctx.append("corner", {
            "index": ctx.resolve(it.get("src_kind", "object"), it["src_id"]),
            "corner_index": CORNER_MAP[it["corner"]],
            "wall_index": WALL_MAP[it["wall"]],
        })
    _walk_axis(ctx, composition.get("horizontal", []), constr_type="horizontal", abs_field="x")
    _walk_axis(ctx, composition.get("vertical", []), constr_type="vertical", abs_field="y")


def _walk_axis(ctx: _Context, items: List[dict], *, constr_type: str, abs_field: str) -> None:
    field = abs_field if cfg.prompt.hv_absolute else "percentile"
    key = f"{constr_type}_abs" if cfg.prompt.hv_absolute else f"{constr_type}_rel"
    for it in items:
        pidx = ctx.param_idx(it, field, constr_type)
        ctx.append(key, {
            "index": ctx.resolve(it.get("src_kind", "object"), it["src_id"]),
            field: float(it[field]),
            f"{field}_param_idx": pidx,
        }, kind=ctx.param_kind(pidx))


def _walk_relational(ctx: _Context, rules: dict, *, in_cluster: Optional[str]) -> None:
    resolve = lambda k, i: ctx.resolve(k, i, in_cluster)
    anchor_idx = ctx.scene.clusters[in_cluster].anchor_index if in_cluster else None

    # Facing: radial/ortho point at a target, wall faces a room wall. A mutual
    # facing involving the origin-fixed anchor degenerates into a direction-only
    # push of the member into the anchor's +y half-plane.
    for it in rules.get("facing", []):
        s = resolve(it["src_kind"], it["src_id"])
        if it["tar_kind"] == "wall":
            ctx.append("fw", {"source_index": s, "wall_index": WALL_MAP[it["tar_id"]]})
            continue
        t, mut = resolve(it["tar_kind"], it["tar_id"]), it["mutual"]
        if it["mode"] == "radial":
            ctx.append("fr", {"source_index": s, "target_index": t})
            if mut:
                ctx.append("fr", {"source_index": t, "target_index": s})
        elif it["mode"] == "ortho":
            ctx.append("fo", {"source_index": s, "target_index": t})
            if mut:
                ctx.append("fo", {"source_index": t, "target_index": s})
        if in_cluster and mut and (it["src_kind"] == "anchor" or it["tar_kind"] == "anchor"):
            member = t if it["src_kind"] == "anchor" else s
            ctx.append("in_front_of_dir_only", {"source_index": member, "target_index": anchor_idx})

    # Directional: the placement term, plus an embedded clearance feeding gap.
    for key in ("left_of", "right_of", "in_front_of", "behind_of"):
        for it in rules.get(key, []):
            s, t = resolve(it["src_kind"], it["src_id"]), resolve(it["tar_kind"], it["tar_id"])
            pidx = ctx.param_idx(it, "percentile", key)
            ctx.append(key, {
                "source_index": s, "target_index": t,
                "percentile": float(it.get("percentile", 0.5)),
                "percentile_param_idx": pidx,
            }, kind=ctx.param_kind(pidx))
            gidx = ctx.param_idx(it, "clearance", key)
            ctx.append("gap", {
                "source_index": s, "target_index": t,
                "gap": float(it["clearance"]),
                "gap_param_idx": gidx,
            }, kind=ctx.param_kind(gidx))

    # Around: per-rule (its sources are spatially coupled), so emit one term each;
    # the embedded target distance feeds the shared distance accumulator.
    for it in rules.get("around", []):
        t = resolve(it["tar_kind"], it["tar_id"])
        srcs = [resolve(x["src_kind"], x["src_id"]) for x in it["src"]]
        sweep_pidx = ctx.param_idx(it, "sweep_deg", "around")
        ctx.emit("around_loss", builders.make_around(
            _to_long(srcs), _to_long([t]),
            _to_f32([float(it["sweep_deg"])]), _to_f32([CENTERLINE_RAD[it.get("centerline", "T")]]),
            sweep_deg_param_idx=_to_long([sweep_pidx]), sweep_deg_param_kind=ctx.param_kind(sweep_pidx),
        ))
        dist_pidx = ctx.param_idx(it, "distance", "around")
        dist_kind = ctx.param_kind(dist_pidx)
        for s in srcs:
            ctx.append("distance", {
                "source_index": s, "target_index": t,
                "distance": float(it["distance"]),
                "distance_param_idx": dist_pidx,
            }, kind=dist_kind)

    # Angle: signed yaw offset from the target.
    for it in rules.get("angle", []):
        s, t = resolve(it["src_kind"], it["src_id"]), resolve(it["tar_kind"], it["tar_id"])
        pidx = ctx.param_idx(it, "angle", "angle")
        ctx.append("angle", {
            "source_index": s, "target_index": t,
            "angle_deg": float(it["angle"]),
            "angle_param_idx": pidx,
        }, kind=ctx.param_kind(pidx))
