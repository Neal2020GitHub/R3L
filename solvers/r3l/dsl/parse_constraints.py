from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


# =============================================================================
# Public IR
# =============================================================================

Scope = Literal["global", "cluster_internal", "group_internal"]
EntityKind = Literal["object", "cluster"]
ParamKind = Literal["unit", "nonneg", "angle_deg"]


class DslParseError(RuntimeError):
    """Fail-fast error for invalid / out-of-spec v2 DSL programs."""


@dataclass(slots=True, frozen=True)
class EntityRef:
    kind: EntityKind  # "object" | "cluster"
    id: str           # object_id | cluster_id


@dataclass(slots=True, frozen=True)
class VarDecl:
    """A Var(...) declaration in the DSL."""
    name: str           # Python variable name
    prior: Union[float, str]  # numeric prior or alignment label string


@dataclass(slots=True, frozen=True)
class VarRef:
    """A reference to a declared Var variable."""
    name: str           # Python variable name referencing a VarDecl


@dataclass(slots=True)
class ParamInfo:
    """Accumulated parameter info during parsing."""
    name: str
    prior: float  # numeric prior (label converted to percentile)
    kind: Optional[ParamKind] = None  # inferred from usage


@dataclass(slots=True, frozen=True)
class CallIR:
    """
    Normalized solver call.

    Notes:
    - `name` uses JSON-level semantics: solver.align always becomes `angle`.
    - `args` contain Python values: floats/bools/strings, plus EntityRef (or lists of EntityRef).
    - Optimizable fields contain VarRef.
    """
    name: str
    args: Dict[str, Any]
    scope: Scope
    cluster_id: Optional[str] = None


@dataclass(slots=True, frozen=True)
class ClusterIR:
    cluster_id: str
    handle_name: str
    anchor_obj_id: str
    member_obj_ids: Tuple[str, ...]
    internal_calls: Tuple[CallIR, ...]


@dataclass(slots=True, frozen=True)
class GroupIR:
    """Semantic group for scope isolation (flat representation, no anchor)."""
    group_id: str
    member_obj_ids: Tuple[str, ...]
    internal_calls: Tuple[CallIR, ...]


@dataclass(slots=True, frozen=True)
class ProgramIR:
    object_ids: Tuple[str, ...]
    clusters: Tuple[ClusterIR, ...]
    groups: Tuple[GroupIR, ...]  # semantic groups
    global_calls: Tuple[CallIR, ...]
    handle_to_cluster: Dict[str, str]
    cluster_anchor: Dict[str, str]  # cluster_id -> anchor_obj_id
    var_decls: Tuple[VarDecl, ...]
    param_infos: Tuple[ParamInfo, ...]  # finalized params with inferred kinds


# =============================================================================
# Parsing
# =============================================================================

_WALLS = {"L", "R", "T", "B"}

# positional-arg order; keyword args are also supported
# NOTE: directional constraints have NO default for alignment (must be explicit Var)
_CALL_SIG: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]] = {
    "against_wall": (("source", "wall"), {}),
    "corner": (("source", "corner", "wall"), {}),
    "horizontal": (("source", "percentile"), {}),
    "vertical": (("source", "percentile"), {}),
    "left_of": (("source", "target", "clearance", "alignment"), {}),
    "right_of": (("source", "target", "clearance", "alignment"), {}),
    "in_front_of": (("source", "target", "clearance", "alignment"), {}),
    "behind_of": (("source", "target", "clearance", "alignment"), {}),
    "around": (("source_list", "target", "distance", "sweep_deg", "centerline"), {}),
    "facing": (("source", "target", "mode", "mutual"), {"mode": "ortho", "mutual": False}),
    "align": (("source", "target", "angle"), {}),
}

# NOTE: horizontal/vertical are config-dependent:
# - rel mode: percentile in [0, 1]
# - abs mode: absolute x/y in meters
def _call_sig(hv_absolute: bool) -> Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]]:
    sig = dict(_CALL_SIG)
    if not hv_absolute:
        return sig

    sig["horizontal"] = (("source", "x"), {})
    sig["vertical"] = (("source", "y"), {})
    return sig

# -----------------------------------------------------------------------------
# Alignment label → percentile mapping (directional constraints)
# -----------------------------------------------------------------------------
_LATERAL_ALIGN = {"backboard": 0.0, "center": 0.5, "frontboard": 1.0}
_FRONTAL_ALIGN = {"left": 0.0, "center": 0.5, "right": 1.0}
_DIRECTIONAL_TYPES = {"left_of", "right_of", "in_front_of", "behind_of"}
_LATERAL_TYPES = {"left_of", "right_of"}


def _alignment_label_to_percentile(label: str) -> float:
    """Convert alignment label string to percentile value."""
    if label in _LATERAL_ALIGN:
        return _LATERAL_ALIGN[label]
    if label in _FRONTAL_ALIGN:
        return _FRONTAL_ALIGN[label]
    raise DslParseError(
        f"Invalid alignment label '{label}'. Valid labels: "
        f"{list(_LATERAL_ALIGN.keys())} or {list(_FRONTAL_ALIGN.keys())}"
    )


_CLUSTER_SIG = (("cluster_id", "anchor", "members"), {})
_GROUP_SIG = (("group_id", "members"), {})  # No anchor parameter
_GLOBAL_ONLY = {"against_wall", "corner", "horizontal", "vertical"}
_COMP_KEYS = ("against_wall", "corner", "horizontal", "vertical")
_REL_KEYS = ("facing", "left_of", "right_of", "in_front_of", "behind_of", "around", "angle")


@dataclass(slots=True)
class _Ctx:
    var_to_obj_id: Dict[str, str]
    handle_to_cluster: Dict[str, str]
    cluster_member_obj_ids: set[str]
    var_decls: Dict[str, VarDecl] = field(default_factory=dict)
    param_usages: Dict[str, List[ParamKind]] = field(default_factory=dict)


def parse_program(code: str, var_to_obj_id: Dict[str, str], *, hv_absolute: bool) -> ProgramIR:
    """
    Parse v2 DSL python code (string) into a unified IR.

    All numeric constraint parameters MUST be Var references.

    Supported surface syntax:
    - top-level `solver.<constraint>(...)` calls
    - cluster blocks: `with solver.cluster(... ) as <handle>: ...`
    - positional and/or keyword args

    Ignored statements:
    - room/walls definitions, assignments (except Var), comments, etc.
    """
    mod = _parse_ast(code)
    object_ids = tuple(var_to_obj_id.values())

    var_decls = _parse_var_decls(mod)
    call_sig = _call_sig(hv_absolute)
    param_usages: Dict[str, List[ParamKind]] = {name: [] for name in var_decls}

    clusters, groups, handle_to_cluster, anchor_by_cluster = _parse_groups_and_clusters(
        mod,
        var_to_obj_id,
        call_sig=call_sig,
        hv_absolute=hv_absolute,
        var_decls=var_decls,
        param_usages=param_usages,
    )
    member_ids = {m for c in clusters for m in c.member_obj_ids}
    ctx = _Ctx(
        var_to_obj_id=var_to_obj_id,
        handle_to_cluster=handle_to_cluster,
        cluster_member_obj_ids=member_ids,
        var_decls=var_decls,
        param_usages=param_usages,
    )

    global_calls = _parse_global_calls(mod, ctx, call_sig=call_sig, hv_absolute=hv_absolute)
    param_infos = _finalize_param_infos(ctx)

    return ProgramIR(
        object_ids=object_ids,
        clusters=tuple(clusters),
        groups=tuple(groups),
        global_calls=tuple(global_calls),
        handle_to_cluster=handle_to_cluster,
        cluster_anchor=anchor_by_cluster,
        var_decls=tuple(var_decls.values()),
        param_infos=param_infos,
    )


def code_to_json(program: ProgramIR) -> dict:
    """
    Convert parsed v2 DSL IR into the JSON schema consumed by `compile`.

    Always produces constraint_params section with parameter metadata.
    """
    clusters = list(program.clusters)
    groups = list(program.groups)
    cluster_member_set = {m for c in clusters for m in c.member_obj_ids}
    # Group members ARE independent objects (flat representation)

    # Independent objects: all objects not in spatial clusters
    independent = [oid for oid in program.object_ids if oid not in cluster_member_set]

    def empty_comp():
        return {k: [] for k in _COMP_KEYS}

    def empty_rel():
        return {k: [] for k in _REL_KEYS}

    out = {
        "scene_entities": {
            "independent_objects": independent,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "anchor": {"anchor_kind": "object", "anchor_object_id": c.anchor_obj_id},
                    "members": list(c.member_obj_ids),
                }
                for c in clusters
            ],
        },
        "constraints": {
            "composition": empty_comp(),
            "cluster_internal": {c.cluster_id: empty_rel() for c in clusters},
            "scene_relational": empty_rel(),
        },
    }

    # Add semantic_groups section (optional, only if groups exist)
    if groups:
        out["semantic_groups"] = {
            g.group_id: list(g.member_obj_ids) for g in groups
        }

    # Always add constraint_params section
    if program.param_infos:
        out["constraint_params"] = {
            "names": [p.name for p in program.param_infos],
            "priors": [p.prior for p in program.param_infos],
            "kinds": [p.kind for p in program.param_infos],
        }

    comp = out["constraints"]["composition"]
    rel = out["constraints"]["scene_relational"]
    internal = out["constraints"]["cluster_internal"]

    for call in program.global_calls:
        _emit_global_call(call, comp, rel)

    for c in clusters:
        dst = internal[c.cluster_id]
        for call in c.internal_calls:
            _emit_internal_call(call, dst, anchor_obj_id=c.anchor_obj_id)

    # Emit group constraints to scene_relational
    for g in groups:
        for call in g.internal_calls:
            _emit_global_call(call, comp, rel)

    return out


# =============================================================================
# Var declaration parsing
# =============================================================================

def _parse_var_decls(mod: ast.Module) -> Dict[str, VarDecl]:
    """Parse all `name = Var(literal)` declarations from the module."""
    decls: Dict[str, VarDecl] = {}

    for stmt in mod.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        var_name = target.id

        if not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "Var"):
            continue

        if len(call.args) != 1 or call.keywords:
            raise DslParseError(
                f"Var declaration '{var_name}' must have exactly one positional argument: "
                f"Var(<numeric_literal>) or Var(<string_label>)"
            )

        arg = call.args[0]
        prior: Union[float, str]

        if isinstance(arg, ast.Constant):
            if isinstance(arg.value, (int, float)) and not isinstance(arg.value, bool):
                prior = float(arg.value)
            elif isinstance(arg.value, str):
                prior = arg.value
            else:
                raise DslParseError(
                    f"Var('{var_name}'): argument must be a number or string literal, "
                    f"got {type(arg.value).__name__}"
                )
        elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, (ast.UAdd, ast.USub)):
            prior = _eval_num(arg)
        else:
            raise DslParseError(
                f"Var('{var_name}'): argument must be a literal, got {type(arg).__name__}"
            )

        if var_name in decls:
            raise DslParseError(f"Duplicate Var declaration: '{var_name}'")

        decls[var_name] = VarDecl(name=var_name, prior=prior)

    return decls


def _finalize_param_infos(ctx: _Ctx) -> Tuple[ParamInfo, ...]:
    """Finalize parameter infos by inferring kinds from usage."""
    infos: List[ParamInfo] = []

    for var_name, decl in ctx.var_decls.items():
        usages = ctx.param_usages.get(var_name, [])

        if not usages:
            # Unused variables are discarded - they have no effect on the constraint system
            continue

        unique_kinds = set(usages)
        if len(unique_kinds) > 1:
            raise DslParseError(
                f"Var '{var_name}' is used with conflicting kinds: {unique_kinds}. "
                f"Each variable must have a consistent type across all usages."
            )

        kind = usages[0]

        if isinstance(decl.prior, str):
            if kind != "unit":
                raise DslParseError(
                    f"Var '{var_name}' has string prior '{decl.prior}' but is used as {kind}. "
                    f"String priors (alignment labels) are only valid for 'unit' kind."
                )
            prior_val = _alignment_label_to_percentile(decl.prior)
        else:
            prior_val = decl.prior

        infos.append(ParamInfo(name=var_name, prior=prior_val, kind=kind))

    return tuple(infos)


def _record_param_usage(ctx: _Ctx, var_name: str, kind: ParamKind) -> None:
    """Record a parameter usage with its kind for later validation."""
    if var_name in ctx.param_usages:
        ctx.param_usages[var_name].append(kind)


# =============================================================================
# Code-to-JSON helpers
# =============================================================================

def _emit_global_call(call: CallIR, comp: dict, rel: dict) -> None:
    a = call.args

    def _val_and_param(key: str) -> Tuple[float, str]:
        """Extract prior value and param name from VarRef arg."""
        v = a[key]
        assert isinstance(v, VarRef), f"Expected VarRef for {key}, got {type(v)}"
        return float(a[f"{key}_prior"]), v.name

    def _add_param_field(item: dict, key: str, param_name: str) -> None:
        item[f"{key}_param"] = param_name

    src: EntityRef
    tar: EntityRef
    srcs: List[EntityRef]
    match call.name:
        case "against_wall":
            src = a["source"]
            comp["against_wall"].append({
                "src_kind": src.kind, "src_id": src.id,
                "wall": a["wall"],
            })
        case "corner":
            src = a["source"]
            comp["corner"].append({"src_kind": src.kind, "src_id": src.id, "corner": a["corner"], "wall": a["wall"]})
        case "horizontal":
            src = a["source"]
            if "x" in a:
                x, x_param = _val_and_param("x")
                item = {"src_kind": src.kind, "src_id": src.id, "x": x}
                _add_param_field(item, "x", x_param)
            else:
                perc, perc_param = _val_and_param("percentile")
                item = {"src_kind": src.kind, "src_id": src.id, "percentile": perc}
                _add_param_field(item, "percentile", perc_param)
            comp["horizontal"].append(item)
        case "vertical":
            src = a["source"]
            if "y" in a:
                y, y_param = _val_and_param("y")
                item = {"src_kind": src.kind, "src_id": src.id, "y": y}
                _add_param_field(item, "y", y_param)
            else:
                perc, perc_param = _val_and_param("percentile")
                item = {"src_kind": src.kind, "src_id": src.id, "percentile": perc}
                _add_param_field(item, "percentile", perc_param)
            comp["vertical"].append(item)
        case "facing":
            src = a["source"]
            rel["facing"].append({
                "src_kind": src.kind, "src_id": src.id,
                "tar_kind": a["tar_kind"], "tar_id": a["tar_id"],
                "mutual": bool(a["mutual"]), "mode": a["mode"],
            })
        case "angle":
            src = a["source"]
            tar = a["target"]
            angle_val, angle_param = _val_and_param("angle")
            item = {
                "src_kind": src.kind, "src_id": src.id,
                "tar_kind": tar.kind, "tar_id": tar.id,
                "angle": angle_val,
            }
            _add_param_field(item, "angle", angle_param)
            rel["angle"].append(item)
        case "around":
            srcs = a["source_list"]
            tar = a["target"]
            dist, dist_param = _val_and_param("distance")
            sweep, sweep_param = _val_and_param("sweep_deg")
            item = {
                "src": [{"src_kind": s.kind, "src_id": s.id} for s in srcs],
                "tar_kind": tar.kind, "tar_id": tar.id,
                "distance": dist, "sweep_deg": sweep, "centerline": a["centerline"],
            }
            _add_param_field(item, "distance", dist_param)
            _add_param_field(item, "sweep_deg", sweep_param)
            rel["around"].append(item)
        case "left_of" | "right_of" | "in_front_of" | "behind_of":
            src = a["source"]
            tar = a["target"]
            clr, clr_param = _val_and_param("clearance")
            perc, perc_param = _val_and_param("percentile")
            item = {
                "src_kind": src.kind, "src_id": src.id,
                "tar_kind": tar.kind, "tar_id": tar.id,
                "clearance": clr, "percentile": perc,
            }
            _add_param_field(item, "clearance", clr_param)
            _add_param_field(item, "percentile", perc_param)
            rel[call.name].append(item)
        case _:
            raise ValueError(f"Unknown global call: {call.name}")


def _emit_internal_call(call: CallIR, dst: dict, *, anchor_obj_id: str) -> None:
    a = call.args
    ki = lambda ref: _internal_kind_id(anchor_obj_id, ref)

    def _val_and_param(key: str) -> Tuple[float, str]:
        v = a[key]
        assert isinstance(v, VarRef), f"Expected VarRef for {key}, got {type(v)}"
        return float(a[f"{key}_prior"]), v.name

    def _add_param_field(item: dict, key: str, param_name: str) -> None:
        item[f"{key}_param"] = param_name

    match call.name:
        case "facing":
            src_k, src_id = ki(a["source"])
            tar_k, tar_id = _internal_kind_id_str(anchor_obj_id, str(a["tar_kind"]), str(a["tar_id"]))
            dst["facing"].append({
                "src_kind": src_k, "src_id": src_id,
                "tar_kind": tar_k, "tar_id": tar_id,
                "mutual": bool(a["mutual"]), "mode": a["mode"],
            })
        case "angle":
            src_k, src_id = ki(a["source"])
            tar_k, tar_id = ki(a["target"])
            angle_val, angle_param = _val_and_param("angle")
            item = {
                "src_kind": src_k, "src_id": src_id,
                "tar_kind": tar_k, "tar_id": tar_id,
                "angle": angle_val,
            }
            _add_param_field(item, "angle", angle_param)
            dst["angle"].append(item)
        case "around":
            srcs: List[EntityRef] = a["source_list"]
            tar: EntityRef = a["target"]
            tar_k, tar_id = ki(tar)
            dist, dist_param = _val_and_param("distance")
            sweep, sweep_param = _val_and_param("sweep_deg")
            item = {
                "src": [_internal_src_item(anchor_obj_id, s) for s in srcs],
                "tar_kind": tar_k, "tar_id": tar_id,
                "distance": dist, "sweep_deg": sweep, "centerline": a["centerline"],
            }
            _add_param_field(item, "distance", dist_param)
            _add_param_field(item, "sweep_deg", sweep_param)
            dst["around"].append(item)
        case "left_of" | "right_of" | "in_front_of" | "behind_of":
            src_k, src_id = ki(a["source"])
            tar_k, tar_id = ki(a["target"])
            clr, clr_param = _val_and_param("clearance")
            perc, perc_param = _val_and_param("percentile")
            item = {
                "src_kind": src_k, "src_id": src_id,
                "tar_kind": tar_k, "tar_id": tar_id,
                "clearance": clr, "percentile": perc,
            }
            _add_param_field(item, "clearance", clr_param)
            _add_param_field(item, "percentile", perc_param)
            dst[call.name].append(item)
        case _:
            raise ValueError(f"Unknown internal call: {call.name}")


def _internal_kind_id(anchor_obj_id: str, ref: EntityRef) -> Tuple[str, str]:
    return ("anchor", "anchor") if ref.kind == "object" and ref.id == anchor_obj_id else (ref.kind, ref.id)


def _internal_kind_id_str(anchor_obj_id: str, kind: str, id_: str) -> Tuple[str, str]:
    if kind == "object" and id_ == anchor_obj_id:
        return "anchor", "anchor"
    return kind, id_


def _internal_src_item(anchor_obj_id: str, ref: EntityRef) -> dict:
    k, i = _internal_kind_id(anchor_obj_id, ref)
    return {"src_kind": k, "src_id": i}


# =============================================================================
# AST helpers
# =============================================================================

def _parse_ast(code: str) -> ast.Module:
    try:
        return ast.parse(code)
    except SyntaxError as e:
        raise DslParseError(f"DSL python parse failed: {e.msg} (line {e.lineno})") from e


def _solver_attr(call: ast.Call) -> Optional[str]:
    fn = call.func
    if not isinstance(fn, ast.Attribute):
        return None
    if not isinstance(fn.value, ast.Name) or fn.value.id != "solver":
        return None
    return fn.attr


def _is_solver_cluster_call(expr: ast.AST) -> bool:
    return isinstance(expr, ast.Call) and _solver_attr(expr) == "cluster"


def _is_solver_group_call(expr: ast.AST) -> bool:
    return isinstance(expr, ast.Call) and _solver_attr(expr) == "group"


def _as_call_stmt(stmt: ast.stmt) -> Optional[ast.Call]:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    return None


def _must_name(node: ast.AST, what: str) -> str:
    if isinstance(node, ast.Name):
        return node.id
    raise DslParseError(f"Expected {what} to be a variable name, got {type(node).__name__}")


def _must_str(v: Any, what: str) -> str:
    if isinstance(v, str):
        return v
    raise DslParseError(f"Expected {what} to be a string, got {type(v).__name__}")


def _must_bool(v: Any, what: str) -> bool:
    if isinstance(v, bool):
        return v
    raise DslParseError(f"Expected {what} to be a bool, got {type(v).__name__}")


def _eval_num(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_num(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    raise DslParseError(f"Expected a numeric literal, got {type(node).__name__}")


def _eval_const(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _eval_num(node)
    raise DslParseError(f"Expected a literal, got {type(node).__name__}")


def _eval_name_list(node: ast.AST) -> List[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise DslParseError(f"Expected a list/tuple literal, got {type(node).__name__}")
    return [_must_name(x, "list element") for x in node.elts]


def _bind_args(call: ast.Call, params: Tuple[str, ...], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for i, arg in enumerate(call.args):
        if i >= len(params):
            raise DslParseError(f"Too many positional args for solver.{_solver_attr(call)}")
        out[params[i]] = arg

    for kw in call.keywords:
        if kw.arg is None:
            raise DslParseError("**kwargs is not allowed in solver calls")
        if kw.arg not in params:
            raise DslParseError(f"Unknown parameter '{kw.arg}' for solver.{_solver_attr(call)}")
        if kw.arg in out:
            raise DslParseError(f"Duplicate argument '{kw.arg}' for solver.{_solver_attr(call)}")
        out[kw.arg] = kw.value

    for k, v in defaults.items():
        out.setdefault(k, v)

    missing = [p for p in params if p not in out]
    if missing:
        raise DslParseError(f"Missing required args for solver.{_solver_attr(call)}: {missing}")

    return out


# =============================================================================
# Cluster parsing
# =============================================================================

def _parse_groups_and_clusters(
    mod: ast.Module,
    var_to_obj_id: Dict[str, str],
    *,
    call_sig: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]],
    hv_absolute: bool,
    var_decls: Optional[Dict[str, VarDecl]] = None,
    param_usages: Optional[Dict[str, List[ParamKind]]] = None,
) -> Tuple[List[ClusterIR], List[GroupIR], Dict[str, str], Dict[str, str]]:
    clusters: List[ClusterIR] = []
    groups: List[GroupIR] = []
    handle_to_cluster: Dict[str, str] = {}
    anchor_by_cluster: Dict[str, str] = {}
    used_members: set[str] = set()
    used_group_ids: set[str] = set()

    for stmt in mod.body:
        if not isinstance(stmt, ast.With):
            continue

        # Try parsing as cluster first
        clu = _parse_one_cluster(
            stmt,
            var_to_obj_id,
            call_sig=call_sig,
            hv_absolute=hv_absolute,
            var_decls=var_decls,
            param_usages=param_usages,
        )
        if clu is not None:
            if clu.handle_name in handle_to_cluster:
                raise DslParseError(f"Duplicate cluster handle name: {clu.handle_name}")
            if clu.cluster_id in anchor_by_cluster:
                raise DslParseError(f"Duplicate cluster_id: {clu.cluster_id}")
            if used_members.intersection(clu.member_obj_ids):
                raise DslParseError(f"Clusters share members; this is forbidden: {clu.cluster_id}")
            used_members.update(clu.member_obj_ids)

            clusters.append(clu)
            handle_to_cluster[clu.handle_name] = clu.cluster_id
            anchor_by_cluster[clu.cluster_id] = clu.anchor_obj_id
            continue

        # Try parsing as group
        grp = _parse_one_group(
            stmt,
            var_to_obj_id,
            call_sig=call_sig,
            hv_absolute=hv_absolute,
            var_decls=var_decls,
            param_usages=param_usages,
        )
        if grp is not None:
            if grp.group_id in used_group_ids:
                raise DslParseError(f"Duplicate group_id: {grp.group_id}")
            # Note: groups can share members (unlike clusters)
            used_group_ids.add(grp.group_id)
            groups.append(grp)

    return clusters, groups, handle_to_cluster, anchor_by_cluster


def _parse_one_cluster(
    stmt: ast.With,
    var_to_obj_id: Dict[str, str],
    *,
    call_sig: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]],
    hv_absolute: bool,
    var_decls: Optional[Dict[str, VarDecl]] = None,
    param_usages: Optional[Dict[str, List[ParamKind]]] = None,
) -> Optional[ClusterIR]:
    if len(stmt.items) != 1:
        raise DslParseError("Only single-item `with` is supported for solver.cluster")

    item = stmt.items[0]
    if not _is_solver_cluster_call(item.context_expr):
        return None

    if item.optional_vars is None:
        raise DslParseError("Cluster declaration must bind a handle via `as <handle>`")
    handle_name = _must_name(item.optional_vars, "cluster handle")

    # Reject handle names that shadow asset variables (Python scoping confound)
    if handle_name in var_to_obj_id:
        raise DslParseError(
            f"Cluster handle '{handle_name}' shadows asset variable '{handle_name}'. "
            f"Use a distinct name (e.g., '{handle_name}_grp')."
        )

    call: ast.Call = item.context_expr  # type: ignore[assignment]
    params, defaults = _CLUSTER_SIG
    bound = _bind_args(call, params, defaults)

    cluster_id = _must_str(_eval_const(bound["cluster_id"]), "cluster_id")
    anchor_var = _must_name(bound["anchor"], "anchor")
    member_vars = _eval_name_list(bound["members"])

    if anchor_var not in member_vars:
        raise DslParseError(f"Cluster '{cluster_id}': anchor must be included in members")

    anchor_obj = _resolve_asset_var(anchor_var, var_to_obj_id, what="anchor")
    member_obj_ids = tuple(_resolve_asset_var(v, var_to_obj_id, what="member") for v in member_vars)

    _fail_if_nested_cluster(stmt)
    internal_calls = _parse_cluster_body_calls(
        stmt.body, cluster_id, anchor_obj, set(member_obj_ids), var_to_obj_id,
        call_sig=call_sig,
        hv_absolute=hv_absolute,
        var_decls=var_decls,
        param_usages=param_usages,
    )

    return ClusterIR(
        cluster_id=cluster_id,
        handle_name=handle_name,
        anchor_obj_id=anchor_obj,
        member_obj_ids=member_obj_ids,
        internal_calls=tuple(internal_calls),
    )


def _parse_one_group(
    stmt: ast.With,
    var_to_obj_id: Dict[str, str],
    *,
    call_sig: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]],
    hv_absolute: bool,
    var_decls: Optional[Dict[str, VarDecl]] = None,
    param_usages: Optional[Dict[str, List[ParamKind]]] = None,
) -> Optional[GroupIR]:
    """Parse solver.group() context manager - semantic grouping (flat representation)."""
    if len(stmt.items) != 1:
        raise DslParseError("Only single-item `with` is supported for solver.group")

    item = stmt.items[0]
    if not _is_solver_group_call(item.context_expr):
        return None

    # Semantic groups do NOT yield a handle (no 'as' binding allowed)
    if item.optional_vars is not None:
        raise DslParseError(
            "Semantic groups do not yield a handle. "
            "Use `with solver.group(...)` without 'as' clause."
        )

    call: ast.Call = item.context_expr  # type: ignore[assignment]
    params, defaults = _GROUP_SIG
    bound = _bind_args(call, params, defaults)

    group_id = _must_str(_eval_const(bound["group_id"]), "group_id")
    member_vars = _eval_name_list(bound["members"])

    if not member_vars:
        raise DslParseError(f"Group '{group_id}' must have at least one member")

    member_obj_ids = tuple(_resolve_asset_var(v, var_to_obj_id, what="member") for v in member_vars)

    _fail_if_nested_cluster(stmt)  # Reuse nested check
    internal_calls = _parse_group_body_calls(
        stmt.body, group_id, set(member_obj_ids), var_to_obj_id,
        call_sig=call_sig,
        hv_absolute=hv_absolute,
        var_decls=var_decls,
        param_usages=param_usages,
    )

    return GroupIR(
        group_id=group_id,
        member_obj_ids=member_obj_ids,
        internal_calls=tuple(internal_calls),
    )


def _parse_group_body_calls(
    body: List[ast.stmt],
    group_id: str,
    member_obj_ids: set[str],
    var_to_obj_id: Dict[str, str],
    *,
    call_sig: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]],
    hv_absolute: bool,
    var_decls: Optional[Dict[str, VarDecl]] = None,
    param_usages: Optional[Dict[str, List[ParamKind]]] = None,
) -> List[CallIR]:
    """Parse constraint calls inside semantic group (allows composition constraints)."""
    if var_decls is None:
        var_decls = {}
    if param_usages is None:
        param_usages = {name: [] for name in var_decls}

    ctx = _Ctx(
        var_to_obj_id=var_to_obj_id,
        handle_to_cluster={},
        cluster_member_obj_ids=set(),  # Empty - group members NOT cluster members
        var_decls=var_decls,
        param_usages=param_usages,
    )

    calls: List[CallIR] = []
    for stmt in body:
        call = _as_call_stmt(stmt)
        if call is None:
            continue
        c = _parse_solver_call(
            call,
            ctx,
            scope="group_internal",  # NEW scope type
            cluster_id=None,
            member_obj_ids=member_obj_ids,
            call_sig=call_sig,
            hv_absolute=hv_absolute,
        )
        if c is None:
            continue
        calls.append(c)

    return calls


def _fail_if_nested_cluster(stmt: ast.With) -> None:
    for node in ast.walk(stmt):
        if node is stmt:
            continue
        if isinstance(node, ast.With):
            if any(_is_solver_cluster_call(it.context_expr) for it in node.items):
                raise DslParseError("Nested solver.cluster blocks are forbidden")


def _parse_cluster_body_calls(
    body: List[ast.stmt],
    cluster_id: str,
    anchor_obj_id: str,
    member_obj_ids: set[str],
    var_to_obj_id: Dict[str, str],
    *,
    call_sig: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]],
    hv_absolute: bool,
    var_decls: Optional[Dict[str, VarDecl]] = None,
    param_usages: Optional[Dict[str, List[ParamKind]]] = None,
) -> List[CallIR]:
    if var_decls is None:
        var_decls = {}
    if param_usages is None:
        param_usages = {name: [] for name in var_decls}

    ctx = _Ctx(
        var_to_obj_id=var_to_obj_id,
        handle_to_cluster={},
        cluster_member_obj_ids=set(),
        var_decls=var_decls,
        param_usages=param_usages,
    )

    calls: List[CallIR] = []
    for stmt in body:
        call = _as_call_stmt(stmt)
        if call is None:
            continue
        c = _parse_solver_call(
            call,
            ctx,
            scope="cluster_internal",
            cluster_id=cluster_id,
            member_obj_ids=member_obj_ids,
            call_sig=call_sig,
            hv_absolute=hv_absolute,
        )
        if c is None:
            continue
        calls.append(c)

    return calls


# =============================================================================
# Global calls parsing
# =============================================================================

def _parse_global_calls(
    mod: ast.Module,
    ctx: _Ctx,
    *,
    call_sig: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]],
    hv_absolute: bool,
) -> List[CallIR]:
    calls: List[CallIR] = []
    for stmt in mod.body:
        call = _as_call_stmt(stmt)
        if call is None:
            continue
        c = _parse_solver_call(
            call,
            ctx,
            scope="global",
            cluster_id=None,
            member_obj_ids=None,
            call_sig=call_sig,
            hv_absolute=hv_absolute,
        )
        if c is not None:
            calls.append(c)
    return calls


def _parse_solver_call(
    call: ast.Call,
    ctx: _Ctx,
    *,
    scope: Scope,
    cluster_id: Optional[str],
    member_obj_ids: Optional[set[str]],
    call_sig: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]],
    hv_absolute: bool,
) -> Optional[CallIR]:
    name = _solver_attr(call)
    if name is None:
        return None
    if name in ("cluster", "group"):  # Ignore both context managers
        return None
    if name not in call_sig:
        return None
    if scope == "cluster_internal" and name in _GLOBAL_ONLY:
        raise DslParseError(
            f"solver.{name} is forbidden inside spatial cluster '{cluster_id}'. "
            f"Use global scope or semantic groups for composition constraints."
        )
    # Semantic groups (scope=="group_internal") allow composition - no check needed

    # Fail-fast: reject deprecated `percentile=` for directional constraints
    if name in _DIRECTIONAL_TYPES:
        _reject_percentile_arg(call, name)

    params, defaults = call_sig[name]
    bound = _bind_args(call, params, defaults)
    norm_name, norm_args = _normalize_call(
        name,
        bound,
        ctx,
        scope=scope,
        cluster_id=cluster_id,
        member_obj_ids=member_obj_ids,
        hv_absolute=hv_absolute,
    )

    return CallIR(name=norm_name, args=norm_args, scope=scope, cluster_id=cluster_id)


def _reject_percentile_arg(call: ast.Call, name: str) -> None:
    """Fail-fast if directional constraint uses deprecated percentile syntax."""
    valid_labels = _LATERAL_ALIGN if name in _LATERAL_TYPES else _FRONTAL_ALIGN
    hint = ", ".join(f'"{k}"' for k in valid_labels)

    for kw in call.keywords:
        if kw.arg == "percentile":
            raise DslParseError(
                f"solver.{name}: `percentile=` is deprecated. "
                f"Use `alignment=` with one of: {hint}"
            )

    if len(call.args) >= 4:
        arg4 = call.args[3]
        if isinstance(arg4, ast.Constant) and isinstance(arg4.value, (int, float)) and not isinstance(arg4.value, bool):
            raise DslParseError(
                f"solver.{name}: numeric 4th argument (percentile) is deprecated. "
                f"Use `alignment=` with one of: {hint}"
            )


# =============================================================================
# Call normalization
# =============================================================================

def _normalize_call(
    name: str,
    bound: Dict[str, Any],
    ctx: _Ctx,
    *,
    scope: Scope,
    cluster_id: Optional[str],
    member_obj_ids: Optional[set[str]],
    hv_absolute: bool,
) -> Tuple[str, Dict[str, Any]]:
    if name in _GLOBAL_ONLY:
        # Composition constraints always use global entity resolution (even in groups)
        entity_scope: Scope = "global" if scope == "group_internal" else scope
        return name, _norm_unary(name, bound, ctx, scope=entity_scope, hv_absolute=hv_absolute)
    if name in {"left_of", "right_of", "in_front_of", "behind_of"}:
        # Relational constraints in groups use global scope; in clusters use cluster scope
        entity_scope = "global" if scope == "group_internal" else scope
        return name, _norm_directional(name, bound, ctx, scope=entity_scope, member_obj_ids=member_obj_ids)
    if name == "around":
        entity_scope = "global" if scope == "group_internal" else scope
        return name, _norm_around(bound, ctx, scope=entity_scope, member_obj_ids=member_obj_ids)
    if name == "facing":
        entity_scope = "global" if scope == "group_internal" else scope
        return name, _norm_facing(bound, ctx, scope=entity_scope, member_obj_ids=member_obj_ids)
    if name == "align":
        entity_scope = "global" if scope == "group_internal" else scope
        return _norm_align(bound, ctx, scope=entity_scope, member_obj_ids=member_obj_ids)
    raise DslParseError(f"Unreachable: unknown solver call {name}")


def _norm_unary(
    name: str,
    bound: Dict[str, Any],
    ctx: _Ctx,
    *,
    scope: Scope,
    hv_absolute: bool,
) -> Dict[str, Any]:
    src = _eval_entity(bound["source"], ctx, scope=scope, member_obj_ids=None)
    if name == "against_wall":
        wall = _must_str(_eval(bound["wall"]), "wall")
        if wall not in _WALLS:
            raise DslParseError(f"Invalid wall id: {wall}")
        return {"source": src, "wall": wall}
    if name == "corner":
        corner_val = _must_str(_eval(bound["corner"]), "corner")
        wall_val = _must_str(_eval(bound["wall"]), "wall")
        if wall_val not in _WALLS:
            raise DslParseError(f"Invalid wall id in corner(): {wall_val}")
        # Validate wall is part of corner
        valid_walls = {"BL": {"B", "L"}, "BR": {"B", "R"}, "TR": {"T", "R"}, "TL": {"T", "L"}}
        if corner_val not in valid_walls:
            raise DslParseError(f"Invalid corner id: {corner_val}")
        if wall_val not in valid_walls[corner_val]:
            raise DslParseError(
                f"corner='{corner_val}' requires wall in {valid_walls[corner_val]}, got '{wall_val}'"
            )
        return {"source": src, "corner": corner_val, "wall": wall_val}
    if name == "horizontal":
        if hv_absolute:
            x, x_prior = _eval_var_or_float(bound["x"], ctx, "x", "nonneg", name)
            return {"source": src, "x": x, "x_prior": x_prior}

        perc, perc_prior = _eval_var_or_float(bound["percentile"], ctx, "percentile", "unit", name)
        return {"source": src, "percentile": perc, "percentile_prior": perc_prior}
    if name == "vertical":
        if hv_absolute:
            y, y_prior = _eval_var_or_float(bound["y"], ctx, "y", "nonneg", name)
            return {"source": src, "y": y, "y_prior": y_prior}

        perc, perc_prior = _eval_var_or_float(bound["percentile"], ctx, "percentile", "unit", name)
        return {"source": src, "percentile": perc, "percentile_prior": perc_prior}
    raise DslParseError(f"Unreachable unary: {name}")


def _norm_directional(
    name: str,
    bound: Dict[str, Any],
    ctx: _Ctx,
    *,
    scope: Scope,
    member_obj_ids: Optional[set[str]],
) -> Dict[str, Any]:
    src = _eval_entity(bound["source"], ctx, scope=scope, member_obj_ids=member_obj_ids)
    tar = _eval_entity(bound["target"], ctx, scope=scope, member_obj_ids=member_obj_ids)
    clr, clr_prior = _eval_var_or_float(bound["clearance"], ctx, "clearance", "nonneg", name)
    alignment, align_prior = _eval_var_or_alignment(bound["alignment"], ctx, name)

    return {
        "source": src, "target": tar,
        "clearance": clr, "clearance_prior": clr_prior,
        "percentile": alignment, "percentile_prior": align_prior,
    }


def _norm_around(
    bound: Dict[str, Any],
    ctx: _Ctx,
    *,
    scope: Scope,
    member_obj_ids: Optional[set[str]],
) -> Dict[str, Any]:
    src_list_node = bound["source_list"]
    if isinstance(src_list_node, (ast.List, ast.Tuple)):
        src_list = [_eval_entity(x, ctx, scope=scope, member_obj_ids=member_obj_ids) for x in src_list_node.elts]
    else:
        raise DslParseError("around(source_list=...) must be a list/tuple literal")

    tar = _eval_entity(bound["target"], ctx, scope=scope, member_obj_ids=member_obj_ids)
    dist, dist_prior = _eval_var_or_float(bound["distance"], ctx, "distance", "nonneg", "around")
    sweep, sweep_prior = _eval_var_or_float(bound["sweep_deg"], ctx, "sweep_deg", "nonneg", "around")
    cl = _must_str(_eval(bound["centerline"]), "centerline")

    return {
        "source_list": src_list, "target": tar,
        "distance": dist, "distance_prior": dist_prior,
        "sweep_deg": sweep, "sweep_deg_prior": sweep_prior,
        "centerline": cl,
    }


def _norm_facing(
    bound: Dict[str, Any],
    ctx: _Ctx,
    *,
    scope: Scope,
    member_obj_ids: Optional[set[str]],
) -> Dict[str, Any]:
    src = _eval_entity(bound["source"], ctx, scope=scope, member_obj_ids=member_obj_ids)

    target_expr = bound["target"]
    if isinstance(target_expr, ast.Constant) and isinstance(target_expr.value, str) and target_expr.value in _WALLS:
        tar_kind = "wall"
        tar_id = target_expr.value
    else:
        tar_ref = _eval_entity(target_expr, ctx, scope=scope, member_obj_ids=member_obj_ids)
        tar_kind = tar_ref.kind
        tar_id = tar_ref.id

    mutual = _must_bool(_eval(bound["mutual"]), "mutual")
    mode_raw = _eval(bound["mode"])

    # Validate mode
    if mode_raw not in {"ortho", "radial"}:
        raise DslParseError(f"facing() requires mode in {{'ortho', 'radial'}}, got '{mode_raw}'")

    if tar_kind == "wall":
        if mutual:
            raise DslParseError("facing(..., target=<WallId>) forbids mutual=True")
        if scope == "cluster_internal":
            raise DslParseError("facing(..., target=<WallId>) is forbidden inside cluster")
        # Wall targets always use ortho mode (mode parameter is ignored)
        mode = "ortho"
    else:
        mode = mode_raw

    return {"source": src, "tar_kind": tar_kind, "tar_id": tar_id, "mutual": mutual, "mode": mode}


def _norm_align(
    bound: Dict[str, Any],
    ctx: _Ctx,
    *,
    scope: Scope,
    member_obj_ids: Optional[set[str]],
) -> Tuple[str, Dict[str, Any]]:
    """Normalize align() - always becomes 'angle' constraint (align-group removed)."""
    src = _eval_entity(bound["source"], ctx, scope=scope, member_obj_ids=member_obj_ids)
    tar = _eval_entity(bound["target"], ctx, scope=scope, member_obj_ids=member_obj_ids)
    angle, angle_prior = _eval_var_or_float(bound["angle"], ctx, "angle", "angle_deg", "align")

    return "angle", {
        "source": src, "target": tar,
        "angle": angle, "angle_prior": angle_prior,
    }


# =============================================================================
# Entity resolution
# =============================================================================

def _eval(x: Any) -> Any:
    return _eval_const(x) if isinstance(x, ast.AST) else x


def _is_var_call(node: ast.AST) -> bool:
    """Check if node is a Var(...) call."""
    return (isinstance(node, ast.Call) and
            isinstance(node.func, ast.Name) and
            node.func.id == "Var")


def _parse_inline_var(
    node: ast.Call,
    ctx: _Ctx,
    kind: ParamKind,
    default_name_prefix: str,
) -> Tuple[str, float]:
    """Parse an inline Var(...) call and register it as an anonymous parameter."""
    if len(node.args) != 1 or node.keywords:
        raise DslParseError("Inline Var(...) must have exactly one positional argument")

    arg = node.args[0]
    prior: Union[float, str]

    if isinstance(arg, ast.Constant):
        if isinstance(arg.value, (int, float)) and not isinstance(arg.value, bool):
            prior = float(arg.value)
        elif isinstance(arg.value, str):
            prior = arg.value
        else:
            raise DslParseError(f"Var(...) argument must be a number or string literal, got {type(arg.value).__name__}")
    elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, (ast.UAdd, ast.USub)):
        prior = _eval_num(arg)
    else:
        raise DslParseError(f"Var(...) argument must be a literal, got {type(arg).__name__}")

    idx = len(ctx.var_decls)
    var_name = f"_anon_{default_name_prefix}_{idx}"

    decl = VarDecl(name=var_name, prior=prior)
    ctx.var_decls[var_name] = decl
    ctx.param_usages[var_name] = []
    _record_param_usage(ctx, var_name, kind)

    if isinstance(prior, str):
        prior_val = _alignment_label_to_percentile(prior)
    else:
        prior_val = prior

    return var_name, prior_val


def _parse_inline_var_alignment(
    node: ast.Call,
    ctx: _Ctx,
    constraint_name: str,
) -> Tuple[str, float]:
    """
    Parse an inline Var(...) call for alignment, validating label against constraint type.
    Returns (var_name, prior_percentile).
    """
    if len(node.args) != 1 or node.keywords:
        raise DslParseError("Inline Var(...) must have exactly one positional argument")

    arg = node.args[0]
    prior: Union[float, str]

    if isinstance(arg, ast.Constant):
        if isinstance(arg.value, (int, float)) and not isinstance(arg.value, bool):
            prior = float(arg.value)
        elif isinstance(arg.value, str):
            prior = arg.value
        else:
            raise DslParseError(f"Var(...) argument must be a number or string literal, got {type(arg.value).__name__}")
    elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, (ast.UAdd, ast.USub)):
        prior = _eval_num(arg)
    else:
        raise DslParseError(f"Var(...) argument must be a literal, got {type(arg).__name__}")

    idx = len(ctx.var_decls)
    var_name = f"_anon_alignment_{idx}"

    decl = VarDecl(name=var_name, prior=prior)
    ctx.var_decls[var_name] = decl
    ctx.param_usages[var_name] = []
    _record_param_usage(ctx, var_name, "unit")

    if isinstance(prior, str):
        # Validate label against constraint type
        prior_val = _alignment_label_to_percentile_typed(prior, constraint_name)
    else:
        prior_val = prior

    return var_name, prior_val


def _eval_var_or_float(
    node: Any,
    ctx: _Ctx,
    field_name: str,
    kind: ParamKind,
    constraint_name: str,
) -> Tuple[VarRef, float]:
    """
    Evaluate a field that MUST be a Var reference.
    Returns (VarRef, prior_value).
    """
    if isinstance(node, ast.Name):
        var_name = node.id
        if var_name in ctx.var_decls:
            decl = ctx.var_decls[var_name]
            _record_param_usage(ctx, var_name, kind)
            if isinstance(decl.prior, str):
                prior_val = _alignment_label_to_percentile(decl.prior)
            else:
                prior_val = decl.prior
            return VarRef(name=var_name), prior_val

    if isinstance(node, ast.Call) and _is_var_call(node):
        var_name, prior_val = _parse_inline_var(node, ctx, kind, field_name)
        return VarRef(name=var_name), prior_val

    # Constants not allowed
    if isinstance(node, (ast.Constant, ast.UnaryOp)):
        raise DslParseError(
            f"solver.{constraint_name}: '{field_name}' must be a Var reference. "
            f"Example: `{field_name}_var = Var(...)`, then use `{field_name}={field_name}_var`"
        )

    raise DslParseError(
        f"solver.{constraint_name}: '{field_name}' must be a Var reference. "
        f"Got unexpected expression type: {type(node).__name__}"
    )


def _alignment_label_to_percentile_typed(label: str, constraint_name: str) -> float:
    """Convert alignment label to percentile, validating against constraint type."""
    table = _LATERAL_ALIGN if constraint_name in _LATERAL_TYPES else _FRONTAL_ALIGN
    if label not in table:
        valid = ", ".join(f'"{k}"' for k in table)
        raise DslParseError(
            f"solver.{constraint_name}: invalid alignment label '{label}'. "
            f"Valid options for {constraint_name}: {valid}"
        )
    return table[label]


def _eval_var_or_alignment(
    node: Any,
    ctx: _Ctx,
    constraint_name: str,
) -> Tuple[VarRef, float]:
    """
    Evaluate an alignment field that MUST be a Var reference.
    Returns (VarRef, prior_percentile).
    """
    if isinstance(node, ast.Name):
        var_name = node.id
        if var_name in ctx.var_decls:
            decl = ctx.var_decls[var_name]
            _record_param_usage(ctx, var_name, "unit")
            if isinstance(decl.prior, str):
                # Validate label against constraint type
                prior_val = _alignment_label_to_percentile_typed(decl.prior, constraint_name)
            else:
                prior_val = decl.prior
            return VarRef(name=var_name), prior_val

    if isinstance(node, ast.Call) and _is_var_call(node):
        var_name, prior_val = _parse_inline_var_alignment(node, ctx, constraint_name)
        return VarRef(name=var_name), prior_val

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        label = node.value
        valid_labels = list(_LATERAL_ALIGN.keys()) if constraint_name in _LATERAL_TYPES else list(_FRONTAL_ALIGN.keys())
        raise DslParseError(
            f"solver.{constraint_name}: 'alignment' must be a Var reference. "
            f"Example: `align_var = Var(\"{label}\")`, then use `alignment=align_var`. "
            f"Valid label priors: {valid_labels}"
        )

    raise DslParseError(
        f"solver.{constraint_name}: 'alignment' must be a Var reference. "
        f"Got unexpected expression type: {type(node).__name__}"
    )


def _resolve_asset_var(name: str, var_to_obj_id: Dict[str, str], *, what: str) -> str:
    if name in var_to_obj_id:
        return var_to_obj_id[name]
    raise DslParseError(f"Unknown asset var '{name}' ({what}); not in provided asset list")


def _eval_entity(node: Any, ctx: _Ctx, *, scope: Scope, member_obj_ids: Optional[set[str]]) -> EntityRef:
    if not isinstance(node, ast.AST):
        raise DslParseError(f"Expected an entity expression, got {type(node).__name__}")

    name = _must_name(node, "entity")
    if scope == "global":
        return _resolve_global_entity(name, ctx)
    return _resolve_internal_entity(name, ctx.var_to_obj_id, member_obj_ids)


def _resolve_global_entity(name: str, ctx: _Ctx) -> EntityRef:
    if name in ctx.handle_to_cluster:
        return EntityRef(kind="cluster", id=ctx.handle_to_cluster[name])
    if name in ctx.var_to_obj_id:
        oid = ctx.var_to_obj_id[name]
        if oid in ctx.cluster_member_obj_ids:
            raise DslParseError(f"GLOBAL scope forbids referencing cluster member '{oid}'. Use the cluster handle instead.")
        return EntityRef(kind="object", id=oid)
    raise DslParseError(f"Unknown entity name '{name}' in GLOBAL scope")


def _resolve_internal_entity(name: str, var_to_obj_id: Dict[str, str], member_obj_ids: Optional[set[str]]) -> EntityRef:
    if member_obj_ids is None:
        raise DslParseError("Internal entity resolution requires member list")
    oid = _resolve_asset_var(name, var_to_obj_id, what="cluster member")
    if oid not in member_obj_ids:
        raise DslParseError(f"Cluster-internal scope forbids referencing non-member object '{oid}'")
    return EntityRef(kind="object", id=oid)
