"""
parse_cogmap.py - Parse cognitive map (Pose/AABB) assignments from v2 DSL.

Produces a cogmap_dict with all numeric values evaluated (no expressions).
Safe AST-based evaluation: only allows +, -, *, /, unary +/-, and room.{length,width}.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from .parse_constraints import ProgramIR


class CogMapParseError(RuntimeError):
    """Fail-fast error for invalid cognitive map assignments."""


# =============================================================================
# Data structures
# =============================================================================

@dataclass(slots=True, frozen=True)
class PoseIR:
    x: float
    y: float
    rz: float  # degrees


@dataclass(slots=True, frozen=True)
class AABBIR:
    lx: float
    ly: float


@dataclass(slots=True)
class CogMapIR:
    """Parsed cognitive map."""
    room_length: float
    room_width: float
    independent_poses: Dict[str, PoseIR]  # object_id -> global pose
    cluster_member_poses: Dict[str, Dict[str, PoseIR]]  # cluster_id -> {obj_id -> local pose}
    cluster_aabbs: Dict[str, AABBIR]  # cluster_id -> AABB
    cluster_poses: Dict[str, PoseIR]  # cluster_id -> global pose (handle)


# =============================================================================
# Safe expression evaluator (whitelist AST nodes)
# =============================================================================

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _safe_eval(node: ast.AST, room_length: float, room_width: float) -> float:
    """
    Evaluate AST node to float using whitelist approach.
    
    Allowed:
    - Numeric literals (int, float)
    - Unary +/-
    - Binary +, -, *, /
    - Names: length, width
    - Attributes: room.length, room.width
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise CogMapParseError("Boolean not allowed in Pose/AABB expression")
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise CogMapParseError(f"Invalid constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand, room_length, room_width)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise CogMapParseError(f"Unsupported unary op: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise CogMapParseError(f"Unsupported binary op: {type(node.op).__name__}")
        left = _safe_eval(node.left, room_length, room_width)
        right = _safe_eval(node.right, room_length, room_width)
        return op_fn(left, right)

    if isinstance(node, ast.Name):
        if node.id == "length":
            return room_length
        if node.id == "width":
            return room_width
        raise CogMapParseError(f"Unsupported variable '{node.id}' in Pose/AABB. Use numeric literals or 'length'/'width'.")

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "room":
            if node.attr == "length":
                return room_length
            if node.attr == "width":
                return room_width
            raise CogMapParseError(f"Invalid room attribute: room.{node.attr}")
        raise CogMapParseError(f"Unsupported attribute access in Pose/AABB expression")

    raise CogMapParseError(f"Unsupported AST node in Pose/AABB: {type(node).__name__}")


# =============================================================================
# Pose/AABB call parsing
# =============================================================================

def _parse_pose_call(call: ast.Call, room_length: float, room_width: float) -> PoseIR:
    """Parse Pose(x=..., y=..., rz=...) call. Caller must verify it's a Pose call."""
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}

    if call.args:
        raise CogMapParseError("Pose(...) must use keyword arguments only")
    if set(kwargs.keys()) != {"x", "y", "rz"}:
        raise CogMapParseError(f"Pose(...) requires exactly x, y, rz. Got: {list(kwargs.keys())}")

    return PoseIR(
        x=_safe_eval(kwargs["x"], room_length, room_width),
        y=_safe_eval(kwargs["y"], room_length, room_width),
        rz=_safe_eval(kwargs["rz"], room_length, room_width),
    )


def _parse_aabb_call(call: ast.Call, room_length: float, room_width: float) -> AABBIR:
    """Parse AABB(lx=..., ly=...) call. Caller must verify it's an AABB call."""
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}

    if call.args:
        raise CogMapParseError("AABB(...) must use keyword arguments only")
    if set(kwargs.keys()) != {"lx", "ly"}:
        raise CogMapParseError(f"AABB(...) requires exactly lx, ly. Got: {list(kwargs.keys())}")

    return AABBIR(
        lx=_safe_eval(kwargs["lx"], room_length, room_width),
        ly=_safe_eval(kwargs["ly"], room_length, room_width),
    )


# =============================================================================
# Room dimensions extraction
# =============================================================================

def _extract_room_dims(mod: ast.Module) -> Tuple[float, float]:
    """Extract room = Room(length=..., width=...) from module."""
    for stmt in mod.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        if not isinstance(stmt.targets[0], ast.Name):
            continue
        if stmt.targets[0].id != "room":
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "Room"):
            continue

        length = width = None

        for kw in call.keywords:
            if kw.arg == "length":
                length = _safe_eval(kw.value, 0.0, 0.0)
            elif kw.arg == "width":
                width = _safe_eval(kw.value, 0.0, 0.0)

        if length is None or width is None:
            raise CogMapParseError("Room(...) must specify both length and width")

        return length, width

    raise CogMapParseError("Missing 'room = Room(length=..., width=...)' declaration")


# =============================================================================
# Assignment iteration and parsing
# =============================================================================

def _iter_assignments(mod: ast.Module) -> Iterator[ast.Assign]:
    """Yield Assign statements from top-level and inside With blocks."""
    for stmt in mod.body:
        if isinstance(stmt, ast.Assign):
            yield stmt
        elif isinstance(stmt, ast.With):
            for inner in stmt.body:
                if isinstance(inner, ast.Assign):
                    yield inner


def _get_attr_assign(stmt: ast.Assign, attr: str) -> Optional[ast.Name]:
    """
    If stmt is `<name>.<attr> = ...`, return the Name node for <name>.
    Otherwise return None.
    """
    if len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Attribute) or target.attr != attr:
        return None
    if not isinstance(target.value, ast.Name):
        return None
    return target.value


# =============================================================================
# Scope tracking for cluster-aware parsing
# =============================================================================

@dataclass(slots=True)
class _ScopeTracker:
    """Track which statements are inside which cluster block."""
    cluster_ranges: Dict[str, Tuple[int, int]]  # cluster_id -> (start_line, end_line)
    handle_to_cluster: Dict[str, str]  # handle_name -> cluster_id


def _build_scope_tracker(mod: ast.Module, program: ProgramIR) -> _ScopeTracker:
    """Build scope tracker from parsed program IR and AST."""
    cluster_ranges: Dict[str, Tuple[int, int]] = {}
    handle_to_cluster = dict(program.handle_to_cluster)

    for stmt in mod.body:
        if not isinstance(stmt, ast.With):
            continue
        if len(stmt.items) != 1:
            continue
        item = stmt.items[0]
        if item.optional_vars is None:
            continue
        if not isinstance(item.optional_vars, ast.Name):
            continue
        handle_name = item.optional_vars.id
        if handle_name not in handle_to_cluster:
            continue

        cluster_id = handle_to_cluster[handle_name]
        start = stmt.lineno
        end = stmt.end_lineno or stmt.lineno
        cluster_ranges[cluster_id] = (start, end)

    return _ScopeTracker(cluster_ranges=cluster_ranges, handle_to_cluster=handle_to_cluster)


def _stmt_in_cluster(stmt: ast.stmt, tracker: _ScopeTracker) -> Optional[str]:
    """Return cluster_id if stmt is inside a cluster block, else None."""
    line = stmt.lineno
    for cluster_id, (start, end) in tracker.cluster_ranges.items():
        if start <= line <= end:
            return cluster_id
    return None


# =============================================================================
# Main parsing
# =============================================================================

def _parse_cogmap_ir(
    mod: ast.Module,
    program: ProgramIR,
    room_length: float,
    room_width: float,
) -> CogMapIR:
    """Parse cognitive map assignments from AST module into IR."""
    tracker = _build_scope_tracker(mod, program)

    # Build lookup tables
    obj_var_to_id = _invert_var_map(program)
    cluster_members = {c.cluster_id: set(c.member_obj_ids) for c in program.clusters}
    member_to_cluster = {}
    for c in program.clusters:
        for m in c.member_obj_ids:
            member_to_cluster[m] = c.cluster_id

    ir = CogMapIR(
        room_length=room_length,
        room_width=room_width,
        independent_poses={},
        cluster_member_poses={cid: {} for cid in cluster_members},
        cluster_aabbs={},
        cluster_poses={},
    )

    for stmt in _iter_assignments(mod):
        if _try_handle_pose(stmt, ir, tracker, obj_var_to_id, member_to_cluster,
                           room_length, room_width):
            continue
        _try_handle_aabb(stmt, ir, tracker, room_length, room_width)

    return ir


def _invert_var_map(program: ProgramIR) -> Dict[str, str]:
    """Create var_name -> object_id mapping by inverting the convention."""
    result = {}
    for oid in program.object_ids:
        var_name = oid.replace("-", "_")
        result[var_name] = oid
    return result


def _try_handle_pose(
    stmt: ast.Assign,
    ir: CogMapIR,
    tracker: _ScopeTracker,
    obj_var_to_id: Dict[str, str],
    member_to_cluster: Dict[str, str],
    room_length: float,
    room_width: float,
) -> bool:
    """
    Handle `<var>.pose = Pose(x=..., y=..., rz=...)` assignment.

    Dispatches to the right storage based on what <var> refers to:
    - Cluster handle -> ir.cluster_poses (must be global scope)
    - Cluster member -> ir.cluster_member_poses (must be inside owning cluster)
    - Independent object -> ir.independent_poses (must be global scope)

    Returns True if this was a pose assignment, False otherwise.
    """
    name_node = _get_attr_assign(stmt, "pose")
    if name_node is None:
        return False
    if not isinstance(stmt.value, ast.Call):
        return False
    if not (isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "Pose"):
        return False

    var_name = name_node.id
    pose = _parse_pose_call(stmt.value, room_length, room_width)
    cluster_scope = _stmt_in_cluster(stmt, tracker)

    # Cluster handle pose (global scope only)
    if var_name in tracker.handle_to_cluster:
        cluster_id = tracker.handle_to_cluster[var_name]
        if cluster_scope is not None:
            raise CogMapParseError(
                f"Cluster handle '{var_name}' pose must be assigned in GLOBAL scope, "
                f"not inside cluster '{cluster_scope}'"
            )
        ir.cluster_poses[cluster_id] = pose
        return True

    if var_name not in obj_var_to_id:
        raise CogMapParseError(f"Unknown variable '{var_name}' in pose assignment")

    obj_id = obj_var_to_id[var_name]

    # Cluster member pose (inside owning cluster only)
    if obj_id in member_to_cluster:
        owning_cluster = member_to_cluster[obj_id]
        if cluster_scope is None:
            raise CogMapParseError(
                f"Cluster member '{var_name}' pose must be assigned INSIDE its cluster block, "
                f"not in global scope"
            )
        if cluster_scope != owning_cluster:
            raise CogMapParseError(
                f"Cluster member '{var_name}' belongs to cluster '{owning_cluster}', "
                f"but pose is assigned inside cluster '{cluster_scope}'"
            )
        ir.cluster_member_poses[owning_cluster][obj_id] = pose
        return True

    # Independent object pose (global scope only)
    if cluster_scope is not None:
        raise CogMapParseError(
            f"Independent object '{var_name}' pose must be assigned in GLOBAL scope, "
            f"not inside cluster '{cluster_scope}'"
        )
    ir.independent_poses[obj_id] = pose
    return True


def _try_handle_aabb(
    stmt: ast.Assign,
    ir: CogMapIR,
    tracker: _ScopeTracker,
    room_length: float,
    room_width: float,
) -> bool:
    """
    Handle `<cluster_handle>.aabb = AABB(lx=..., ly=...)` assignment.

    AABB defines the bounding box for a cluster. Only valid on cluster handles,
    and must be assigned in global scope (not inside a cluster block).

    Returns True if this was an AABB assignment, False otherwise.
    """
    name_node = _get_attr_assign(stmt, "aabb")
    if name_node is None:
        return False
    if not isinstance(stmt.value, ast.Call):
        return False
    if not (isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "AABB"):
        return False

    var_name = name_node.id
    if var_name not in tracker.handle_to_cluster:
        raise CogMapParseError(
            f"AABB can only be assigned to cluster handles. '{var_name}' is not a cluster handle."
        )

    ir.cluster_aabbs[tracker.handle_to_cluster[var_name]] = _parse_aabb_call(
        stmt.value, room_length, room_width
    )
    return True


# =============================================================================
# Coverage validation
# =============================================================================

def _validate_coverage(ir: CogMapIR, program: ProgramIR) -> None:
    """Validate that all required poses/AABBs are present."""
    errors: List[str] = []

    # Build member set
    cluster_members = {c.cluster_id: set(c.member_obj_ids) for c in program.clusters}
    all_members = set()
    for members in cluster_members.values():
        all_members.update(members)

    # Check independent objects
    independent = [oid for oid in program.object_ids if oid not in all_members]
    for oid in independent:
        if oid not in ir.independent_poses:
            var_name = oid.replace("-", "_")
            errors.append(f"Missing pose for independent object '{var_name}'")

    # Check clusters
    for cluster in program.clusters:
        cid = cluster.cluster_id
        handle_name = cluster.handle_name

        # Check AABB
        if cid not in ir.cluster_aabbs:
            errors.append(f"Missing AABB for cluster handle '{handle_name}'")

        # Check cluster handle pose
        if cid not in ir.cluster_poses:
            errors.append(f"Missing pose for cluster handle '{handle_name}'")

        # Check member poses
        for member_id in cluster.member_obj_ids:
            if member_id not in ir.cluster_member_poses.get(cid, {}):
                var_name = member_id.replace("-", "_")
                errors.append(f"Missing local pose for cluster member '{var_name}' in cluster '{cid}'")

    if errors:
        raise CogMapParseError("Cognitive map coverage errors:\n  - " + "\n  - ".join(errors))


# =============================================================================
# JSON output
# =============================================================================

def _pose_to_dict(pose: PoseIR) -> Dict[str, float]:
    return {"x": pose.x, "y": pose.y, "rz": pose.rz}


def _aabb_to_dict(aabb: AABBIR) -> Dict[str, float]:
    return {"lx": aabb.lx, "ly": aabb.ly}


def cogmap_to_json(ir: CogMapIR, program: ProgramIR) -> dict:
    """Convert CogMapIR to JSON dict (schema_minimal)."""
    out: dict = {
        "room": {
            "length": ir.room_length,
            "width": ir.room_width,
        },
        "independent": {
            oid: _pose_to_dict(pose)
            for oid, pose in ir.independent_poses.items()
        },
    }

    # Only include clusters if there are any
    if program.clusters:
        clusters_dict = {}
        for cluster in program.clusters:
            cid = cluster.cluster_id
            clusters_dict[cid] = {
                "aabb": _aabb_to_dict(ir.cluster_aabbs[cid]),
                "pose": _pose_to_dict(ir.cluster_poses[cid]),
                "members": {
                    oid: _pose_to_dict(pose)
                    for oid, pose in ir.cluster_member_poses[cid].items()
                },
            }
        out["clusters"] = clusters_dict

    return out


# =============================================================================
# Public API
# =============================================================================

def parse_cogmap(code: str, program: ProgramIR) -> dict:
    """
    Parse cognitive map from v2 DSL code.
    
    Args:
        code: The DSL program source code
        program: Parsed ProgramIR from parse_program()
    
    Returns:
        cogmap_dict in schema_minimal format:
        {
            "room": {"length": float, "width": float},
            "independent": {object_id: {"x": float, "y": float, "rz": float}},
            "clusters": {cluster_id: {  # omitted if no clusters
                "aabb": {"lx": float, "ly": float},
                "pose": {"x": float, "y": float, "rz": float},
                "members": {object_id: {"x": float, "y": float, "rz": float}},
            }},
        }
    
    Raises:
        CogMapParseError: On parse failure or coverage validation failure
    """
    mod = ast.parse(code)
    room_length, room_width = _extract_room_dims(mod)

    ir = _parse_cogmap_ir(mod, program, room_length, room_width)
    _validate_coverage(ir, program)

    return cogmap_to_json(ir, program)
