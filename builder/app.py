"""
Scene Builder Web App

Minimal Flask server for interactively authoring benchmark scene files.
No fancy abstractions. Just HTTP endpoints that do the work.
"""
import os
import sys
import json
import re
import base64
import socket
import requests
import compress_json
import subprocess
import tempfile
import uuid
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, send_file
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import SecretStr
from planners.holodeck_v2 import create_clip_models, create_sbert_model
from retrievers.holodeck_v2.factory import create_holodeck_retriever
from utils.asset import get_annotations
from utils.holodeck_v2.constants import BASE_URL
from builder.prompts import TASK_PROMPT, REGENERATE_PROMPT

# Compute absolute path to assets directory (don't trust relative paths)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "assets")

app = Flask(__name__)

# The address the Scene Builder serves on, shared with cli/builder.py's launch screen so
# the printed URL always matches the bound address (single source of truth).
HOST = "127.0.0.1"
# Port search range (inclusive start, exclusive end). 5000 is the Flask/Python
# ecosystem default; on macOS it's held by the AirPlay Receiver (ControlCenter),
# so find_available_port() walks up to the first free port. Stays below the
# ephemeral range (49152+) so the OS won't hand our chosen port to an outgoing
# connection. cli/builder.py resolves the actual port BEFORE printing the URL,
# so the printed URL always matches the bound address.
PORT_RANGE_START = 5000
PORT_RANGE_END = 5100  # exclusive → 5000..5099

# The LLM the Scene Builder uses for room generation and instruction regeneration.
# cli/preflight.py (and refactor_log/sim_cli.py) mirror this value in the preflight panel (BUILDER_MODEL).
BUILDER_LLM_MODEL = "gpt-5"

# Retriever singleton - loads once, then fast
_retriever = None

# Render job tracking
# {job_id: {"status": "running"|"complete"|"failed", "output_dir": path, "error": str|None}}
_render_jobs = {}
_render_lock = threading.Lock()

# Asset conversion job tracking (msgpack.gz → GLB)
# {asset_id: {"status": "converting"|"complete"|"failed", "error": str|None}}
_conversion_jobs = {}
_conversion_lock = threading.Lock()


def get_retriever():
    """Lazy-load the retriever. CLIP + SBERT take ~10s to load."""
    global _retriever
    if _retriever is None:
        print("Loading CLIP and SBERT models (this takes ~10s)...")
        clip_models = create_clip_models()
        sbert_model = create_sbert_model()
        _retriever = create_holodeck_retriever(clip_models, sbert_model)
        print("Models loaded.")
    return _retriever


# ============================================================================
# Asset Download Functions (copied from utils/asset.py, no bpy dependency)
# ============================================================================

def download_asset(asset_id: str) -> bool:
    """
    Download asset files (preview images + annotations).
    Returns True if asset exists or download succeeded.
    """
    asset_dir = os.path.join(ASSET_BASE_DIR, asset_id)
    annotations_path = os.path.join(asset_dir, "annotations.json")

    # Already downloaded
    if os.path.exists(annotations_path):
        return True

    os.makedirs(ASSET_BASE_DIR, exist_ok=True)

    tar_path = os.path.join(ASSET_BASE_DIR, f"{asset_id}.tar")
    url = f"{BASE_URL}/assets/{asset_id}.tar"

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/html"):
            return False

        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        subprocess.run(["tar", "-xf", tar_path, "-C", ASSET_BASE_DIR],
                       capture_output=True, text=True, check=True)
        os.remove(tar_path)

        # Merge annotations
        _reformat_annotations(asset_id)
        return True

    except Exception:
        return False


def _reformat_annotations(asset_id: str):
    """Merge compressed annotations with thor_metadata."""
    asset_dir = os.path.join(ASSET_BASE_DIR, asset_id)

    ann_gz = os.path.join(asset_dir, "annotations.json.gz")
    thor_path = os.path.join(asset_dir, "thor_metadata.json")

    if not os.path.exists(ann_gz):
        return

    annotations = compress_json.load(ann_gz)

    if os.path.exists(thor_path):
        with open(thor_path, "r") as f:
            thor_metadata = json.load(f)
        annotations["thor_metadata"] = thor_metadata

    with open(os.path.join(asset_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)

    # Cleanup
    if os.path.exists(ann_gz):
        os.remove(ann_gz)
    if os.path.exists(thor_path):
        os.remove(thor_path)


def get_asset_metadata(asset_id: str) -> dict:
    """Extract bbox and front_view from annotations.json (single file read)."""
    annotations_path = os.path.join(ASSET_BASE_DIR, asset_id, "annotations.json")

    fallback = {
        "bbox": {"x": 0.5, "y": 0.5, "z": 0.5},
        "front_view": 0,
    }

    if not os.path.exists(annotations_path):
        return fallback

    with open(annotations_path, "r") as f:
        ann = json.load(f)

    result = {
        "front_view": ann.get("frontView", 0),
        "name": ann.get("category", asset_id),
    }

    bb = ann.get("thor_metadata", {}).get("assetMetadata", {}).get("boundingBox", {})
    if bb:
        result["bbox"] = {
            "x": bb["max"]["x"] - bb["min"]["x"],
            "y": bb["max"]["y"] - bb["min"]["y"],
            "z": bb["max"]["z"] - bb["min"]["z"],
        }
    else:
        result["bbox"] = fallback["bbox"]

    return result


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/init-room", methods=["POST"])
def init_room():
    """
    Generate room layout via LLM, retrieve initial assets.

    Input: {"room_type": "bedroom", "difficulty": "medium", "room_width": 5.0, "room_height": 4.0, "additional_requirements": "..."}
    Output: {prompt, difficulty, objects, asset_palette[]}
    """
    data = request.json
    room_type = data["room_type"]
    room_width = float(data["room_width"])
    room_height = float(data["room_height"])
    difficulty = data.get("difficulty", "medium")
    additional_requirements = data.get("additional_requirements", "").strip()

    # Generate objects via LLM
    llm = ChatOpenAI(
        model=BUILDER_LLM_MODEL,
        temperature=1.0,
        max_completion_tokens=16000,  # gpt-5 is a reasoning model; needs tokens for thinking + output
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    )

    # TASK_PROMPT is the paper's verbatim template; supply the room and difficulty it asks for.
    prompt = (
        f"{TASK_PROMPT}\n\n"
        f"Room type: {room_type}\n"
        f"Room size: {room_height} m x {room_width} m\n"
        f"Difficulty level: {difficulty}"
    )
    if additional_requirements:
        prompt += f"\nAdditional requirements: {additional_requirements}"

    raw_response = llm.invoke([HumanMessage(content=prompt)]).content
    response_text = str(raw_response) if raw_response else ""

    # Parse JSON (handle ```json``` wrapper)
    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)

    parsed = json.loads(response_text)
    instruction = parsed.get("instruction", "")
    objects = parsed.get("objects", {})

    # Retrieve best asset for each object
    retriever = get_retriever()
    database = retriever.database
    asset_palette = []

    for obj_name, obj_info in objects.items():
        obj_desc = obj_info.get("description", obj_name)
        obj_size = obj_info.get("size")

        candidates = retriever.retrieve_with_name_and_desc(
            object_names=[obj_name],
            object_descriptions=[obj_desc],
            threshold=31,
        )

        # Filter: floor objects only
        candidates = [
            c for c in candidates
            if _is_floor_object(database.get(c[0], {}))
        ]

        # Size ranking
        if obj_size and candidates:
            candidates = retriever.compute_size_difference(obj_size, candidates)

        if not candidates:
            continue

        # Download top asset
        asset_id = candidates[0][0]
        if download_asset(asset_id):
            asset_info = database.get(asset_id, {})
            metadata = get_asset_metadata(asset_id)
            asset_palette.append({
                "asset_id": asset_id,
                "object_name": obj_name,
                "name": asset_info.get("category", obj_name),
                "description": asset_info.get("description", ""),
                "bbox": metadata["bbox"],
                "front_view": metadata["front_view"],
                "preview_url": f"/api/asset-preview/{asset_id}",
            })

    return jsonify({
        "prompt": instruction,
        "room_type": room_type,
        "room_size": [room_width, room_height],
        "difficulty": difficulty,
        "objects": objects,
        "asset_palette": asset_palette,
    })


@app.route("/api/regenerate-prompt", methods=["POST"])
def regenerate_prompt():
    """
    Generate new layout prompt based on current canvas objects.

    Input: {room_type, room_size, objects: [{name, count}, ...]}
    Output: {prompt}
    """
    data = request.json
    room_type = data["room_type"]
    room_width, room_height = data["room_size"]
    objects = data.get("objects", [])

    # Build object list
    obj_lines = [f"- {obj['name']} ({obj['count']})" for obj in objects]
    object_list = "\n".join(obj_lines) if obj_lines else "(empty room)"

    prompt = (
        REGENERATE_PROMPT
        .replace("ROOM_TYPE", room_type)
        .replace("ROOM_WIDTH", str(room_width))
        .replace("ROOM_LENGTH", str(room_height))
        .replace("OBJECT_LIST", object_list)
    )

    llm = ChatOpenAI(
        model=BUILDER_LLM_MODEL,
        temperature=0.7,
        max_completion_tokens=16000,  # gpt-5 is a reasoning model; needs tokens for thinking + output
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return jsonify({"prompt": str(response.content).strip()})


def _is_floor_object(db_entry: dict) -> bool:
    """Check if asset is a floor object (not ceiling, not door/window)."""
    try:
        ann = get_annotations(db_entry)
        if not ann.get("onFloor", False):
            return False
        if ann.get("onCeiling", False):
            return False
        category = ann.get("category", "").lower()
        if any(k in category for k in ["door", "window", "frame"]):
            return False
        return True
    except:
        return True  # Assume valid if we can't check


@app.route("/api/search-assets", methods=["POST"])
def search_assets():
    """
    Search for assets by keyword or asset ID.

    Input: {"query": "wooden chair", "offset": 0, "limit": 8}
           OR {"query": "3229a8c480954131940ff791f6af4cf7", ...}  # exact ID
    Output: {assets[], has_more}
    """
    data = request.json
    query = data["query"].strip()
    offset = int(data.get("offset", 0))
    limit = int(data.get("limit", 8))

    retriever = get_retriever()
    database = retriever.database

    # Direct ID lookup: if query matches an asset_id exactly
    if query in database:
        asset_id = query
        if _is_floor_object(database.get(asset_id, {})):
            if offset > 0:
                # Pagination beyond single result
                return jsonify({"assets": [], "has_more": False})
            if download_asset(asset_id):
                asset_info = database.get(asset_id, {})
                metadata = get_asset_metadata(asset_id)
                return jsonify({
                    "assets": [{
                        "asset_id": asset_id,
                        "name": asset_info.get("category", "Unknown"),
                        "description": asset_info.get("description", ""),
                        "score": 100.0,
                        "bbox": metadata["bbox"],
                        "front_view": metadata["front_view"],
                        "preview_url": f"/api/asset-preview/{asset_id}",
                    }],
                    "has_more": False,
                })
        # Asset exists but isn't a floor object, or download failed - fall through to semantic search

    # Semantic search
    candidates = retriever.retrieve_with_name_and_desc(
        object_names=[query],
        object_descriptions=[query],
        threshold=28,
    )

    # Filter floor objects
    candidates = [c for c in candidates if _is_floor_object(database.get(c[0], {}))]

    # Paginate
    page = candidates[offset:offset + limit]
    has_more = len(candidates) > offset + limit

    assets = []
    for asset_id, score in page:
        if download_asset(asset_id):
            asset_info = database.get(asset_id, {})
            metadata = get_asset_metadata(asset_id)
            assets.append({
                "asset_id": asset_id,
                "name": asset_info.get("category", "Unknown"),
                "description": asset_info.get("description", ""),
                "score": round(score, 1),
                "bbox": metadata["bbox"],
                "front_view": metadata["front_view"],
                "preview_url": f"/api/asset-preview/{asset_id}",
            })

    return jsonify({"assets": assets, "has_more": has_more})


@app.route("/api/asset-preview/<asset_id>")
def asset_preview(asset_id: str):
    """Serve preview image for an asset. Use ?view=N to select angle (0-3)."""
    asset_dir = os.path.join(ASSET_BASE_DIR, asset_id)
    render_dir = os.path.join(asset_dir, "blender_renders")
    annotations_path = os.path.join(asset_dir, "annotations.json")

    # Determine view angle: ?view=N overrides default frontView
    view_angle = 0.0
    if os.path.exists(annotations_path):
        with open(annotations_path, "r") as f:
            ann = json.load(f)
        view_param = request.args.get("view")
        if view_param is not None:
            view_idx = int(view_param) % 4
        else:
            view_idx = ann.get("frontView", 0)
        view_angle = view_idx * 90.0

    # Try requested view first
    render_path = os.path.join(render_dir, f"render_{view_angle}.jpg")
    if os.path.exists(render_path):
        return send_file(render_path, mimetype="image/jpeg")

    # Fallback: any available render
    if os.path.exists(render_dir):
        for f in sorted(os.listdir(render_dir)):
            if f.endswith(".jpg"):
                return send_file(os.path.join(render_dir, f), mimetype="image/jpeg")

    return "", 404


@app.route("/api/export", methods=["POST"])
def export_benchmark():
    """
    Export the current room + canvas as a benchmark scene JSON.

    Input: {prompt, objects, room_type, difficulty, room_size, asset_palette[]}
    Output: {prompt, room_type, difficulty, room_size, boundary, assets}
    """
    data = request.json
    room_width, room_height = float(data["room_size"][0]), float(data["room_size"][1])

    # Build boundary (rectangular room)
    boundary = {
        "floor_vertices": [
            [0.0, 0.0, 0.0],
            [0.0, room_height, 0.0],
            [room_width, room_height, 0.0],
            [room_width, 0.0, 0.0],
        ],
        "wall_height": 2.5,
    }

    # Build assets dict with instance IDs
    assets = {}
    asset_counts = {}

    for item in data.get("asset_palette", []):
        asset_id = item["asset_id"]
        count = asset_counts.get(asset_id, 0)
        instance_id = f"{asset_id}-{count}"
        assets[instance_id] = {}
        asset_counts[asset_id] = count + 1

    result = {
        "prompt": data.get("prompt", ""),
        "room_type": data.get("room_type", "room"),
        "difficulty": data.get("difficulty", "medium"),
        "room_size": [room_width, room_height],
        "boundary": boundary,
        "assets": assets,
        "_canvas": data.get("_canvas", ""),
    }

    return jsonify(result)


@app.route("/api/import", methods=["POST"])
def import_benchmark():
    """
    Import a previously exported benchmark JSON.

    Validates assets exist (downloads if needed), decodes canvas placements.
    Returns data needed to restore full UI state.
    """
    data = request.json

    # Validate required fields
    if not data.get("_canvas") or not data.get("room_size") or not data.get("room_type"):
        return jsonify({"error": "Invalid benchmark file: missing required fields"}), 400

    # Decode canvas blob
    try:
        canvas_items = json.loads(base64.b64decode(data["_canvas"]))
    except Exception as e:
        return jsonify({"error": f"Corrupt canvas data: {e}"}), 400

    # Collect unique asset IDs
    asset_ids = set(item["assetId"] for item in canvas_items)

    # Validate/download each asset
    assets_metadata = {}
    for asset_id in asset_ids:
        if not download_asset(asset_id):
            return jsonify({"error": f"Asset not found in database: {asset_id}"}), 400
        assets_metadata[asset_id] = get_asset_metadata(asset_id)

    return jsonify({
        "room_type": data["room_type"],
        "room_size": data["room_size"],
        "difficulty": data.get("difficulty", "medium"),
        "prompt": data.get("prompt", ""),
        "canvas_items": canvas_items,
        "assets": assets_metadata,
    })


# ============================================================================
# Asset Conversion Endpoints (msgpack.gz → GLB)
# ============================================================================

@app.route("/api/ensure-glb", methods=["POST"])
def ensure_glb():
    """
    Ensure GLB file exists for an asset. Starts background conversion if needed.

    Input: {"asset_id": "xxx"}
    Output: {"status": "exists"|"converting"|"complete"|"failed", "error": str|None}
    """
    data = request.json
    asset_id = data["asset_id"]

    asset_dir = os.path.join(ASSET_BASE_DIR, asset_id)
    glb_path = os.path.join(asset_dir, f"{asset_id}.glb")
    msgpack_path = os.path.join(asset_dir, f"{asset_id}.msgpack.gz")

    # Already exists - no conversion needed
    if os.path.exists(glb_path):
        return jsonify({"status": "exists"})

    # Check if conversion already in progress or completed
    with _conversion_lock:
        if asset_id in _conversion_jobs:
            job = _conversion_jobs[asset_id]
            # If previously completed, check if GLB now exists
            if job["status"] == "complete" and os.path.exists(glb_path):
                return jsonify({"status": "complete"})
            return jsonify({"status": job["status"], "error": job["error"]})

    # Check if msgpack exists (required for conversion)
    if not os.path.exists(msgpack_path):
        return jsonify({
            "status": "failed",
            "error": f"No msgpack.gz file found. Asset may need to be re-downloaded."
        })

    # Register conversion job
    with _conversion_lock:
        _conversion_jobs[asset_id] = {"status": "converting", "error": None}

    # Path to converter script
    converter_path = os.path.join(os.path.dirname(__file__), "asset_converter.py")

    def run_conversion():
        """Background thread that runs Blender to convert msgpack → GLB."""
        try:
            result = subprocess.run(
                [sys.executable, converter_path, asset_id, asset_dir],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            # Check if conversion succeeded
            with _conversion_lock:
                if result.returncode == 0 and os.path.exists(glb_path):
                    _conversion_jobs[asset_id]["status"] = "complete"
                else:
                    _conversion_jobs[asset_id]["status"] = "failed"
                    _conversion_jobs[asset_id]["error"] = result.stderr or "Conversion produced no GLB"

        except subprocess.TimeoutExpired:
            with _conversion_lock:
                _conversion_jobs[asset_id]["status"] = "failed"
                _conversion_jobs[asset_id]["error"] = "Conversion timeout (>2 min)"

        except Exception as e:
            with _conversion_lock:
                _conversion_jobs[asset_id]["status"] = "failed"
                _conversion_jobs[asset_id]["error"] = str(e)

    # Start conversion in background thread
    thread = threading.Thread(target=run_conversion, daemon=True)
    thread.start()

    return jsonify({"status": "converting"})


@app.route("/api/ensure-glb/<asset_id>")
def get_conversion_status(asset_id):
    """Get status of a conversion job."""
    glb_path = os.path.join(ASSET_BASE_DIR, asset_id, f"{asset_id}.glb")

    # Check if GLB exists (may have been converted by full pipeline)
    if os.path.exists(glb_path):
        return jsonify({"status": "exists"})

    with _conversion_lock:
        job = _conversion_jobs.get(asset_id)

    if not job:
        return jsonify({"status": "unknown", "error": "No conversion job found"})

    return jsonify({
        "status": job["status"],
        "error": job["error"],
    })


# ============================================================================
# Blender Render Endpoints
# ============================================================================

@app.route("/api/render", methods=["POST"])
def start_render():
    """
    Start a Blender render job for current canvas state.

    Input: {
        room_size: [w, h],
        items: [{asset_id, x, y, rotation, bbox_y}, ...],
        high_res: bool (optional)
    }
    Output: {job_id, status: "running"}
            OR {error, missing_glbs: [asset_ids...]} if GLBs not ready
    """
    data = request.json

    # Check all assets have GLB files before starting render
    missing_glbs = []
    converting_glbs = []
    for item in data.get("items", []):
        asset_id = item["asset_id"]
        glb_path = os.path.join(ASSET_BASE_DIR, asset_id, f"{asset_id}.glb")
        if not os.path.exists(glb_path):
            # Check if conversion is in progress
            with _conversion_lock:
                job = _conversion_jobs.get(asset_id)
            if job and job["status"] == "converting":
                converting_glbs.append(asset_id)
            else:
                missing_glbs.append(asset_id)

    if converting_glbs:
        return jsonify({
            "error": "Some assets are still converting to GLB format. Please wait a moment and try again.",
            "converting_glbs": converting_glbs,
        }), 202  # 202 Accepted = processing not complete

    if missing_glbs:
        return jsonify({
            "error": "Some assets are missing GLB files. They need to be converted first.",
            "missing_glbs": missing_glbs,
        }), 400

    job_id = uuid.uuid4().hex[:8]

    # Create temp directory for this job
    output_dir = tempfile.mkdtemp(prefix=f"render_{job_id}_")

    # Build input JSON for render worker
    input_data = {
        "room_size": data["room_size"],
        "wall_height": data.get("wall_height", 2.5),
        "asset_dir": ASSET_BASE_DIR,
        "items": data["items"],
        "high_res": data.get("high_res", False),
        "export_blend": data.get("export_blend", False),
        "disable_floor_plane": data.get("disable_floor_plane", False),
        "side_view_position": data.get("side_view_position", 3),
    }

    input_json_path = os.path.join(output_dir, "input.json")
    with open(input_json_path, "w") as f:
        json.dump(input_data, f)

    # Register job as running
    with _render_lock:
        _render_jobs[job_id] = {
            "status": "running",
            "output_dir": output_dir,
            "error": None,
        }

    # Path to render worker script
    worker_path = os.path.join(os.path.dirname(__file__), "render_worker.py")

    cmd = [sys.executable, worker_path, input_json_path, output_dir]

    def run_render():
        """Background thread that runs Blender and updates job status."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            # Check if render succeeded
            top_down_path = os.path.join(output_dir, "top_down_rendering.png")

            with _render_lock:
                if result.returncode == 0 and os.path.exists(top_down_path):
                    _render_jobs[job_id]["status"] = "complete"
                else:
                    _render_jobs[job_id]["status"] = "failed"
                    _render_jobs[job_id]["error"] = result.stderr or "Render produced no output"

        except subprocess.TimeoutExpired:
            with _render_lock:
                _render_jobs[job_id]["status"] = "failed"
                _render_jobs[job_id]["error"] = "Render timeout (>2 min)"

        except Exception as e:
            with _render_lock:
                _render_jobs[job_id]["status"] = "failed"
                _render_jobs[job_id]["error"] = str(e)

    # Start render in background thread
    thread = threading.Thread(target=run_render, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/api/render/<job_id>")
def get_render_status(job_id):
    """Get status of a render job."""
    with _render_lock:
        job = _render_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "error": job["error"],
    })


@app.route("/api/render/<job_id>/image/<view>")
def get_render_image(job_id, view):
    """
    Serve rendered image.

    Args:
        job_id: The render job ID
        view: 'top_down' or 'side'
    """
    with _render_lock:
        job = _render_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] != "complete":
        return jsonify({"error": f"Job not complete (status: {job['status']})"}), 400

    filename = "top_down_rendering.png" if view == "top_down" else "side_rendering.png"
    image_path = os.path.join(job["output_dir"], filename)

    if os.path.exists(image_path):
        return send_file(image_path, mimetype="image/png")

    return jsonify({"error": f"Image not found: {filename}"}), 404


@app.route("/api/render/<job_id>/blend")
def get_render_blend(job_id):
    """Serve exported .blend file if it exists."""
    with _render_lock:
        job = _render_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] != "complete":
        return jsonify({"error": f"Job not complete (status: {job['status']})"}), 400

    blend_path = os.path.join(job["output_dir"], "scene.blend")
    if os.path.exists(blend_path):
        return send_file(
            blend_path,
            mimetype="application/x-blender",
            as_attachment=True,
            download_name="scene.blend"
        )

    return jsonify({"error": ".blend export was not requested for this job"}), 404


class NoAvailablePortError(RuntimeError):
    """All ports in the search range are occupied."""


def find_available_port(
    host: str = HOST,
    start: int = PORT_RANGE_START,
    end: int = PORT_RANGE_END,
) -> int:
    """Return the first port in [start, end) that binds cleanly on `host`.

    We probe by actually binding (not lsof) so the check matches what werkzeug
    will do; SO_REUSEADDR avoids false-busy on TIME_WAIT sockets. There is a
    tiny TOCTOU window between close and werkzeug's rebind — same as Flask's
    own find_free_port; acceptable for a dev server.
    """
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
            return port
        except OSError:
            continue
    raise NoAvailablePortError(
        f"No free port in {start}-{end - 1} on {host}. "
        f"Free one (e.g. `lsof -i :{start}-{end - 1}`) and retry."
    )


def serve(host: str = HOST, port: int = PORT_RANGE_START, debug: bool = False) -> None:
    """Run the Scene Builder web server, blocking until Ctrl-C. We call werkzeug's
    run_simple directly rather than app.run, because app.run prints Flask's CLI banner
    (" * Serving Flask app ...") which would double the CLI's launch screen; run_simple
    skips it. Werkzeug's own startup/request logs are dropped to ERROR, so the CLI's
    launch screen stays the single source of server chrome. threaded=True restores the
    concurrency app.run gives by default (run_simple's default is single-threaded), so a
    slow request — e.g. the room-gen LLM call — doesn't block thumbnail / poll / GLB
    requests. Production: debug OFF, no reloader."""
    import logging
    from werkzeug.serving import run_simple
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    os.makedirs(ASSET_BASE_DIR, exist_ok=True)
    run_simple(host, port, app, use_reloader=False, use_debugger=debug, threaded=True)


if __name__ == "__main__":
    get_retriever()  # pre-load CLIP/SBERT so the first browser request is fast
    port = find_available_port()
    print(f"Serving at http://{HOST}:{port}")
    serve(HOST, port)
