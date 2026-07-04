"""Export per-object assets for the interactive web viewer.

Two products per object:

- **Stage GLBs** (``stages/{raw,coacd,simplified}.glb``) — the mid-pipeline
  meshes the viewer toggles between. ``coacd`` is colored per convex piece so the
  decomposition is visible. These are large and meant to be uploaded to
  HuggingFace alongside the existing ``mesh.glb``.

- **Web info JSON** (``info.json``) — small overlay data in the *object frame*
  (same frame as every stage mesh): the oriented bounding box, the rotational
  symmetry axes, and the stable tabletop poses. Small enough to commit to git and
  serve from GitHub Pages.

Both the meshes and the overlays share one coordinate frame, so the viewer can
draw the overlays directly without any frame conversion.
"""

import glob
import json
import os

import numpy as np
import trimesh

from object_processing.utils.config import obj_dir
from object_processing.visualization.poses import teaser_pose

# Per-piece colors for the convex decomposition (RGB 0-255), matching render.py.
_PALETTE = [
    (229, 92, 92), (92, 158, 229), (117, 204, 102), (242, 184, 64),
    (178, 122, 219), (92, 204, 199), (237, 140, 77), (140, 166, 102),
    (217, 115, 178),
]

_STAGES = {
    "raw":        "raw_mesh/{obj}.obj",
    "coacd":      "processed_data/mesh/coacd.obj",
    "manifold":   "processed_data/mesh/manifold.obj",
    "simplified": "processed_data/mesh/simplified.obj",
}


def _filtered_symmetry_axes(sym, duplicate_dot=0.98):
    """Return display axes after removing visualization-only redundancy.

    ``Dinf`` objects can produce many equally valid axes, so the web viewer hides
    them entirely. Near-collinear axes for other symmetry types are treated as
    duplicates and the lower-residual hypothesis is kept.
    """
    if sym.get("type") == "Dinf":
        return []

    kept = []
    axes = sym.get("axes") or []
    axes = sorted(axes, key=lambda a: float(a.get("residual", float("inf"))))
    for axis in axes:
        vec = np.asarray(axis.get("axis", []), dtype=float)
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            continue
        vec = vec / norm
        if any(abs(float(np.dot(vec, kept_vec))) >= duplicate_dot for kept_vec, _ in kept):
            continue
        kept.append((vec, axis))
    return [{"axis": axis["axis"], "fold": axis["fold"]} for _, axis in kept]


def _color_components(mesh):
    """Paint each connected component of ``mesh`` a distinct palette color."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        mesh.visual.face_colors = _PALETTE[0] + (255,)
        return mesh
    for i, part in enumerate(parts):
        part.visual.face_colors = _PALETTE[i % len(_PALETTE)] + (255,)
    return trimesh.util.concatenate(parts)


def export_stage_glbs(obj_name, out_dir, root=None, stages=("raw", "coacd", "simplified")):
    """Export the available stage meshes of one object to GLB. Returns the list
    of stage names written. Missing stage meshes are skipped."""
    base = obj_dir(obj_name) if root is None else os.path.join(root, obj_name)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for stage in stages:
        src = os.path.join(base, _STAGES[stage].format(obj=obj_name))
        if not os.path.exists(src):
            continue
        if stage == "coacd":
            mesh = trimesh.load(src, force="mesh")
            scene = _color_components(mesh).scene()
        else:
            # force='scene' keeps materials/texture (raw) intact.
            scene = trimesh.load(src, force="scene")
        out = os.path.join(out_dir, f"{stage}.glb")
        scene.export(out, file_type="glb")
        written.append(stage)
    return written


def export_web_info(obj_name, out_path, root=None, max_poses=24):
    """Bundle OBB + symmetry + tabletop poses into one small JSON (object frame).

    Returns the dict written.
    """
    base = obj_dir(obj_name) if root is None else os.path.join(root, obj_name)
    info_dir = os.path.join(base, "processed_data", "info")

    info = {"id": obj_name}

    # Oriented bounding box (from basic_info).
    simp = os.path.join(info_dir, "simplified.json")
    if os.path.exists(simp):
        with open(simp) as f:
            s = json.load(f)
        info["obb"] = {"extents": s["obb"], "transform": s["obb_transform"]}
        info["gravity_center"] = s.get("gravity_center")

    # Rotational symmetry from the pipeline output.  The pipeline may choose a
    # higher-fidelity source than simplified.obj for noisy scans.
    sym_path = os.path.join(info_dir, "symmetry.json")
    if os.path.exists(sym_path):
        with open(sym_path) as f:
            sym = json.load(f)
        info["symmetry"] = {
            "type": sym["type"],
            "center": sym["center"],
            "scale": sym["scale"],
            "axes": _filtered_symmetry_axes(sym),
        }

    # Stable tabletop poses (4x4 SE3 each).
    pose_files = sorted(glob.glob(os.path.join(info_dir, "tabletop", "*.npy")))
    poses = [np.load(p).tolist() for p in pose_files[:max_poses]]
    pose_ids = [os.path.splitext(os.path.basename(p))[0] for p in pose_files[:max_poses]]
    info["tabletop_poses"] = poses
    info["tabletop_pose_ids"] = pose_ids
    info["n_tabletop_poses"] = len(pose_files)

    # Curated 'teaser' resting pose (object -> table) used for the hero shot in
    # the viewer (symmetry / tabletop overlays) and the thumbnails. Loaded from
    # disk directly so curated overrides beyond max_poses still resolve.
    P = teaser_pose(obj_name, base)
    if P is not None:
        info["display_pose"] = P.tolist()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f)
    return info


def export_object(obj_name, glb_out_root, info_out_root, root=None):
    """Export one object's stage GLBs (to ``glb_out_root/objects/{id}/stages``)
    and web info JSON (to ``info_out_root/objects/{id}/info.json``)."""
    stages_dir = os.path.join(glb_out_root, "objects", obj_name, "stages")
    info_path = os.path.join(info_out_root, "objects", obj_name, "info.json")
    written = export_stage_glbs(obj_name, stages_dir, root=root)
    info = export_web_info(obj_name, info_path, root=root)
    return written, info
