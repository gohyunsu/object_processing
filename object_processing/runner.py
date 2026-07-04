"""Batch orchestration shared by the per-command scripts in ``src/``.

Each ``*_object`` function does the work for ONE object and raises on failure.
:func:`run_batch` maps one of them over many objects (optionally in parallel) and
:func:`report` prints a loud per-object failure summary. The thin scripts in
``src/`` (process.py, decimate.py, symmetry.py, tabletop.py) just parse args and
call these — no business logic lives in ``src/``.
"""

import json
import os
import sys
import traceback
from multiprocessing import Pool

from object_processing.utils.config import object_root, obj_dir, find_raw_mesh
from object_processing import pipeline


def _load_symmetry(info_dir):
    """Detected symmetry (info/symmetry.json) if present, else None.

    Tabletop de-duplication uses it to fold out the object's rotational symmetry
    (replacing the old hardcoded cylindrical-objects list).
    """
    path = os.path.join(info_dir, "symmetry.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _compute_object_symmetry(obj_name, output_path, rel_tol=0.01, n_samples=4000):
    """Compute symmetry from the highest-fidelity mesh that gives a useful axis.

    The processed ``simplified.obj`` can lose rotational symmetry for noisy
    high-resolution scans after convex decomposition/remeshing, while the raw
    scan usually preserves it.  Some scans are the opposite, so fall back to the
    simplified mesh only when raw reports no rotational symmetry.
    """
    root = obj_dir(obj_name)
    mesh_dir = os.path.join(root, "processed_data", "mesh")
    raw_obj = os.path.join(mesh_dir, "raw.obj")
    raw_input = raw_obj if os.path.exists(raw_obj) else find_raw_mesh(obj_name)

    sym = pipeline.compute_symmetry(
        raw_input, output_path, rel_tol=rel_tol, n_samples=n_samples)
    if sym.get("type") != "none":
        return sym

    simplified_obj = os.path.join(mesh_dir, "simplified.obj")
    if os.path.exists(simplified_obj):
        simp_sym = pipeline.compute_symmetry(
            simplified_obj, output_path, rel_tol=rel_tol, n_samples=n_samples)
        if simp_sym.get("type") != "none":
            return simp_sym

    return sym


# ── per-object actions (each raises on failure) ──────────────────────────────

def process_object(obj_name, skip=False, quiet=True):
    """Run the full mesh pipeline for one object.

    Produces under ``{OBJECT_ROOT}/{obj_name}/processed_data/``:
        mesh/{raw,coacd,manifold,simplified}.obj
        urdf/{coacd.urdf,coacd.xml,meshes/}
        info/{simplified.json, symmetry.json, tabletop/*.npy, debug/}
    """
    root = obj_dir(obj_name)
    raw_in = find_raw_mesh(obj_name)  # format/name-agnostic; raises if absent/ambiguous

    mesh_dir = os.path.join(root, "processed_data", "mesh")
    urdf_dir = os.path.join(root, "processed_data", "urdf")
    info_dir = os.path.join(root, "processed_data", "info")
    parts_dir = os.path.join(urdf_dir, "meshes")

    raw_obj = os.path.join(mesh_dir, "raw.obj")
    coacd_obj = os.path.join(mesh_dir, "coacd.obj")
    manifold_obj = os.path.join(mesh_dir, "manifold.obj")
    simplified_ply = os.path.join(mesh_dir, "simplified.ply")
    simplified_obj = os.path.join(mesh_dir, "simplified.obj")
    urdf_path = os.path.join(urdf_dir, "coacd.urdf")
    mjcf_path = os.path.join(urdf_dir, "coacd.xml")
    info_path = os.path.join(info_dir, "simplified.json")
    symmetry_path = os.path.join(info_dir, "symmetry.json")
    tabletop_dir = os.path.join(info_dir, "tabletop")

    # (output-path-to-test-for-skip, callable).
    steps = [
        (raw_obj, lambda: pipeline.change_format(raw_in, raw_obj, keep_material=True)),
        (coacd_obj, lambda: pipeline.convex_decompose(raw_obj, coacd_obj, parts_dir, quiet=quiet)),
        (urdf_path, lambda: pipeline.export_urdf(parts_dir, urdf_path)),
        (mjcf_path, lambda: pipeline.export_mjcf(parts_dir, mjcf_path)),
        (manifold_obj, lambda: pipeline.manifold(coacd_obj, manifold_obj, quiet=quiet)),
        (simplified_ply, lambda: pipeline.simplify(manifold_obj, simplified_ply, quiet=quiet)),
        (simplified_obj, lambda: pipeline.change_format(
            simplified_ply, simplified_obj, keep_material=False, delete_input=True)),
        (info_path, lambda: pipeline.basic_info(simplified_obj, info_path)),
        (symmetry_path, lambda: _compute_object_symmetry(obj_name, symmetry_path)),
        # Tabletop generation always saves the per-candidate settling motion to
        # info/debug/ (the table_top viewer's "show motion" toggle reads it).
        (tabletop_dir, lambda: pipeline.generate_tabletop_poses(
            mjcf_path, tabletop_dir,
            symmetry=_load_symmetry(info_dir),
            debug_vis_path=os.path.join(info_dir, "debug"))),
    ]

    for out_path, step in steps:
        if skip and out_path is not None and os.path.exists(out_path):
            continue
        step()


def decimate_object(obj_name, target_faces=50_000, overwrite=False):
    """Decimate one object's raw mesh for fast viewing."""
    from object_processing.pipeline.decimate import decimate  # needs pymeshlab

    root = obj_dir(obj_name)
    out_dir = os.path.join(root, "raw_mesh_decimated")
    if os.path.isdir(out_dir) and not overwrite:
        print(f"  skip (exists): {obj_name}")
        return
    decimate(obj_name, os.path.join(root, "raw_mesh"), out_dir, target_faces)


def symmetry_object(obj_name, rel_tol=0.01, n_samples=4000, overwrite=False):
    """Detect rotational symmetry for one object (writes info/symmetry.json)."""
    info_dir = os.path.join(obj_dir(obj_name), "processed_data", "info")
    out_path = os.path.join(info_dir, "symmetry.json")
    if os.path.exists(out_path) and not overwrite:
        print(f"  skip (exists): {obj_name}")
        return
    sym = _compute_object_symmetry(
        obj_name, out_path, rel_tol=rel_tol, n_samples=n_samples)
    axes = ", ".join(
        f"{a['fold']}-fold (res {a['residual']:.4f})" for a in sym["axes"]
    ) or "none"
    print(f"  {obj_name}: {sym['type']}  [{axes}]")


def tabletop_object(obj_name, max_try=200):
    """(Re)generate stable tabletop poses for one object.

    Always also writes the per-candidate settling/stability trajectories to
    ``info/debug/{True,False}/`` for the table_top viewer's motion playback.
    """
    info_dir = os.path.join(obj_dir(obj_name), "processed_data", "info")
    mesh_dir = os.path.join(obj_dir(obj_name), "processed_data", "mesh")
    mjcf_path = os.path.join(obj_dir(obj_name), "processed_data", "urdf", "coacd.xml")

    # Symmetry must be known FIRST so redundant poses (equivalent under the
    # object's rotational symmetry) can be dropped — compute it if missing.
    sym_path = os.path.join(info_dir, "symmetry.json")
    if not os.path.exists(sym_path):
        _compute_object_symmetry(obj_name, sym_path)

    saved = pipeline.generate_tabletop_poses(
        mjcf_path, os.path.join(info_dir, "tabletop"),
        max_try_num=max_try,
        symmetry=_load_symmetry(info_dir),
        debug_vis_path=os.path.join(info_dir, "debug"),
    )
    print(f"  {obj_name}: {len(saved)} stable poses (+motion at info/debug/)")


# ── batch helpers ────────────────────────────────────────────────────────────

def discover_objects():
    """Every object under OBJECT_ROOT that has a raw_mesh/ directory."""
    root = object_root()
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, "raw_mesh"))
    )


def resolve_objects(names, all_):
    if all_:
        return discover_objects()
    if names:
        return list(names)
    raise SystemExit("specify object name(s) or --all")


def _run_one(args):
    fn, obj = args
    try:
        fn(obj)
        return obj, None
    except Exception:
        return obj, traceback.format_exc()


def run_batch(objs, fn, workers=1):
    """Run ``fn(obj)`` for each obj, returning ``[(obj, traceback_or_None)]``.

    ``fn`` must be picklable when ``workers > 1`` (use a module-level function or
    a ``functools.partial`` of one).
    """
    work = [(fn, o) for o in objs]
    if workers > 1:
        with Pool(processes=workers) as pool:
            return pool.map(_run_one, work)
    return [_run_one(w) for w in work]


def report(results):
    """Print a loud per-object summary; return the number of failures."""
    failures = [(name, tb) for name, tb in results if tb is not None]
    n_ok = len(results) - len(failures)
    print(f"\n=== {n_ok}/{len(results)} objects succeeded ===")
    for name, tb in failures:
        print(f"\n----- FAILED: {name} -----\n{tb}", file=sys.stderr)
    if failures:
        print(
            f"{len(failures)} object(s) FAILED: "
            + ", ".join(name for name, _ in failures),
            file=sys.stderr,
        )
    return len(failures)
