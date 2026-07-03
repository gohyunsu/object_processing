"""Convex decomposition and lightweight manifold repair."""

import os
import shutil

import coacd
import trimesh

from object_processing.utils.tools import require_output

# CoACD 1.x exposes Python bindings and a CLI that no longer supports the legacy
# ``-pf/-pn`` piece-output flags this pipeline used.  Call the API directly so we
# can keep the same downstream layout: merged ``coacd.obj`` plus per-piece OBJs.
DEFAULT_DECOMP_ARGS = ()


def convex_decompose(
    input_path,
    output_path,
    parts_dir,
    part_prefix="convex_piece",
    extra_args=DEFAULT_DECOMP_ARGS,
    quiet=True,
    threshold=0.05,
):
    """Decompose a mesh into convex pieces with CoACD.

    Writes the merged decomposition to ``output_path`` (e.g. ``coacd.obj``) and
    the individual convex pieces to ``parts_dir`` as ``{part_prefix}*.obj`` —
    those pieces feed :func:`export_urdf` / :func:`export_mjcf`.
    """
    if extra_args:
        print(f"warning: ignoring legacy CoACD args: {' '.join(extra_args)}")
    if quiet:
        coacd.set_log_level("error")

    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    src = trimesh.load(input_path, force="mesh")
    if not isinstance(src, trimesh.Trimesh):
        raise RuntimeError(f"convex_decompose: could not load mesh from {input_path}")

    decomposition = coacd.run_coacd(
        coacd.Mesh(src.vertices, src.faces),
        threshold=threshold,
        preprocess_mode="auto",
        decimate=True,
        seed=0,
    )
    if not decomposition:
        raise RuntimeError(f"convex_decompose: CoACD returned no parts for {input_path}")

    parts = []
    for i, (vertices, faces) in enumerate(decomposition):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        parts.append(part)
        part.export(os.path.join(parts_dir, f"{part_prefix}_{i:03d}.obj"))

    trimesh.util.concatenate(parts).export(output_path)
    require_output(output_path, "convex_decompose")


def manifold(input_path, output_path, level_set=0.1, quiet=True):
    """Produce the best available repaired mesh from ``input_path``.

    The legacy CoACD binary exposed a remesh-output mode; the current CoACD
    Python/CLI distribution does not.  For this pipeline stage we still emit a
    repaired mesh so ACVD can simplify it and downstream info/symmetry export has
    a stable path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    mesh = trimesh.load(input_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"manifold: could not load mesh from {input_path}")
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    mesh.export(output_path)
    require_output(output_path, "manifold")
