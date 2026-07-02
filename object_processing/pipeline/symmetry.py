"""Geometric rotational-symmetry detection for a mesh.

Detects the *proper* rotational symmetry group of a mesh (cyclic ``Cn`` /
continuous ``Cinf`` about an axis, plus dihedral perpendicular ``C2`` axes) and
records it as per-object metadata (``info/symmetry.json``). Reflections are
intentionally ignored — only proper rotations (the ones that map the solid onto
itself) are reported.

Frame convention
----------------
All axes are reported in the *mesh's own object frame* (the same frame as
``simplified.obj``, ``obb_transform``, and the tabletop poses), and pass through
the mesh center of mass.

Algorithm
---------
1. The inertia tensor's principal axes are the *only* candidate symmetry axes
   (a rotational symmetry axis is always a principal axis of inertia). This is
   cheap and exact, and narrows an SO(3) search to 3 directions.
2. Each candidate axis is *verified* geometrically: rotate a surface point
   sample about the axis and measure the RMS point-to-surface distance back to
   the original mesh, normalized by object scale. Point-to-surface (not
   point-to-point) makes this robust to sampling discretization. A residual
   below ``rel_tol`` means the rotation is a true symmetry.
3. The fold order is the largest ``n`` whose ``2*pi/n`` rotation verifies;
   incommensurate probe angles distinguish continuous (``"inf"``) from discrete.
"""

import json
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from object_processing.utils.tools import require_output

# Discrete fold orders to probe, in addition to the continuous test.
_FOLD_CANDIDATES = (2, 3, 4, 5, 6, 8, 10, 12)

# Incommensurate angles (radians) used to tell a continuous axis from a discrete
# one: none is close to 2*pi/n for any small n, so a finite Cn fails them all.
_CONTINUOUS_PROBES = (0.37, 0.91, 1.61, 2.21)


def _load_mesh(mesh_or_path):
    if isinstance(mesh_or_path, trimesh.Trimesh):
        return mesh_or_path
    mesh = trimesh.load(mesh_or_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"symmetry: expected a single mesh from {mesh_or_path}")
    return mesh


def _axis_rotation(axis, angle):
    """3x3 rotation matrix about a unit ``axis`` by ``angle`` radians."""
    return Rot.from_rotvec(np.asarray(axis, float) * angle).as_matrix()


def _residual(mesh, prox, pts, com, axis, angle):
    """Normalized RMS point-to-surface distance after rotating ``pts`` about the
    line through ``com`` along ``axis`` by ``angle``. ~0 iff the rotation is a
    symmetry. ``pts`` is already centered scale carries the normalization."""
    R = _axis_rotation(axis, angle)
    moved = (pts - com) @ R.T + com
    _, dist, _ = prox.on_surface(moved)
    return float(np.sqrt(np.mean(dist ** 2)))


def _fold_about_axis(mesh, prox, pts, com, axis, scale, rel_tol, max_fold):
    """Return ``(fold, residual)`` for one candidate axis.

    ``fold`` is ``"inf"`` (continuous), an int >= 2, or 1 (no symmetry). The
    residual is the worst normalized distance among the accepted generators.
    """
    tol = rel_tol * scale
    # Discrete folds must clear a stricter margin than the continuous test. A
    # marginal high-fold match near ``tol`` is almost always an accidental
    # near-alignment on a roughly-round-but-not-symmetric object (e.g. a brush
    # handle scraping C12 at res ~= tol), and a false Cn wrongly collapses real
    # tabletop poses. Genuine faceted revolutions clear this comfortably.
    disc_tol = 0.8 * tol

    # Continuous test first: if every incommensurate probe angle is invariant,
    # the axis is a body-of-revolution axis.
    cont_res = [_residual(mesh, prox, pts, com, axis, a) for a in _CONTINUOUS_PROBES]
    if max(cont_res) < tol:
        return "inf", max(cont_res) / scale

    best_n, best_res = 1, 0.0
    for n in _FOLD_CANDIDATES:
        if n > max_fold:
            break
        res = _residual(mesh, prox, pts, com, axis, 2 * np.pi / n)
        if res < disc_tol and n > best_n:
            best_n, best_res = n, res
    return best_n, best_res / scale


def detect_symmetry(
    mesh_or_path,
    n_samples=4000,
    rel_tol=0.01,
    max_fold=12,
    seed=0,
):
    """Detect the proper rotational symmetry group of a mesh.

    Returns a dict::

        {
          "type": "none" | "C2" | "C6" | "Cinf" | "D2" | "D6" | "Dinf" | ...,
          "center": [x, y, z],            # center of mass, in object frame
          "scale": float,                 # AABB diagonal (normalization)
          "axes": [                       # sorted by fold, highest first
            {"axis": [x,y,z], "fold": 6 | "inf", "residual": float},
            ...
          ],
          "rel_tol": float,               # acceptance threshold used
        }

    Only axes with fold >= 2 are listed. ``residual`` is the normalized
    point-to-surface RMS at the accepted generator (smaller = more exact); audit
    it when a detection looks wrong.
    """
    mesh = _load_mesh(mesh_or_path)
    com = np.asarray(mesh.center_mass, float)
    scale = float(mesh.scale)  # AABB diagonal
    if scale <= 0:
        raise ValueError("symmetry: degenerate mesh (zero scale)")

    # Principal axes of inertia (about COM) are the exact candidate axes when the
    # inertia ellipsoid is non-degenerate.
    eigvals, eigvecs = np.linalg.eigh(mesh.moment_inertia)
    candidate_axes = [eigvecs[:, i] for i in range(3)]

    # When the inertia ellipsoid is (near-)isotropic, the principal axes returned
    # by eigh are numerically arbitrary and can miss a real symmetry axis — this
    # is exactly the tall-as-wide body-of-revolution case (e.g. a cup whose
    # height ~= diameter makes the axial and transverse moments coincide). A
    # *relative* spread test catches it (the old absolute 1e-6 test never fired
    # for physical meshes). In that case also probe the object's coordinate axes
    # and body diagonals, since the true symmetry axis is no longer distinguished
    # as a principal axis.
    if np.ptp(eigvals) < 0.05 * max(eigvals.max(), 1e-12):
        for e in np.eye(3):
            candidate_axes.append(e)
        for s in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)):
            candidate_axes.append(np.array(s, float) / np.sqrt(3.0))

    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=int(rng.integers(1 << 31)))
    prox = trimesh.proximity.ProximityQuery(mesh)

    found = []
    for axis in candidate_axes:
        axis = axis / np.linalg.norm(axis)
        # Skip an axis already covered (duplicate direction up to sign).
        if any(abs(abs(np.dot(axis, f["_axis"])) - 1.0) < 1e-3 for f in found):
            continue
        fold, residual = _fold_about_axis(
            mesh, prox, pts, com, axis, scale, rel_tol, max_fold
        )
        if fold == 1:
            continue
        found.append({"_axis": axis, "axis": axis.tolist(), "fold": fold,
                      "residual": residual})

    def _rank(f):
        return np.inf if f["fold"] == "inf" else f["fold"]

    found.sort(key=_rank, reverse=True)

    sym_type = _classify(found)
    for f in found:
        del f["_axis"]

    return {
        "type": sym_type,
        "center": com.tolist(),
        "scale": scale,
        "axes": found,
        "rel_tol": rel_tol,
    }


def _classify(found):
    """Label the group from the detected axes (rotational part only)."""
    if not found:
        return "none"
    primary = found[0]
    n = "inf" if primary["fold"] == "inf" else int(primary["fold"])
    n_str = "inf" if n == "inf" else str(n)
    # Dihedral if there is a perpendicular 2-fold axis besides the primary.
    p_axis = np.array(primary["axis"])
    has_perp_c2 = any(
        f is not primary
        and (f["fold"] == 2 or f["fold"] == "inf")
        and abs(np.dot(p_axis, np.array(f["axis"]))) < 1e-2
        for f in found
    )
    return f"D{n_str}" if has_perp_c2 else f"C{n_str}"


# ── Pipeline stage ──────────────────────────────────────────────────────────


def compute_symmetry(input_path, output_path, **kwargs):
    """Pipeline stage: detect symmetry for ``input_path`` and write JSON.

    ``kwargs`` are forwarded to :func:`detect_symmetry`. Raises if the output is
    not produced.
    """
    sym = detect_symmetry(input_path, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sym, f, indent=1)
    require_output(output_path, "compute_symmetry")
    return sym


def load_symmetry(path):
    """Load a ``symmetry.json`` written by :func:`compute_symmetry`."""
    with open(path) as f:
        return json.load(f)
