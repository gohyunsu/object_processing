"""Stable tabletop-pose generation via MuJoCo physics settling.

Samples candidate orientations from the mesh surface, drops each onto a ground
plane in MuJoCo, keeps the ones that settle without moving, de-duplicates them,
and writes each accepted pose as a 4x4 SE(3) ``.npy`` (translation zeroed in the
table plane).
"""

import os
import json
import shutil
from copy import deepcopy

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering backend

import numpy as np
import mujoco
import transforms3d as t3d
import trimesh
from scipy.spatial.transform import Rotation as Rot

from object_processing.utils.rotation import batched_quat_delta

# A high-order rotational symmetry means the cross-section is effectively round:
# a simplified mesh of a body of revolution is a faceted prism, so its true Cinf
# is detected as a large finite Cn (e.g. C12). For tabletop de-duplication we
# treat any fold at/above this as continuous, so the "rolling on the round side"
# family of poses collapses to one instead of one-per-facet.
_CONTINUOUS_FOLD_MIN = 6


def _resting_up(qpos):
    """World-up direction expressed in the object frame.

    Invariant to **yaw**: spinning the whole object about world-z leaves this
    vector unchanged, so two poses resting on the same face compare equal
    regardless of their rotation on the table.
    """
    R = Rot.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_matrix()
    return R[2, :]


def _symmetry_ops(symmetry):
    """From a detected-symmetry dict, derive ``(continuous_axes, discrete_rots,
    spherical)``: object-frame revolute axes, n-fold rotation matrices, and a
    flag for full spherical symmetry (>= 2 independent *true* revolute axes)."""
    cont_axes, discrete_rots = [], []
    n_true_revolute = 0  # only genuine inf-fold axes imply a sphere
    for a in (symmetry or {}).get("axes", []):
        axis = np.asarray(a["axis"], float)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis / norm
        fold = a["fold"]
        if fold == "inf":
            cont_axes.append(axis)
            n_true_revolute += 1
        elif int(fold) >= _CONTINUOUS_FOLD_MIN:
            # A faceted body of revolution reads as a high finite Cn; treat it as
            # a revolute axis for cone de-duplication, but it is NOT evidence of a
            # sphere (else a round object with any secondary high fold would
            # wrongly collapse to a single pose).
            cont_axes.append(axis)
        else:
            for k in range(1, int(fold)):
                discrete_rots.append(
                    Rot.from_rotvec(axis * (2 * np.pi * k / int(fold))).as_matrix())
    spherical = n_true_revolute >= 2  # only a sphere has two+ revolute axes
    return cont_axes, discrete_rots, spherical


def _is_duplicated_pose(qpos, accepted, sym_ops, angle_thresh_deg=12.0, axis_tol=0.05):
    """True if this resting pose matches an accepted one, up to **yaw** and the
    object's **rotational symmetry**:

    - yaw: compared via the object-frame up-vector (:func:`_resting_up`);
    - revolute symmetry: up-vectors on the same cone about a revolute axis are
      equivalent (equal projection onto the axis);
    - discrete n-fold: the up-vector maps onto the other under an n-fold rotation;
    - spherical: every pose is equivalent, so only the first is kept.
    """
    if not accepted:
        return False
    cont_axes, discrete_rots, spherical = sym_ops
    if spherical:
        return True

    u = _resting_up(qpos)
    cos_th = np.cos(np.deg2rad(angle_thresh_deg))
    for prev in accepted:
        v = _resting_up(prev)
        if np.dot(u, v) > cos_th:                      # same resting face (yaw)
            return True
        for a in cont_axes:                            # same cone about a revolute axis
            if abs(np.dot(u, a) - np.dot(v, a)) < axis_tol:
                return True
        for Rk in discrete_rots:                        # related by an n-fold rotation
            if np.dot(Rk @ u, v) > cos_th:
                return True
    return False


def _build_settling_model(obj_spec):
    """Build a MuJoCo model: the object as a free body on a ground plane."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Native CCD (faster, more robust convex collision) was added in newer
    # MuJoCo; fall back gracefully on older builds rather than crashing.
    if hasattr(mujoco.mjtEnableBit, "mjENBL_NATIVECCD"):
        spec.option.enableflags = mujoco.mjtEnableBit.mjENBL_NATIVECCD
    else:
        print("warning: MuJoCo build lacks NATIVECCD; using default collision")
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 2
    spec.option.impratio = 10

    attach_frame = spec.worldbody.add_frame()
    child_world = attach_frame.attach_body(obj_spec.worldbody, "object", "")
    child_world.add_freejoint(name="freejoint")
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1.0],
        rgba=[0.5, 0.5, 0.5, 1.0],
    )
    spec.worldbody.add_camera(
        name="closeup", pos=[0.0, 1.0, 1.0], xyaxes=[-1, 0, 0, 0, -1, 1]
    )
    for g in spec.geoms:
        if g.contype != 0:
            g.condim = 4
            g.friction = [0.6, 0.02, 0.0001]
            if g.name != "floor":
                g.rgba = [0.2, 0.8, 0.2, 1.0]

    model = spec.compile()
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]
    model.vis.map.force = 0.1
    model.vis.map.fogstart = 3
    model.vis.map.fogend = 10
    return model


def generate_tabletop_poses(
    mjcf_path,
    output_dir,
    max_try_num=200,
    remove_duplicated=True,
    symmetry=None,
    debug_vis_path=None,
    seed=0,
):
    """Generate stable resting poses for the object described by ``mjcf_path``.

    Writes accepted poses to ``output_dir/{idx:03d}.npy`` as 4x4 SE(3) matrices.
    De-duplication folds out **yaw** (spin on the table) and, when ``symmetry`` is
    given (a detect_symmetry dict), the object's **rotational symmetry** — so a
    revolute can keeps one upright pose instead of dozens. Both ``output_dir`` and
    ``debug_vis_path`` are cleared first.
    """
    sym_ops = _symmetry_ops(symmetry)
    rng = np.random.default_rng(seed)  # deterministic surface sampling + in-plane spin
    shutil.rmtree(output_dir, ignore_errors=True)
    if debug_vis_path is not None:
        shutil.rmtree(debug_vis_path, ignore_errors=True)

    obj_spec = mujoco.MjSpec.from_file(mjcf_path)

    # Concatenate all collision meshes referenced by the MJCF.
    mesh_list = []
    for mesh in obj_spec.meshes:
        if mesh.file is not None:
            mesh_path = os.path.join(os.path.dirname(mjcf_path), mesh.file)
            mesh_list.append(trimesh.load(mesh_path, force="mesh"))
    if not mesh_list:
        raise RuntimeError(f"generate_tabletop_poses: no meshes in {mjcf_path}")
    combined = (
        trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]
    )

    # Candidate orientations: sample surface points, align each face normal to -Z
    # plus a random in-plane spin, then rest the lowest vertex just above z=0.
    points, face_indices = trimesh.sample.sample_surface_even(
        combined, count=max_try_num, seed=int(rng.integers(1 << 31)))

    trans_list, quat_list, face_info_list = [], [], []
    for p, face_idx in zip(points, face_indices):
        face_normal = combined.face_normals[face_idx]

        if remove_duplicated and any(
            np.dot(face_normal, info["face_normal"]) > 0.99 for info in face_info_list
        ):
            continue

        rotation, _ = Rot.align_vectors([[0, 0, -1]], [face_normal])
        z_spin = Rot.from_rotvec([0, 0, rng.uniform(0, 2 * np.pi)])
        final_rotation = z_spin * rotation

        quat = final_rotation.as_quat()  # [x, y, z, w]
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

        rotated_verts = (final_rotation.as_matrix() @ combined.vertices.T).T
        lowest_z = rotated_verts[:, 2].min()
        z_offset = 0.0005 - lowest_z

        trans_list.append([0, 0, z_offset])
        quat_list.append(quat_wxyz)
        face_info_list.append(
            {
                "face_idx": int(face_idx),
                "surface_point": p.tolist(),
                "face_normal": face_normal.tolist(),
                "lowest_z": float(lowest_z),
                "z_offset": float(z_offset),
            }
        )

    candidate_poses = np.concatenate(
        [np.array(trans_list), np.array(quat_list)], axis=-1
    )

    model = _build_settling_model(obj_spec)
    data = mujoco.MjData(model)

    accepted, accepted_ids = [], []
    for pose_id in range(len(candidate_poses)):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos = deepcopy(candidate_poses[pose_id])
        data.qvel = 0
        mujoco.mj_forward(model, data)

        settling_traj = []
        for i in range(1000):
            mujoco.mj_step(model, data)
            if i % 5 == 0:
                settling_traj.append(deepcopy(data.qpos))
        settled_qpos = deepcopy(data.qpos)

        if remove_duplicated and _is_duplicated_pose(settled_qpos, accepted, sym_ops):
            continue

        # Stability check: re-simulate briefly and require it stays put.
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos = deepcopy(settled_qpos)
        data.qvel = 0
        stability_traj = []
        for j in range(100):
            mujoco.mj_step(model, data)
            if j % 5 == 0:
                stability_traj.append(deepcopy(data.qpos))

        delta_angle, _ = batched_quat_delta(data.qpos[3:], settled_qpos[3:])
        delta_trans = np.linalg.norm(data.qpos[:3] - settled_qpos[:3])
        stable = delta_trans <= 0.001 and np.degrees(delta_angle) <= 1

        if stable:
            accepted.append(settled_qpos)
            accepted_ids.append(pose_id)

        if debug_vis_path is not None:
            save_dir = os.path.join(debug_vis_path, str(stable))
            os.makedirs(save_dir, exist_ok=True)
            traj = np.concatenate([np.array(settling_traj), np.array(stability_traj)], axis=0)
            np.save(os.path.join(save_dir, f"{pose_id}_traj.npy"), traj)
            with open(os.path.join(save_dir, f"{pose_id}_info.json"), "w") as f:
                json.dump(face_info_list[pose_id], f, indent=2)

    if not accepted:
        return []

    pose_array = np.array(accepted)
    pose_array[..., :2] = 0  # objects rest at the table origin in XY

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for pose_id, pose in zip(accepted_ids, pose_array):
        se3 = np.eye(4)
        se3[:3, 3] = pose[:3]
        se3[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
        out = os.path.join(output_dir, f"{pose_id:03d}.npy")
        np.save(out, se3)
        saved.append(out)
    return saved
