#!/usr/bin/env python3
"""Prepare Object_3DScan_1 meshes as current objects and keep *_prev copies.

The Artec OBJ exports are in millimeters and offset in scanner/world space.  The
web texture overrides already use bbox-centered meters, so this script writes the
same geometry transform into OBJECT_ROOT/<id>/raw_mesh/<id>.obj before rerunning
the processing pipeline.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import trimesh
from trimesh.registration import icp
from trimesh.visual.material import PBRMaterial, SimpleMaterial


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT_ROOT = Path.home() / "shared_data" / "object_processing"
ALIGNMENT_SAMPLES = 1200

SOURCES = {
    "paper_cup": ("PaperCup", "PaperCup"),
    "tennis_ball": ("Tennis_Ball", "Tennis_Ball"),
    "tissue_box": ("Tissue", "Tissue"),
    "pepper_tuna_light": ("Tuna_Can", "Tuna_Can"),
    "paper_bowl": ("Paper_Soup_Bowl", "Paper_Soup_Bowl"),
    "tea_case": ("Osulloc", "Osulloc"),
}


def object_root() -> Path:
    return Path(os.environ.get("OBJECT_ROOT", DEFAULT_OBJECT_ROOT)).expanduser().resolve()


def source_paths(source_dir_name: str, source_stem: str) -> tuple[Path, Path, Path]:
    base = REPO_ROOT / "Object_3DScan_1" / source_dir_name
    return base / f"{source_stem}.obj", base / f"{source_stem}.mtl", base / f"{source_stem}_1.png"


def parse_vertex(line: str) -> tuple[float, float, float] | None:
    parts = line.split()
    if len(parts) < 4 or parts[0] != "v":
        return None
    return float(parts[1]), float(parts[2]), float(parts[3])


def scan_bounds(obj_path: Path) -> tuple[list[float], list[float]]:
    mn = [float("inf"), float("inf"), float("inf")]
    mx = [float("-inf"), float("-inf"), float("-inf")]
    with obj_path.open() as f:
        for line in f:
            vertex = parse_vertex(line)
            if vertex is None:
                continue
            for i, value in enumerate(vertex):
                mn[i] = min(mn[i], value)
                mx[i] = max(mx[i], value)
    if not all(value < float("inf") for value in mn):
        raise ValueError(f"{obj_path}: no OBJ vertices found")
    return mn, mx


def write_transformed_obj(src: Path, dst: Path, mtl_name: str, center: list[float]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            vertex = parse_vertex(line)
            if vertex is not None:
                x, y, z = ((vertex[i] - center[i]) * 0.001 for i in range(3))
                fout.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            elif line.startswith("mtllib "):
                fout.write(f"mtllib {mtl_name}\n")
            else:
                fout.write(line)


def write_aligned_obj(
    src: Path,
    dst: Path,
    mtl_name: str,
    center: list[float],
    align_transform: np.ndarray,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    R = align_transform[:3, :3]
    t = align_transform[:3, 3]
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            vertex = parse_vertex(line)
            if vertex is not None:
                v = (np.asarray(vertex, dtype=float) - np.asarray(center, dtype=float)) * 0.001
                x, y, z = R @ v + t
                fout.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            elif line.startswith("mtllib "):
                fout.write(f"mtllib {mtl_name}\n")
            else:
                fout.write(line)


def write_mtl(src: Path, dst: Path, texture_name: str) -> None:
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if line.lstrip().startswith("map_Kd "):
                indent = line[: len(line) - len(line.lstrip())]
                fout.write(f"{indent}map_Kd {texture_name}\n")
            else:
                fout.write(line)


def _proper_signed_permutations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            mat = np.zeros((3, 3))
            for i, j in enumerate(perm):
                mat[i, j] = signs[i]
            if np.linalg.det(mat) > 0.5:
                mats.append(mat)
    return mats


PROPER_SIGNED_PERMUTATIONS = _proper_signed_permutations()


def load_centered_scan_mesh(src_obj: Path, center: list[float]) -> trimesh.Trimesh:
    mesh = trimesh.load(src_obj, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{src_obj}: expected a single mesh")
    mesh.vertices = (np.asarray(mesh.vertices, dtype=float) - np.asarray(center, dtype=float)) * 0.001
    return mesh


def load_prev_mesh(prev_dir: Path, object_id: str) -> trimesh.Trimesh:
    mesh_path = prev_dir / "raw_mesh" / f"{object_id}.obj"
    if not mesh_path.exists():
        candidates = sorted((prev_dir / "raw_mesh").glob("*.obj"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Could not find previous raw OBJ for {object_id}")
        mesh_path = candidates[0]
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{mesh_path}: expected a single mesh")
    return mesh


def inertia_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    _, axes = np.linalg.eigh(mesh.moment_inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1.0
    return axes


def align_scan_to_prev(scan_mesh: trimesh.Trimesh, prev_mesh: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """Proper-rigid ICP alignment from centered scan mesh into previous frame."""
    np.random.seed(0)
    scan_axes = inertia_axes(scan_mesh)
    prev_axes = inertia_axes(prev_mesh)
    points = scan_mesh.sample(ALIGNMENT_SAMPLES)
    scan_center = scan_mesh.centroid
    prev_center = prev_mesh.centroid

    best_transform = None
    best_cost = float("inf")
    for perm in PROPER_SIGNED_PERMUTATIONS:
        R = prev_axes @ perm @ scan_axes.T
        initial = np.eye(4)
        initial[:3, :3] = R
        initial[:3, 3] = prev_center - R @ scan_center
        transform, _, cost = icp(
            points,
            prev_mesh,
            initial=initial,
            threshold=1e-7,
            max_iterations=20,
            reflection=False,
            scale=False,
        )
        if np.linalg.det(transform[:3, :3]) < 0:
            continue
        if cost < best_cost:
            best_transform = transform
            best_cost = float(cost)

    if best_transform is None:
        raise RuntimeError("alignment failed: no proper rotation candidate")
    return best_transform, best_cost / ALIGNMENT_SAMPLES


def normalize_glb_materials(scene: trimesh.Scene) -> None:
    """Export scan textures as matte PBR instead of Artec's glossy OBJ material.

    Artec MTL files use ``Ks 1`` and ``Ns 1000``.  Trimesh converts that Phong
    shininess to a very low glTF roughness, which looks much darker in Babylon
    than the OBJ does in typical ambient/diffuse viewers.  Keep the texture
    image and diffuse color, but make the generated GLB material non-metallic
    and fully rough so it behaves like a diffuse object in glTF viewers.
    """
    for geometry in scene.geometry.values():
        material = getattr(geometry.visual, "material", None)
        if isinstance(material, SimpleMaterial):
            geometry.visual.material = PBRMaterial(
                baseColorTexture=material.image,
                baseColorFactor=material.diffuse,
                metallicFactor=0.0,
                roughnessFactor=1.0,
            )
        elif isinstance(material, PBRMaterial):
            material.metallicFactor = 0.0
            material.roughnessFactor = 1.0


def set_glb_materials_double_sided(glb_path: Path) -> None:
    """Mark all glTF materials as double-sided after GLB export.

    Trimesh preserves the scanned surfaces as single-sided glTF materials.  For
    thin/open scans this makes back faces disappear in Babylon, so interior views
    look hollow.  ``doubleSided`` keeps the same exterior texture visible from
    the reverse side without changing geometry.
    """
    data = glb_path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{glb_path}: expected glTF 2.0 GLB")

    offset = 12
    chunks = []
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8:offset + 8 + chunk_len]
        chunks.append((chunk_type, chunk))
        offset += 8 + chunk_len

    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise ValueError(f"{glb_path}: first chunk is not JSON")

    gltf = json.loads(chunks[0][1].rstrip(b" \t\r\n\0").decode("utf-8"))
    for material in gltf.get("materials", []):
        material["doubleSided"] = True

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    chunks[0] = (0x4E4F534A, json_bytes)

    total = 12 + sum(8 + len(chunk) for _, chunk in chunks)
    out = bytearray(struct.pack("<III", 0x46546C67, 2, total))
    for chunk_type, chunk in chunks:
        out.extend(struct.pack("<II", len(chunk), chunk_type))
        out.extend(chunk)
    glb_path.write_bytes(out)


def export_texture_override(target_obj: Path, object_id: str) -> tuple[str, int]:
    out_dir = REPO_ROOT / "docs" / "texture_overrides" / object_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mesh.glb"
    scene = trimesh.load(target_obj, force="scene")
    normalize_glb_materials(scene)
    scene.export(out_path, file_type="glb")
    set_glb_materials_double_sided(out_path)
    return str(out_path.relative_to(REPO_ROOT / "docs")), out_path.stat().st_size


def main() -> None:
    root = object_root()
    manifest = {}

    for object_id, (source_dir_name, source_stem) in SOURCES.items():
        current_dir = root / object_id
        prev_dir = root / f"{object_id}_prev"
        src_obj, src_mtl, src_png = source_paths(source_dir_name, source_stem)
        for src in (src_obj, src_mtl, src_png):
            if not src.exists():
                raise FileNotFoundError(src)

        if not current_dir.exists():
            raise FileNotFoundError(current_dir)

        if not prev_dir.exists():
            shutil.copytree(current_dir, prev_dir, symlinks=True)
            print(f"created {prev_dir}")
        else:
            print(f"kept existing {prev_dir}")

        processed_dir = current_dir / "processed_data"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
            print(f"cleared {processed_dir}")

        mn, mx = scan_bounds(src_obj)
        center = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
        extents = [mx[i] - mn[i] for i in range(3)]
        scan_mesh = load_centered_scan_mesh(src_obj, center)
        prev_mesh = load_prev_mesh(prev_dir, object_id)
        align_transform, align_cost = align_scan_to_prev(scan_mesh, prev_mesh)

        raw_dir = current_dir / "raw_mesh"
        target_obj = raw_dir / f"{object_id}.obj"
        target_mtl = raw_dir / f"{object_id}.mtl"
        target_png = raw_dir / f"{object_id}_1.png"

        write_aligned_obj(src_obj, target_obj, target_mtl.name, center, align_transform)
        write_mtl(src_mtl, target_mtl, target_png.name)
        shutil.copy2(src_png, target_png)
        texture_mesh, texture_size = export_texture_override(target_obj, object_id)

        meta = {
            "source": str(src_obj.relative_to(REPO_ROOT)),
            "transform": "bbox_centered_mm_to_m_then_prev_frame_rigid_icp",
            "source_bounds_mm": [mn, mx],
            "source_extents_mm": extents,
            "alignment_target": str((prev_dir / "raw_mesh" / f"{object_id}.obj").relative_to(root)),
            "alignment_samples": ALIGNMENT_SAMPLES,
            "alignment_cost": align_cost,
            "alignment_transform": align_transform.tolist(),
            "raw_mesh": str(target_obj),
            "texture_override": texture_mesh,
            "texture_override_size_bytes": texture_size,
            "prev_object_id": f"{object_id}_prev",
        }
        with (raw_dir / "object_3dscan_1.json").open("w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        manifest[object_id] = meta
        print(f"prepared {object_id}: {target_obj}  alignment_cost={align_cost:.3e}")

    manifest_path = root / "object_3dscan_1_manifest.json"
    with manifest_path.open("w") as f:
        json.dump({"objects": manifest}, f, indent=2)
        f.write("\n")
    print(f"wrote {manifest_path}")

    texture_manifest_path = REPO_ROOT / "docs" / "texture_overrides" / "manifest.json"
    texture_manifest = {
        "objects": {
            object_id: {
                "source_bounds_mm": meta["source_bounds_mm"],
                "source_extents_mm": meta["source_extents_mm"],
                "transform": meta["transform"],
                "alignment_target": meta["alignment_target"],
                "alignment_samples": meta["alignment_samples"],
                "alignment_cost": meta["alignment_cost"],
                "alignment_transform": meta["alignment_transform"],
                "mesh": meta["texture_override"],
                "source": meta["source"],
                "size_bytes": meta["texture_override_size_bytes"],
            }
            for object_id, meta in manifest.items()
        }
    }
    with texture_manifest_path.open("w") as f:
        json.dump(texture_manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {texture_manifest_path}")


if __name__ == "__main__":
    main()
