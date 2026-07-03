#!/usr/bin/env python3
"""Prepare Object_3DScan_1 meshes as current objects and keep *_prev copies.

The Artec OBJ exports are in millimeters and offset in scanner/world space.  The
web texture overrides already use bbox-centered meters, so this script writes the
same geometry transform into OBJECT_ROOT/<id>/raw_mesh/<id>.obj before rerunning
the processing pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT_ROOT = Path.home() / "shared_data" / "object_processing"

SOURCES = {
    "paper_cup": ("PaperCup", "PaperCup"),
    "tennis_ball": ("Tennis_Ball", "Tennis_Ball"),
    "tissue_box": ("Tissue", "Tissue"),
    "pepper_tuna": ("Tuna_Can", "Tuna_Can"),
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


def write_mtl(src: Path, dst: Path, texture_name: str) -> None:
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if line.lstrip().startswith("map_Kd "):
                indent = line[: len(line) - len(line.lstrip())]
                fout.write(f"{indent}map_Kd {texture_name}\n")
            else:
                fout.write(line)


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

        raw_dir = current_dir / "raw_mesh"
        target_obj = raw_dir / f"{object_id}.obj"
        target_mtl = raw_dir / f"{object_id}.mtl"
        target_png = raw_dir / f"{object_id}_1.png"

        write_transformed_obj(src_obj, target_obj, target_mtl.name, center)
        write_mtl(src_mtl, target_mtl, target_png.name)
        shutil.copy2(src_png, target_png)

        meta = {
            "source": str(src_obj.relative_to(REPO_ROOT)),
            "transform": "bbox_centered_mm_to_m",
            "source_bounds_mm": [mn, mx],
            "source_extents_mm": extents,
            "raw_mesh": str(target_obj),
            "prev_object_id": f"{object_id}_prev",
        }
        with (raw_dir / "object_3dscan_1.json").open("w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        manifest[object_id] = meta
        print(f"prepared {object_id}: {target_obj}")

    manifest_path = root / "object_3dscan_1_manifest.json"
    with manifest_path.open("w") as f:
        json.dump({"objects": manifest}, f, indent=2)
        f.write("\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
