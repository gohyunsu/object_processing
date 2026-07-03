#!/usr/bin/env python3
"""Add *_prev comparison entries to the static docs catalog."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_IDS = [
    "paper_cup",
    "tennis_ball",
    "tissue_box",
    "pepper_tuna",
    "paper_bowl",
    "tea_case",
]


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def copy_prev_object_docs() -> None:
    docs_objects = REPO_ROOT / "docs" / "objects"
    for object_id in OBJECT_IDS:
        src = docs_objects / object_id
        dst = docs_objects / f"{object_id}_prev"
        if dst.exists():
            print(f"kept existing docs/objects/{object_id}_prev")
            continue
        shutil.copytree(src, dst)

        info_path = dst / "info.json"
        if info_path.exists():
            info = read_json(info_path)
            info["id"] = f"{object_id}_prev"
            info["source_object_id"] = object_id
            info["variant"] = "prev"
            write_json(info_path, info)
        print(f"registered docs/objects/{object_id}_prev")


def register_catalog(path: Path) -> None:
    catalog = read_json(path)
    objects = catalog.get("objects", [])
    by_id = {entry.get("id"): entry for entry in objects}

    next_objects = []
    for entry in objects:
        object_id = entry.get("id")
        if object_id in OBJECT_IDS:
            entry.setdefault("variant", "current")
            entry.setdefault("compare_group", object_id)
        if object_id and object_id.endswith("_prev") and object_id[:-5] in OBJECT_IDS:
            continue

        next_objects.append(entry)

        if object_id in OBJECT_IDS:
            prev_id = f"{object_id}_prev"
            prev = dict(entry)
            prev["id"] = prev_id
            prev["label"] = f"{entry.get('label', object_id)} (prev)"
            prev["url"] = f"objects/{object_id}/mesh.glb"
            prev["thumb"] = f"objects/{prev_id}/thumb.png"
            prev["mesh_id"] = object_id
            prev["variant"] = "prev"
            prev["compare_group"] = object_id
            next_objects.append(prev)

    catalog["objects"] = next_objects
    write_json(path, catalog)
    print(f"updated {path.relative_to(REPO_ROOT)}: {len(by_id)} -> {len(next_objects)} entries")


def main() -> None:
    copy_prev_object_docs()
    register_catalog(REPO_ROOT / "docs" / "catalog.json")
    register_catalog(REPO_ROOT / "wiki" / "catalog.json")


if __name__ == "__main__":
    main()
