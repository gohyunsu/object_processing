"""Curated 'teaser' tabletop pose per object — the single source of truth.

The pipeline emits several stable tabletop poses per object. The tallest one is a
sensible default 'hero' shot, but for some objects that pose is a near-duplicate,
a physically implausible resting pose, or simply an uninteresting side. For those
objects ``POSE_OVERRIDE`` pins a hand-picked tabletop pose (the ``.npy`` filename
stem under ``processed_data/info/tabletop/``).

This module is shared by everything that needs the same pose:
- the thumbnail renderer (``wiki/render_thumbnails.py``),
- the overlay figure (``render.render_overlays_figure``),
- the web viewer, via ``display_pose`` written into ``info.json`` by
  ``webexport.export_web_info``.

Only numpy / stdlib are imported here so it stays import-cheap (no Open3D).
"""

import glob
import json
import os

import numpy as np

# Per-object pose override: use this specific tabletop pose (.npy stem in
# processed_data/info/tabletop/) instead of the auto-selected tallest pose.
#
# NOTE: these stems were remapped after the symmetry / tabletop-dedup fix
# regenerated every pose file with new indices. Each was rematched to its
# original curated resting face by yaw-invariant up-vector (near-exact matches).
# pepper_tuna, pepper_tuna_light, soaptray and yellow_funnel lost their curated
# resting face during regeneration and now fall back to the auto-selected
# tallest pose — re-curate if their hero shot looks wrong.
POSE_OVERRIDE = {
    "blue_alarm": "003",   # stands the clock up on its legs (vs lying flat)
    "apple": "004",
    "baby_beaker": "021",
    "beige_brush": "005",
    "colander_green": "001",
    "blue_vase": "007",
    "box_pink": "003",
    "brass_pot": "002",
    "brown_ramen": "001",
    "container_pink": "001",
    "french_mustard": "000",
    "frog_bowl": "002",
    "frog_cup": "002",
    "fruit_cutter_base": "001",
    "fruit_cutter_green": "000",
    "fruit_cutter_light_green": "000",
    "green_attached_container": "000",
    "green_lamp": "011",
    "green_soap_dispenser": "015",
    "icecream_scoop": "000",
    "large_peg": "006",
    "lemon_squeezer": "000",
    "light_green_basket": "004",
    "open_box": "000",
    "organizer_beige": "001",
    "pastel_blue_cup": "007",
    "mug_holder": "005",
    "jja_ramen": "001",
    "pink_clock": "003",
    "paper_bowl": "001",
    "paper_cup": "006",
    "yellow_plastic_cup": "017",
    "wood_tray_big": "002",
    "wood_tray_small": "000",
    "standing_frame": "035",
    "plant_pot": "013",
    "toothbrush_holder": "000",
    "white_plastic_box": "002",
    "white_soap_dish": "001",
    "white_table_lamp": "011",
    "white_watering_can": "003",
    "magazine_file": "006",
    "shoe_organizer": "000",
    "smallbowl": "001",
    "spam_can": "002",
    "wateringcan": "001",
    "white_pen_cup": "006",
}


def _tallest_pose(poses, obb):
    """4x4 object->table for the most-upright stable pose (largest z-extent of
    the OBB after resting), or None. ``poses`` is a list of 4x4 array-likes;
    ``obb`` is ``{"extents": [...], "transform": [[...]]}``."""
    if not poses or not obb:
        return None
    e = obb["extents"]
    T = np.asarray(obb["transform"])
    hx, hy, hz = e[0] / 2, e[1] / 2, e[2] / 2
    local = np.array([[sx * hx, sy * hy, sz * hz, 1.0]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]).T
    cobj = T @ local
    best, best_h = None, -1e9
    for pose in poses:
        P = np.asarray(pose)
        w = P @ cobj
        h = w[2].max() - w[2].min()
        if h > best_h:
            best_h, best = h, P
    return best


def teaser_pose(obj_name, base):
    """Return the 4x4 object->table 'teaser' pose for ``obj_name`` as an
    ``np.ndarray``, or ``None`` if the object has no tabletop poses.

    Uses the curated ``POSE_OVERRIDE`` pose if listed, otherwise the tallest
    stable pose. ``base`` is the object's source dir
    (``…/object_processing/<name>``); poses and OBB are read from
    ``<base>/processed_data/info/``.
    """
    info_dir = os.path.join(base, "processed_data", "info")
    stem = POSE_OVERRIDE.get(obj_name)
    if stem is not None:
        return np.load(os.path.join(info_dir, "tabletop", f"{stem}.npy"))

    pose_files = sorted(glob.glob(os.path.join(info_dir, "tabletop", "*.npy")))
    simp = os.path.join(info_dir, "simplified.json")
    if not pose_files or not os.path.exists(simp):
        return None
    with open(simp) as f:
        s = json.load(f)
    obb = {"extents": s["obb"], "transform": s["obb_transform"]}
    return _tallest_pose([np.load(p) for p in pose_files], obb)
