"""Build a MuJoCo scene with sparse full-width stripe obstacles (G1 stage-2 style)."""

import os
from typing import List, Tuple

import numpy as np


def generate_stripes(
    num_stripes: int = 4,
    difficulty: float = 0.4,
    min_gap: float = 2.2,
    max_gap: float = 5.0,
    width: float = 0.1,
    start_x: float = 1.2,
    seed: int = 0,
) -> List[Tuple[float, float]]:
    """Return list of (stripe_center_x, height_m)."""
    rng = np.random.default_rng(seed)
    min_h = 0.05 + difficulty * 0.08
    max_h = 0.07 + difficulty * 0.08
    stripes = []
    x = start_x
    for _ in range(num_stripes):
        h = float(rng.uniform(min_h, max_h))
        stripes.append((x, h))
        x += width + float(rng.uniform(min_gap, max_gap))
    return stripes


def write_stripe_scene(
    out_path: str,
    unitree_root: str,
    stripes: List[Tuple[float, float]],
    half_width_y: float = 3.75,
    g1_dir: str = None,
) -> str:
    """Write MuJoCo XML: Unitree G1 + floor + stripe boxes (default geom appearance)."""
    if g1_dir is None:
        g1_dir = os.path.join(
            unitree_root, "resources", "robots", "g1_description"
        )
    g1_xml = os.path.join(g1_dir, "g1_12dof.xml")
    if not os.path.isfile(g1_xml):
        raise FileNotFoundError(f"G1 MuJoCo model not found: {g1_xml}")

    stripe_geoms = []
    for i, (x, h) in enumerate(stripes):
        stripe_geoms.append(
            f'    <geom name="stripe_{i}" type="box" pos="{x:.4f} 0 {h / 2:.4f}" '
            f'size="0.05 {half_width_y:.3f} {h / 2:.4f}" '
            f'friction="0.8 0.6" condim="3"/>'
        )
    stripes_xml = "\n".join(stripe_geoms)

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="g1_wmp_stripe_scene">
  <include file="g1_12dof.xml"/>
  <statistic center="2.0 0 0.8" extent="2.5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="2 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.8 0.6"/>
{stripes_xml}
  </worldbody>
</mujoco>
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path


def resolve_scene_xml(cfg: dict, repo_root: str, unitree_root: str) -> Tuple[str, List[Tuple[float, float]]]:
    terrain = str(cfg.get("terrain", "stripe")).lower()
    if terrain in ("flat", "plane", "none"):
        flat = cfg.get("xml_path")
        if not flat:
            flat = "{UNITREE_RL_GYM_ROOT}/resources/robots/g1_description/scene.xml"
        path = os.path.abspath(
            flat.replace("{UNITREE_RL_GYM_ROOT}", unitree_root).replace("{WMP_ROOT}", repo_root)
        )
        return path, []

    seed = int(cfg.get("terrain_seed", -1))
    if seed < 0:
        seed = int(np.random.randint(0, 10000))
    stripes = generate_stripes(
        num_stripes=int(cfg.get("stripe_num", 4)),
        difficulty=float(cfg.get("terrain_difficulty", 0.4)),
        min_gap=float(cfg.get("stripe_min_gap", 2.2)),
        max_gap=float(cfg.get("stripe_max_gap", 5.0)),
        seed=seed,
    )
    g1_dir = os.path.join(unitree_root, "resources", "robots", "g1_description")
    out_path = os.path.join(g1_dir, f"wmp_stripe_seed{seed}.xml")
    write_stripe_scene(out_path, unitree_root, stripes, g1_dir=g1_dir)
    print(f"[sim2sim] terrain=stripe seed={seed}")
    for i, (x, h) in enumerate(stripes):
        print(f"  stripe {i}: x={x:.2f}m  h={h * 100:.1f}cm")
    return out_path, stripes
