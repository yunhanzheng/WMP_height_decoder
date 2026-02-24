#!/usr/bin/env python3
"""
Generate terrain height maps as 16-bit (or 8-bit) grayscale PNG files for MuJoCo.

Terrain type index mapping (terrain_proportions):
  0  smooth_slope      – pyramid-shaped slope
  1  rough_slope       – pyramid slope + noise
  2  stairs_down       – inward pyramid stairs (descending)
  3  stairs_up         – outward pyramid stairs (ascending)
  4  discrete_obstacles – random rectangular blocks
  5  domino            – dense grid of thin walls
  6  stripes           – horizontal stripe obstacles
  7  edge_obstacle     – one stripe at each end of the cell

Proportions must sum to 1.0.  Unused types should be set to 0.

Example (all domino terrain, same as GO2 blind config):
  python generate_terrain_heightmap.py \\
    --terrain_proportions 0 0 0 0 0 1 0 \\
    --terrain_length 15 --terrain_width 15 \\
    --num_rows 10 --num_cols 10 \\
    --output terrain.png --visualize

Example (mixed curriculum):
  python generate_terrain_heightmap.py \\
    --terrain_proportions 0.1 0.1 0.3 0.25 0.15 0.1 \\
    --mode curriculum --num_rows 10 --num_cols 20 \\
    --output terrain_mixed.png --visualize
"""

import argparse
import os
import sys

import numpy as np

# ── Terrain primitive functions (no isaacgym required) ────────────────────────


class SubTerrain:
    """Lightweight sub-terrain tile (mirrors legged_gym/utils/terrain_utils.py)."""

    def __init__(self, name="terrain", width=256, length=256,
                 vertical_scale=0.005, horizontal_scale=0.1):
        self.terrain_name = name
        self.width = width          # pixels along x
        self.length = length        # pixels along y
        self.vertical_scale = vertical_scale    # m per height unit
        self.horizontal_scale = horizontal_scale  # m per pixel
        self.height_field_raw = np.zeros((width, length), dtype=np.int16)


# ── individual terrain generators ────────────────────────────────────────────

def random_uniform_terrain(terrain, min_height, max_height, step=1.0,
                            downsampled_scale=None):
    """Uniform random noise terrain, optionally smoothed by downsampling."""
    from scipy.ndimage import zoom as nd_zoom

    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    min_h = int(min_height / terrain.vertical_scale)
    max_h = int(max_height / terrain.vertical_scale)
    step_h = max(1, int(step / terrain.vertical_scale))

    heights_range = np.arange(min_h, max_h + step_h, step_h)
    ds_w = max(1, int(terrain.width * terrain.horizontal_scale / downsampled_scale))
    ds_l = max(1, int(terrain.length * terrain.horizontal_scale / downsampled_scale))
    downsampled = np.random.choice(heights_range, (ds_w, ds_l))

    zoom_w = terrain.width / ds_w
    zoom_l = terrain.length / ds_l
    upsampled = nd_zoom(downsampled, (zoom_w, zoom_l), order=1)
    # Ensure correct shape after zoom
    upsampled = upsampled[:terrain.width, :terrain.length]
    terrain.height_field_raw += np.rint(upsampled).astype(np.int16)


def pyramid_sloped_terrain(terrain, slope=1.0, platform_size=1.0):
    """Pyramid-shaped slope (peak at centre, descends to edges)."""
    cx = terrain.width // 2
    cy = terrain.length // 2
    x = np.arange(terrain.width)
    y = np.arange(terrain.length)
    fx = ((cx - np.abs(cx - x)) / max(cx, 1)).reshape(-1, 1)
    fy = ((cy - np.abs(cy - y)) / max(cy, 1)).reshape(1, -1)
    max_h = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) *
                (terrain.width / 2))
    terrain.height_field_raw += (max_h * fx * fy).astype(np.int16)

    ps = int(platform_size / terrain.horizontal_scale / 2)
    x1, x2 = cx - ps, cx + ps
    y1, y2 = cy - ps, cy + ps
    clip_val = int(terrain.height_field_raw[x1, y1])
    lo, hi = min(clip_val, 0), max(clip_val, 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, lo, hi)


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    """Concentric staircase (positive step_height → ascending towards centre)."""
    sw = int(step_width / terrain.horizontal_scale)
    sh = int(step_height / terrain.vertical_scale)
    ps = int(platform_size / terrain.horizontal_scale)
    h = 0
    x0, x1 = 0, terrain.width
    y0, y1 = 0, terrain.length
    while (x1 - x0) > ps and (y1 - y0) > ps:
        x0 += sw;  x1 -= sw
        y0 += sw;  y1 -= sw
        h += sh
        terrain.height_field_raw[x0:x1, y0:y1] = h


def discrete_obstacles_terrain(terrain, max_height, min_size, max_size,
                                num_rects, platform_size=1.0):
    """Random rectangular obstacles at various heights (positive and negative)."""
    mh = int(max_height / terrain.vertical_scale)
    ms = max(1, int(min_size / terrain.horizontal_scale))
    Ms = max(ms + 1, int(max_size / terrain.horizontal_scale))
    ps = int(platform_size / terrain.horizontal_scale)
    H, W = terrain.height_field_raw.shape
    h_range = [-mh, -mh // 2, mh // 2, mh]
    for _ in range(num_rects):
        w = np.random.randint(ms, Ms)
        l = np.random.randint(ms, Ms)
        i = np.random.randint(0, max(1, H - w))
        j = np.random.randint(0, max(1, W - l))
        terrain.height_field_raw[i:i + w, j:j + l] = np.random.choice(h_range)
    cx, cy = H // 2, W // 2
    terrain.height_field_raw[cx - ps // 2:cx + ps // 2,
                              cy - ps // 2:cy + ps // 2] = 0


def discrete_obstacles_terrain_cells(terrain, min_height, max_height,
                                     min_size, max_size, num_rects,
                                     platform_size=1.0, width=2):
    """Dense domino-style obstacles.

    min_size / max_size / width are in **pixel** units (not metres), matching
    the original legged_gym implementation.
    """
    ps = int(platform_size / terrain.horizontal_scale)
    H, W = terrain.height_field_raw.shape
    w = max(1, int(width))
    for _ in range(num_rects):
        l = np.random.randint(min_size, max(min_size + 1, max_size + 1))
        i = (np.random.randint(0, max(1, H - w - 1)) // 4) * 4
        j = (np.random.randint(0, max(1, W - l - 1)) // 4) * 4
        h_val = (min_height + np.random.rand() * (max_height - min_height)) \
                / terrain.vertical_scale
        terrain.height_field_raw[i:i + w, j:j + l] = int(h_val)
    cx, cy = H // 2, W // 2
    terrain.height_field_raw[cx - ps // 2:cx + ps // 2,
                              cy - ps // 2:cy + ps // 2] = 0


def discrete_stripes_obstacle_terrain(terrain, height, filled_rate,
                                      width=0.2, platform_size=3.0):
    """Horizontal stripe obstacles distributed across the terrain."""
    ps = int(platform_size / terrain.horizontal_scale)
    sw = max(1, int(width / terrain.horizontal_scale))
    hd = int(height / terrain.vertical_scale)
    H, W = terrain.height_field_raw.shape
    num_stripes = int(H * filled_rate / sw)
    if num_stripes > 0:
        spacing = H / num_stripes
        for k in range(num_stripes):
            i0 = int(k * spacing)
            i1 = min(i0 + sw, H)
            terrain.height_field_raw[i0:i1, :] = hd
    cx, cy = H // 2, W // 2
    terrain.height_field_raw[cx - ps // 2:cx + ps // 2,
                              cy - ps // 2:cy + ps // 2] = 0


def discrete_one_obstacle_terrain(terrain, height, width=0.2, platform_size=3.0):
    """One stripe at each end of the terrain cell."""
    ps = int(platform_size / terrain.horizontal_scale)
    sw = max(1, int(width / terrain.horizontal_scale))
    hd = int(height / terrain.vertical_scale)
    H, W = terrain.height_field_raw.shape
    terrain.height_field_raw[0:sw, :] = hd
    terrain.height_field_raw[max(0, H - sw):H, :] = hd
    cx, cy = H // 2, W // 2
    terrain.height_field_raw[cx - ps // 2:cx + ps // 2,
                              cy - ps // 2:cy + ps // 2] = 0


def pit_terrain(terrain, depth, platform_size=4.0):
    """Pit / hole in the centre of the cell."""
    d = int(depth / terrain.vertical_scale)
    ps = int(platform_size / terrain.horizontal_scale / 2)
    H, W = terrain.height_field_raw.shape
    cx, cy = H // 2, W // 2
    terrain.height_field_raw[cx - ps:cx + ps, cy - ps:cy + ps] = -d


# ── Terrain grid generator ────────────────────────────────────────────────────

TERRAIN_NAMES = [
    "smooth_slope",
    "rough_slope",
    "stairs_down",
    "stairs_up",
    "discrete_obstacles",
    "domino",
    "stripes",
    "edge_obstacle",
]


class TerrainGenerator:
    """Assembles a grid of sub-terrains into one height field.

    Replicates the logic in legged_gym/utils/terrain.py (Terrain class)
    without any isaacgym dependency.
    """

    def __init__(self, terrain_proportions, terrain_length, terrain_width,
                 num_rows, num_cols, horizontal_scale=0.1, vertical_scale=0.005,
                 border_size=0.0, difficulty=1.0, mode="curriculum", seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Pad to 8 elements and compute cumulative proportions
        tp = list(terrain_proportions)
        while len(tp) < 8:
            tp.append(0.0)
        self.proportions = [sum(tp[:i + 1]) for i in range(len(tp))]

        self.terrain_length = terrain_length
        self.terrain_width = terrain_width
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.difficulty = difficulty
        self.mode = mode

        # Pixels per sub-terrain cell
        self.l_px = int(terrain_length / horizontal_scale)
        self.w_px = int(terrain_width / horizontal_scale)
        self.border = int(border_size / horizontal_scale)

        tot_rows = num_rows * self.l_px + 2 * self.border
        tot_cols = num_cols * self.w_px + 2 * self.border
        self.height_field_raw = np.zeros((tot_rows, tot_cols), dtype=np.int16)

    # ── single-cell terrain builder (mirrors Terrain.make_terrain) ──────────

    def _make_terrain(self, choice, difficulty):
        t = SubTerrain(
            width=self.l_px, length=self.w_px,
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )
        p = self.proportions
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        obs_height = 0.05 + difficulty * 0.2

        if choice < p[0]:
            # Smooth slope (half pos, half neg)
            if choice < p[0] / 2:
                slope *= -1
            pyramid_sloped_terrain(t, slope=slope, platform_size=3.0)

        elif choice < p[1]:
            # Rough slope
            pyramid_sloped_terrain(t, slope=slope, platform_size=3.0)
            random_uniform_terrain(t, 0.0, 0.0, step=0.005, downsampled_scale=0.2)

        elif choice < p[3]:
            # Stairs: p[1]..p[2] → down, p[2]..p[3] → up
            if choice < p[2]:
                step_height *= -1
            pyramid_stairs_terrain(t, step_width=0.31, step_height=step_height,
                                   platform_size=3.0)

        elif choice < p[4]:
            # Random discrete obstacles
            discrete_obstacles_terrain(t, obs_height, 1.0, 2.0, 20, platform_size=3.0)

        elif choice < p[5]:
            # Domino (dense thin walls) – num_rects scales with difficulty
            num_rects = int(400 * difficulty)
            discrete_obstacles_terrain_cells(t, 0.10, 0.15, 2, 3, num_rects,
                                             platform_size=3.0, width=2)

        elif choice < p[6]:
            # Stripe obstacles
            filled_rate = 0.1 + difficulty * 0.5  # 10 % – 60 %
            discrete_stripes_obstacle_terrain(t, height=0.12, filled_rate=filled_rate)

        elif choice < p[7]:
            # Edge obstacles
            discrete_one_obstacle_terrain(t, height=0.12, width=0.2, platform_size=1.0)

        else:
            # Pit
            pit_terrain(t, depth=difficulty * 1.0, platform_size=4.0)

        return t

    # ── place a tile into the global map ────────────────────────────────────

    def _add_to_map(self, terrain, row, col):
        x0 = self.border + row * self.l_px
        y0 = self.border + col * self.w_px
        self.height_field_raw[x0:x0 + self.l_px, y0:y0 + self.w_px] = \
            terrain.height_field_raw

    # ── main entry point ─────────────────────────────────────────────────────

    def generate(self):
        """Fill the height field and return it as an int16 numpy array."""
        if self.mode == "curriculum":
            # col → terrain type, row → difficulty (mirrors Terrain.curiculum)
            for j in range(self.num_cols):
                for i in range(self.num_rows):
                    difficulty = i / self.num_rows          # 0 .. (n-1)/n
                    choice = j / self.num_cols + 0.001
                    self._add_to_map(self._make_terrain(choice, difficulty), i, j)
        else:
            # randomized mode – random type per cell, fixed difficulty
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    choice = np.random.uniform(0.0, 1.0)
                    self._add_to_map(
                        self._make_terrain(choice, self.difficulty), i, j)

        return self.height_field_raw


# ── Box geom export ───────────────────────────────────────────────────────────

def heightmap_to_boxes(height_field_raw, horizontal_scale, vertical_scale,
                       slope_threshold=None):
    """Convert a height field into a list of axis-aligned box obstacles.

    Uses a greedy rectangle-merging scan: each pixel above the minimum height
    is merged into the largest possible rectangle of the same height before
    stepping to the next unclaimed pixel.  This minimises the number of geoms
    while producing exact coverage.

    Only positive-height regions (above the terrain minimum) are returned.
    Negative features such as pits are ignored.

    When *slope_threshold* is provided (default 0.75, matching legged_gym), the
    function replicates the edge correction applied by
    ``convert_heightfield_to_trimesh``: wherever the drop at the far edge of a
    rectangle exceeds the threshold, the box is shrunk by one pixel on that
    side.  This makes the exported box geoms match the effective flat-top width
    seen in Isaac Gym (e.g. a 2-pixel stripe becomes a 1-pixel-wide box).

    Returns
    -------
    boxes : list of (cx, cy, cz, sx, sy, sz) tuples, all in metres.
        (cx, cy, cz) = box centre, (sx, sy, sz) = half-extents.
        The ground plane is assumed to be at z = 0.
    """
    hf = height_field_raw.astype(np.int32)
    ground = int(hf.min())
    H, W = hf.shape
    claimed = np.zeros((H, W), dtype=bool)
    boxes = []

    # Convert slope_threshold to raw height-units / pixel, matching the
    # scaling done inside convert_heightfield_to_trimesh:
    #   slope_threshold *= horizontal_scale / vertical_scale
    slope_raw = None
    if slope_threshold is not None:
        slope_raw = slope_threshold * horizontal_scale / vertical_scale

    for i in range(H):
        for j in range(W):
            if claimed[i, j] or hf[i, j] <= ground:
                claimed[i, j] = True
                continue

            h_val = hf[i, j]

            # Extend right while same height and unclaimed
            j2 = j + 1
            while j2 < W and hf[i, j2] == h_val and not claimed[i, j2]:
                j2 += 1

            # Extend down while the full strip [j:j2] stays same height & unclaimed
            i2 = i + 1
            while i2 < H and (np.all(hf[i2, j:j2] == h_val) and
                               not np.any(claimed[i2, j:j2])):
                i2 += 1

            claimed[i:i2, j:j2] = True

            height_m = (h_val - ground) * vertical_scale
            if height_m < 1e-6:
                continue

            # ── Slope correction (mirrors convert_heightfield_to_trimesh) ──
            # In the trimesh the ground vertex just past each steep edge is
            # moved inward by one pixel, making a vertical wall at the last
            # raised-pixel boundary.  The effective flat top therefore ends one
            # pixel earlier than the raw rectangle extent.  Apply the same
            # correction here so box geoms match the trimesh geometry.
            i_end = i2
            j_end = j2
            if slope_raw is not None:
                # Far x-edge: drop from last raised row to the next row
                if i2 < H and (h_val - int(hf[i2, j])) > slope_raw:
                    i_end = i2 - 1
                # Far y-edge: drop from last raised column to the next column
                if j2 < W and (h_val - int(hf[i, j2])) > slope_raw:
                    j_end = j2 - 1

            if i_end <= i or j_end <= j:
                continue  # degenerate after correction, skip

            sx = (i_end - i) * horizontal_scale / 2.0
            sy = (j_end - j) * horizontal_scale / 2.0
            sz = height_m / 2.0
            cx = (i + i_end) / 2.0 * horizontal_scale - H * horizontal_scale / 2.0
            cy = (j + j_end) / 2.0 * horizontal_scale - W * horizontal_scale / 2.0
            cz = sz  # bottom of box sits on z = 0 ground plane

            boxes.append((cx, cy, cz, sx, sy, sz))

    return boxes


def save_mjcf_xml(boxes, output_path, total_x_m, total_y_m):
    """Write a minimal MJCF XML file with box geoms for terrain obstacles.

    The ground plane is placed at z = 0.  Each box in *boxes* sits on top of
    it with perfectly vertical walls.

    Parameters
    ----------
    boxes       : list of (cx, cy, cz, sx, sy, sz) from heightmap_to_boxes()
    output_path : str – destination .xml file
    total_x_m   : float – total terrain extent in X (metres, for ground vis)
    total_y_m   : float – total terrain extent in Y (metres, for ground vis)
    """
    lines = [
        '<mujoco model="terrain">',
        '  <worldbody>',
        '    <!-- Infinite ground plane (size 0 0 1 = infinite for physics) -->',
        '    <geom name="ground" type="plane" size="0 0 1"',
        '          contype="1" conaffinity="1"/>',
    ]

    for k, (cx, cy, cz, sx, sy, sz) in enumerate(boxes):
        lines.append(
            f'    <geom name="obs_{k}" type="box"'
            f' pos="{cx:.4f} {cy:.4f} {cz:.4f}"'
            f' size="{sx:.4f} {sy:.4f} {sz:.4f}"'
            f' contype="1" conaffinity="1"/>'
        )

    lines += ['  </worldbody>', '</mujoco>']

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return len(boxes)


# ── PNG export ────────────────────────────────────────────────────────────────

def save_heightmap_png(height_field_raw, output_path, vertical_scale, bit_depth=16):
    """Normalise height field and save as a grayscale PNG for MuJoCo.

    MuJoCo maps pixel intensity in [0, 1] linearly to height in [0, elevation].
    We shift the field so that 0 corresponds to the minimum height.

    Returns
    -------
    h_min_m : float
        Minimum height in metres (pixel value 0 → this elevation).
    h_range_m : float
        Height range in metres (= elevation attribute in MuJoCo hfield).
    """
    hf = height_field_raw.astype(np.float64)
    h_min = hf.min()
    h_max = hf.max()
    h_min_m = float(h_min * vertical_scale)
    h_range_m = float((h_max - h_min) * vertical_scale) if h_max != h_min else 1e-3

    if h_max != h_min:
        normalised = (hf - h_min) / (h_max - h_min)
    else:
        normalised = np.zeros_like(hf)

    max_val = 65535 if bit_depth == 16 else 255
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    img_data = (normalised * max_val).astype(dtype)

    # Try imageio first (already a project dependency), fall back to PIL
    try:
        import imageio
        imageio.imwrite(output_path, img_data)
    except ImportError:
        try:
            from PIL import Image
            if bit_depth == 16:
                # PIL needs mode "I;16" for 16-bit grayscale PNG
                img = Image.fromarray(img_data.astype(np.int32), mode="I")
                img.save(output_path)
            else:
                img = Image.fromarray(img_data, mode="L")
                img.save(output_path)
        except ImportError:
            # Last resort: write raw bytes and rely on numpy's PNG writer
            import struct, zlib
            _write_png_manual(img_data, output_path, bit_depth)

    return h_min_m, h_range_m


def _write_png_manual(array, path, bit_depth):
    """Minimal 16-bit or 8-bit grayscale PNG writer (no external deps)."""
    import struct, zlib

    h, w = array.shape
    depth = bit_depth
    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'
    # IHDR
    ihdr_data = struct.pack('>IIBBBBB', w, h, depth, 0, 0, 0, 0)
    ihdr = _png_chunk(b'IHDR', ihdr_data)
    # IDAT
    raw_rows = []
    for row in array:
        if depth == 16:
            # big-endian uint16
            row_bytes = row.astype('>u2').tobytes()
        else:
            row_bytes = row.astype(np.uint8).tobytes()
        raw_rows.append(b'\x00' + row_bytes)  # filter byte = None
    raw = b''.join(raw_rows)
    compressed = zlib.compress(raw, 9)
    idat = _png_chunk(b'IDAT', compressed)
    iend = _png_chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(sig + ihdr + idat + iend)


def _png_chunk(tag, data):
    import struct, zlib
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_visualisation(height_field_raw, output_path, horizontal_scale,
                       num_rows, num_cols, terrain_length, terrain_width):
    """Save a colourised top-down view with sub-terrain grid lines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        hf = height_field_raw.astype(np.float32) * 1
        total_h_m = hf.shape[0] * horizontal_scale
        total_w_m = hf.shape[1] * horizontal_scale

        fig, ax = plt.subplots(figsize=(14, 14))
        im = ax.imshow(
            hf, cmap="terrain", origin="upper",
            extent=[0, total_w_m, total_h_m, 0],
        )
        # Grid lines showing sub-terrain boundaries
        for i in range(1, num_rows):
            ax.axhline(i * terrain_length, color="red", lw=0.5, alpha=0.6)
        for j in range(1, num_cols):
            ax.axvline(j * terrain_width, color="red", lw=0.5, alpha=0.6)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax, label="Height (raw int16 units)")
        ax.set_title("Generated Terrain Height Map  (red lines = sub-terrain borders)")
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("X [m]")
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        return True
    except ImportError as e:
        print(f"  (Visualisation skipped: {e})")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--terrain_proportions", type=float, nargs="+",
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        metavar="P",
        help=(
            "Proportion for each terrain type (must sum to 1.0). "
            "Indices: 0=smooth_slope, 1=rough_slope, 2=stairs_down, "
            "3=stairs_up, 4=discrete_obstacles, 5=domino, 6=stripes, 7=edge_obstacle"
        ),
    )
    parser.add_argument("--terrain_length", type=float, default=15.0,
                        help="Length of each sub-terrain cell [m] (default: 15.0)")
    parser.add_argument("--terrain_width", type=float, default=15.0,
                        help="Width of each sub-terrain cell [m] (default: 15.0)")
    parser.add_argument("--num_rows", type=int, default=10,
                        help="Number of terrain rows / difficulty levels (default: 10)")
    parser.add_argument("--num_cols", type=int, default=10,
                        help="Number of terrain columns / types (default: 10)")
    parser.add_argument("--horizontal_scale", type=float, default=0.1,
                        help="Metres per pixel (default: 0.1)")
    parser.add_argument("--vertical_scale", type=float, default=0.005,
                        help="Metres per height unit (default: 0.005)")
    parser.add_argument("--border_size", type=float, default=0.0,
                        help="Flat border around the whole map [m] (default: 0)")
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="Obstacle difficulty 0-1, only used in randomized mode (default: 1.0)")
    parser.add_argument(
        "--mode", choices=["curriculum", "randomized"], default="curriculum",
        help=(
            "curriculum: column→terrain type, row→difficulty  "
            "[matches legged_gym curiculum()]; "
            "randomized: random terrain type per cell  "
            "(default: curriculum)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--format", choices=["png", "xml", "both"], default="xml",
        help=(
            "Output format: "
            "xml = MuJoCo box geoms (vertical walls, no interpolation artefacts); "
            "png = 16/8-bit heightmap PNG; "
            "both = save both  (default: xml)"
        ),
    )
    parser.add_argument("--bit_depth", type=int, choices=[8, 16], default=16,
                        help="PNG bit depth: 16 for higher precision (default: 16)")
    parser.add_argument("--slope_threshold", type=float, default=0.75,
                        help=(
                            "Slope threshold for edge correction in box-geom export, "
                            "matching convert_heightfield_to_trimesh in legged_gym "
                            "(default: 0.75). Set to 0 to disable."
                        ))
    parser.add_argument("--visualize", action="store_true",
                        help="Also save a colourised visualisation PNG (*_vis.png)")
    parser.add_argument("--output", type=str, default="terrain",
                        help="Output file base name without extension (default: terrain)")
    args = parser.parse_args()

    # ── validate ─────────────────────────────────────────────────────────────
    total = sum(args.terrain_proportions)
    if abs(total - 1.0) > 1e-3:
        print(f"Warning: terrain_proportions sum to {total:.4f} (expected 1.0)")

    labels = TERRAIN_NAMES[:len(args.terrain_proportions)]
    active = {n: v for n, v in zip(labels, args.terrain_proportions) if v > 0}

    print("=" * 60)
    print("Terrain configuration")
    print("=" * 60)
    print(f"  mode             : {args.mode}")
    print(f"  grid             : {args.num_rows} rows × {args.num_cols} cols")
    print(f"  cell size        : {args.terrain_length} m × {args.terrain_width} m")
    print(f"  horizontal scale : {args.horizontal_scale} m/pixel")
    print(f"  vertical scale   : {args.vertical_scale} m/unit")
    print(f"  border           : {args.border_size} m")
    print(f"  seed             : {args.seed}")
    if args.mode == "randomized":
        print(f"  difficulty       : {args.difficulty}")
    print(f"  active terrains  : {active if active else '(pit fallback)'}")
    print()

    # ── generate ─────────────────────────────────────────────────────────────
    gen = TerrainGenerator(
        terrain_proportions=args.terrain_proportions,
        terrain_length=args.terrain_length,
        terrain_width=args.terrain_width,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        horizontal_scale=args.horizontal_scale,
        vertical_scale=args.vertical_scale,
        border_size=args.border_size,
        difficulty=args.difficulty,
        mode=args.mode,
        seed=args.seed,
    )
    hf = gen.generate()

    nrows, ncols = hf.shape
    total_x_m = nrows * args.horizontal_scale
    total_y_m = ncols * args.horizontal_scale
    print(f"Height field : {nrows} × {ncols} pixels  "
          f"({total_x_m:.1f} m × {total_y_m:.1f} m)")
    print(f"Height stats : min={hf.min()}  max={hf.max()}  "
          f"unique values={len(np.unique(hf))}")

    base = os.path.splitext(args.output)[0]  # strip extension if user added one
    do_xml = args.format in ("xml", "both")
    do_png = args.format in ("png", "both")

    # ── save XML (box geoms) ──────────────────────────────────────────────────
    if do_xml:
        xml_path = base + ".xml"
        print(f"Extracting box geoms …", end=" ", flush=True)
        slope_thr = args.slope_threshold if args.slope_threshold > 0 else None
        boxes = heightmap_to_boxes(hf, args.horizontal_scale, args.vertical_scale,
                                   slope_threshold=slope_thr)
        n = save_mjcf_xml(boxes, xml_path, total_x_m, total_y_m)
        print(f"{n} boxes")
        print(f"Saved XML            : {xml_path}")

    # ── save PNG ──────────────────────────────────────────────────────────────
    if do_png:
        png_path = base + ".png"
        h_min_m, h_range_m = save_heightmap_png(
            hf, png_path, args.vertical_scale, args.bit_depth)
        print(f"Saved {args.bit_depth}-bit PNG     : {png_path}")
        print(f"Height range         : {h_min_m:.4f} m  →  {h_min_m + h_range_m:.4f} m  "
              f"(range = {h_range_m:.4f} m)")

    # ── optional visualisation ────────────────────────────────────────────────
    if args.visualize:
        vis_path = base + "_vis.png"
        print(f"Saving visualisation : {vis_path} …", end=" ")
        ok = save_visualisation(hf, vis_path, args.horizontal_scale,
                                args.num_rows, args.num_cols,
                                args.terrain_length, args.terrain_width)
        if ok:
            print("done")

    # ── MuJoCo configuration hints ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("MuJoCo usage")
    print("=" * 60)

    if do_xml:
        print(f"""
Box-geom XML  →  include or merge into your model:

  <include file="{os.path.basename(base)}.xml"/>

Or copy the <geom> elements from {os.path.basename(base)}.xml
into your existing <worldbody>.
""")

    if do_png:
        rx = total_y_m / 2
        ry = total_x_m / 2
        fname = os.path.basename(base) + ".png"
        print(f"""
Heightmap PNG  →  hfield asset + geom:

<asset>
  <hfield name="terrain" file="{fname}"
          size="{rx:.4f} {ry:.4f} {h_range_m:.4f} 0.1"/>
</asset>
<worldbody>
  <geom name="terrain" type="hfield" hfield="terrain"
        pos="{rx:.4f} {ry:.4f} {h_min_m:.4f}"
        contype="1" conaffinity="1"/>
</worldbody>
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
