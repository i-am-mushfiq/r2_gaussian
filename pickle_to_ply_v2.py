"""
Script: pickle_to_gaussian_splat_ply_v4.py
Purpose: Convert point_cloud.pickle to a full Gaussian splat PLY compatible with r2_gaussian viewer.
Requirements: numpy, open3d, json
"""

import pickle
import numpy as np
import os
import json
from pathlib import Path

# ----------------------------
# User configuration
_HERE = Path(__file__).parent / "output/Brain2_training/point_cloud/iteration_30000"
pickle_path = _HERE / "point_cloud.pickle"
ply_output_path = _HERE / "point_cloud.ply"
meta_json_path = _HERE / "meta_data.json"
# ----------------------------

# Check files exist
for path in [pickle_path, meta_json_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load pickle
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Load JSON metadata
with open(meta_json_path, "r") as f:
    meta = json.load(f)

# Extract bounding box and voxel size
bbox_min = np.array(meta["bbox"][0], dtype=np.float32)
bbox_max = np.array(meta["bbox"][1], dtype=np.float32)
sVoxel = np.array(meta["scanner"]["sVoxel"], dtype=np.float32)

# ----------------------------
# Extract points
if isinstance(data, dict):
    if "xyz" in data:
        points = np.array(data["xyz"], dtype=np.float32)
    elif "points" in data:
        points = np.array(data["points"], dtype=np.float32)
    else:
        # fallback: use first numpy array found
        found = False
        for k, v in data.items():
            if isinstance(v, (np.ndarray, list)):
                points = np.array(v, dtype=np.float32)
                found = True
                print(f"Using '{k}' as point coordinates")
                break
        if not found:
            raise KeyError("No valid points array found in pickle")

    N = points.shape[0]

    # Scale/radius and density
    radii = np.array(data["scale"], dtype=np.float32) if "scale" in data else np.ones(N, dtype=np.float32) * 0.01
    density = np.array(data["density"], dtype=np.float32) if "density" in data else np.ones(N, dtype=np.float32) * 1.0

    # Colors
    if "colors" in data:
        colors = np.array(data["colors"], dtype=np.uint8)
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
    else:
        colors = np.ones((N,3), dtype=np.uint8) * 128
else:
    points = np.array(data, dtype=np.float32)
    N = points.shape[0]
    radii = np.ones(N, dtype=np.float32) * 0.01
    density = np.ones(N, dtype=np.float32) * 1.0
    colors = np.ones((N,3), dtype=np.uint8) * 128

# ----------------------------
# Normalize points to bbox
scale = bbox_max - bbox_min
points_normalized = bbox_min + points * scale

# ----------------------------
# Write ASCII PLY with Gaussian splat properties
with open(ply_output_path, "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {N}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("property float radius\n")
    f.write("property float density\n")
    f.write("end_header\n")
    for i in range(N):
        x, y, z = points_normalized[i]
        r, g, b = colors[i]
        rad = radii[i]
        dens = density[i]
        f.write(f"{x} {y} {z} {r} {g} {b} {rad} {dens}\n")

print(f"Saved Gaussian splat PLY to {ply_output_path}")
