"""
Script: pickle_to_gaussian_splat_ply_v3.py
Purpose: Convert point_cloud.pickle to a full Gaussian splat PLY compatible with r2_gaussian viewer.
Requirements: numpy, open3d
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

bbox_min = np.array(meta["bbox"][0])
bbox_max = np.array(meta["bbox"][1])
sVoxel = np.array(meta["scanner"]["sVoxel"])

# ----------------------------
# Extract points safely
if isinstance(data, dict):
    if "xyz" in data and data["xyz"] is not None:
        points = np.array(data["xyz"])
    elif "points" in data and data["points"] is not None:
        points = np.array(data["points"])
    else:
        raise ValueError("No 'xyz' or 'points' key found in the pickle data.")
    N = points.shape[0]
    print("Number of points:", N)

    # Extract radii
    if "scale" in data and data["scale"] is not None:
        radii = np.array(data["scale"], dtype=np.float32)
    else:
        radii = np.ones(N, dtype=np.float32) * 0.01

    # Extract density
    if "density" in data and data["density"] is not None:
        density = np.array(data["density"], dtype=np.float32)
    else:
        density = np.ones(N, dtype=np.float32)

    # Extract colors
    if "colors" in data and data["colors"] is not None:
        colors = np.array(data["colors"])
    else:
        colors = np.ones((N,3)) * 128
else:
    points = np.array(data)
    N = points.shape[0]
    radii = np.ones(N, dtype=np.float32) * 0.01
    density = np.ones(N, dtype=np.float32)
    colors = np.ones((N,3)) * 128

# Normalize points to bbox
scale = bbox_max - bbox_min
points_normalized = bbox_min + points * scale

# Ensure colors are uint8
if colors.max() <= 1.0:
    colors = (colors * 255).astype(np.uint8)
else:
    colors = colors.astype(np.uint8)

# Ensure radii and density are float32
radii = radii.astype(np.float32)
density = density.astype(np.float32)

# ----------------------------
# Write ASCII PLY
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
