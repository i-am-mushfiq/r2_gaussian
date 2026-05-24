"""
Script: pickle_to_ply.py
Purpose: Convert a point cloud stored in a .pickle file to a .ply file.
Requirements: open3d, numpy
"""

import pickle
import numpy as np
import open3d as o3d
import os
from pathlib import Path

# ----------------------------
# User configuration
_HERE = Path(__file__).parent / "output/Brain2_training/point_cloud/iteration_30000"
pickle_path = _HERE / "point_cloud.pickle"
ply_output_path = _HERE / "point_cloud.ply"
# ----------------------------

# Check if file exists
if not os.path.exists(pickle_path):
    raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

# Load pickle
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Inspect keys if dictionary
if isinstance(data, dict):
    print("Keys in pickle:", list(data.keys()))
    # Try to find points
    if "points" in data:
        points = np.array(data["points"])
    elif "xyz" in data:
        points = np.array(data["xyz"])
    else:
        # fallback: try first numpy array found
        found = False
        for k, v in data.items():
            if isinstance(v, (np.ndarray, list)):
                points = np.array(v)
                found = True
                print(f"Using '{k}' as point coordinates.")
                break
        if not found:
            raise ValueError("No valid points array found in pickle.")
else:
    # Assume pickle directly contains numpy array
    points = np.array(data)

print("Number of points:", points.shape[0])

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optional: Add color if available
if isinstance(data, dict) and "colors" in data:
    colors = np.array(data["colors"])
    pcd.colors = o3d.utility.Vector3dVector(colors)
else:
    # Default: gray color
    colors = np.ones_like(points) * 0.5
    pcd.colors = o3d.utility.Vector3dVector(colors)

# Save as PLY
o3d.io.write_point_cloud(ply_output_path, pcd)
print(f"Saved PLY file to {ply_output_path}")
