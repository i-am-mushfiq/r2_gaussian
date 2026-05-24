import open3d as o3d
import numpy as np
from pathlib import Path

# --- Load original ASCII PLY ---
_HERE = Path(__file__).parent
input_file = _HERE / "output/Brain2_training/point_cloud/iteration_30000/point_cloud.ply"
output_file = _HERE / "output/Brain2_training/point_cloud/iteration_30000/point_cloud_ready.ply"


# Open3D may ignore custom attributes; we only read vertices and colors
pcd = o3d.io.read_point_cloud(input_file, format='ply')  # reads x, y, z, r, g, b

# Check if colors exist
if not pcd.has_colors():
    print("No colors found. Assigning default white color.")
    colors = np.ones((np.asarray(pcd.points).shape[0], 3))  # white
    pcd.colors = o3d.utility.Vector3dVector(colors)

# Optionally generate normals if missing
if not pcd.has_normals():
    print("Normals not found. Generating dummy normals...")
    points = np.asarray(pcd.points)
    # Simple dummy normals (all pointing up)
    normals = np.tile([0.0, 0.0, 1.0], (points.shape[0], 1))
    pcd.normals = o3d.utility.Vector3dVector(normals)

# --- Save as binary PLY compatible with viewer ---
o3d.io.write_point_cloud(output_file, pcd, write_ascii=False)
print(f"Converted PLY saved as: {output_file}")
