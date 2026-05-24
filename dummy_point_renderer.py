import pickle
import open3d as o3d
import numpy as np
from pathlib import Path

# Load the pickle
with open(Path(__file__).parent / "output/Brain2_training/point_cloud/iteration_30000/point_cloud.pickle", "rb") as f:
    pc = pickle.load(f)

points = pc['xyz']  # N x 3 array

# Optional: use density as color
densities = pc['density'] if 'density' in pc else None
if densities is not None:
    densities = (densities - densities.min()) / (densities.max() - densities.min())  # normalize
    colors = np.stack([densities]*3, axis=1)
else:
    colors = None

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])