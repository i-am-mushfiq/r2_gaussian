import os
import pickle
import time
from pathlib import Path
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

start = time.time()

# -------------------------------
# GPU argument (optional)
def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", help="GPU to use")
    return parser

parser = config_parser()
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# -------------------------------
# Paths
category = 'brain'
pickle_path = Path(__file__).parent.parent / "output/Brain2_training/point_cloud/iteration_30000/point_cloud.pickle"

# Load pickle
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# -------------------------------
# Convert point cloud to voxel grid
points = data["xyz"]           # Nx3 array
density = data.get("density")  # optional

grid_size = 128
grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

min_vals = points.min(axis=0)
max_vals = points.max(axis=0)
norm_points = ((points - min_vals) / (max_vals - min_vals) * (grid_size - 1)).astype(int)

for i, pt in enumerate(norm_points):
    x, y, z = pt
    grid[x, y, z] = density[i] if density is not None else 1.0

# -------------------------------
# Generate mesh with marching cubes
verts, faces, _, _ = measure.marching_cubes(grid)

# -------------------------------
# Interactive 3D plot
plt.ion()  # Enable interactive mode
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

alpha = 0.5
mesh = Poly3DCollection(verts[faces], alpha=alpha)
mesh.set_facecolor([0.5, 0.5, 0.5])
ax.add_collection3d(mesh)

ax.set_xlim(0, grid.shape[0])
ax.set_ylim(0, grid.shape[1])
ax.set_zlim(0, grid.shape[2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()  # Opens interactive window
print(f"Used time: {time.time()-start:.2f} s")

# Keep the window open until manually closed
plt.pause(0.001)
input("Press Enter to exit...")
