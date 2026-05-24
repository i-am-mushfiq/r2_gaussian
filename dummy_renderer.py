import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import time
from pathlib import Path

start = time.time()

# ------------------------
# Configuration
# ------------------------
category = 'brain'
save_dir = category + '/'
os.makedirs(save_dir, exist_ok=True)

pickle_path = Path(__file__).parent / "output/Brain2_training/point_cloud/iteration_30000/point_cloud.pickle"

# ------------------------
# Load point cloud
# ------------------------
with open(pickle_path, "rb") as f:
    pc = pickle.load(f)

xyz = pc['xyz']             # Nx3
density = pc['density']      # N
scale = pc['scale']          # Nx3

# ------------------------
# Create voxel grid
# ------------------------
grid_size = 128
volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

# Normalize xyz to grid
min_coords = xyz.min(axis=0)
max_coords = xyz.max(axis=0)
xyz_norm = ((xyz - min_coords) / (max_coords - min_coords) * (grid_size-1)).astype(int)

# Helper: simple Gaussian kernel
def gaussian_kernel(size, scale_vec):
    ax = np.linspace(-1, 1, size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-((xx/scale_vec[0])**2 + (yy/scale_vec[1])**2 + (zz/scale_vec[2])**2))
    return kernel

# Fill voxel grid
for i in range(0, len(xyz), 5):  # subsample for speed
    x, y, z = xyz_norm[i]
    s = max(1, int(np.ceil(np.mean(scale[i]*5))))
    kernel = gaussian_kernel(2*s+1, scale[i])
    
    # Compute voxel slices
    xs = slice(max(0, x-s), min(grid_size, x+s+1))
    ys = slice(max(0, y-s), min(grid_size, y+s+1))
    zs = slice(max(0, z-s), min(grid_size, z+s+1))
    
    # Crop kernel to match volume patch
    kx0 = max(0, 0 - (x - s))
    kx1 = kx0 + (xs.stop - xs.start)
    ky0 = max(0, 0 - (y - s))
    ky1 = ky0 + (ys.stop - ys.start)
    kz0 = max(0, 0 - (z - s))
    kz1 = kz0 + (zs.stop - zs.start)
    
    vol_patch = volume[xs, ys, zs]
    vol_patch += density[i] * kernel[kx0:kx1, ky0:ky1, kz0:kz1]
    volume[xs, ys, zs] = vol_patch

print(f"Voxelization done: {time.time()-start:.2f} s")

# ------------------------
# Apply marching cubes
# ------------------------
sigma = 0.6
threshold = sigma * volume.min() + (1 - sigma) * volume.max()
verts, faces, _, _ = measure.marching_cubes(volume, threshold)

# ------------------------
# Plot 3D mesh
# ------------------------
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
alpha = 0.3
mesh = Poly3DCollection(verts[faces], alpha=alpha)
face_color = [0.5, 0.5, 0.5]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_zlim(0, grid_size)
ax.set_box_aspect([1,1,1])

# ------------------------
# Save multiple views
# ------------------------
proj_num = 4
angle_interval = 360 / proj_num
elevation = 10

series_save_dir = osp.join(save_dir, f"elevation_{elevation}_sigma_{sigma}_alpha_{alpha}/")
os.makedirs(series_save_dir, exist_ok=True)

for i in tqdm(range(proj_num)):
    angle = angle_interval * i
    ax.view_init(elev=elevation, azim=angle)
    plt.savefig(f'{series_save_dir}angle_{angle}.png')

print(f"Rendering used time: {time.time()-start:.2f} s")
