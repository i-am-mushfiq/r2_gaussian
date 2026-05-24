import numpy as np
import os
from pathlib import Path

_HERE = Path(__file__).parent
vol_path = _HERE / "data_generator/synthetic_dataset/cone_ntrain_75_angle_360/Brain2_PNG_cone/vol_gt.npy"
output_path = _HERE / "data_generator/synthetic_dataset/cone_ntrain_75_angle_360/Brain2_PNG_cone/init_Brain2_PNG_cone.npy"

n_points = 50000
dVoxel = np.array([2.0, 2.0, 2.0])
offOrigin = np.array([0, 0, 0])

# --- LOAD VOLUME ---
vol = np.load(vol_path)
shape = np.array(vol.shape, dtype=np.float32)

# --- THRESHOLD BASED ON PERCENTILE ---
non_zero = vol[vol > 0]
density_thresh = np.percentile(non_zero, 50)  # 50th percentile
mask = vol > density_thresh
valid_indices = np.argwhere(mask)

if len(valid_indices) < n_points:
    replace_flag = True
else:
    replace_flag = False

# --- SAMPLE POINTS ---
sampled_indices = valid_indices[np.random.choice(len(valid_indices), n_points, replace=replace_flag)]
positions = sampled_indices * dVoxel - shape/2 + offOrigin
positions += np.random.normal(0, 0.1, positions.shape)  # small jitter

# --- SCALE DENSITIES ---
densities = vol[sampled_indices[:,0], sampled_indices[:,1], sampled_indices[:,2]] * 5.0 + 0.01

# --- SAVE POINT CLOUD ---
init_cloud = np.concatenate([positions, densities[:,None]], axis=-1)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, init_cloud)

print(f"Initial point cloud saved to {output_path} with shape {init_cloud.shape}")
