import pickle
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent / "output/Brain2_training/point_cloud/iteration_30000"

# Load the pickle
with open(_HERE / "point_cloud.pickle", "rb") as f:
    pc = pickle.load(f)

# Extract 'xyz' array
points = pc['xyz']

# Save as binary
points.astype(np.float32).tofile(_HERE / "point_cloud.bin")

print("Conversion done. point_cloud.bin created.")
