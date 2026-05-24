import numpy as np
import tigre
import tigre.algorithms as algs

print("TIGRE imported from:", tigre.__file__)

# Create an empty geometry object
geo = tigre.geometry()   # must be an instance of geometry
geo.nVoxel = [64, 64, 64]       # small volume
geo.sVoxel = [2.0, 2.0, 2.0]
geo.DSD = 7.0
geo.DSO = 5.0
geo.nDetector = [64, 64]
geo.sDetector = [4.0, 4.0]
geo.offOrigin = [0, 0, 0]
geo.offDetector = [0, 0]

# Dummy volume
vol = np.ones(geo.nVoxel, dtype=np.float32)

# Projection angles
angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)

# Forward projection
try:
    print("Running forward projection (Ax)...")
    projs = tigre.Ax(vol, geo, angles)
    print("Forward projection done. Projections shape:", projs.shape)
except Exception as e:
    print("Error during Ax:", e)

# FDK reconstruction
try:
    print("Running FDK reconstruction...")
    vol_rec = algs.fdk(projs, geo, angles)
    print("FDK reconstruction done. Volume shape:", vol_rec.shape)
except Exception as e:
    print("Error during FDK:", e)
