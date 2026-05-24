# png_stack_to_npy.py
import sys, os
from pathlib import Path
import numpy as np
from skimage import io
from skimage.transform import resize
from scipy.ndimage import zoom

def load_png_stack(folder):
    p = Path(folder)
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff')])
    if not files:
        raise SystemExit("No image files found in folder: "+str(folder))
    imgs = [io.imread(str(f)) for f in files]
    # if RGBA/RGB -> convert to luminance
    stack = []
    for im in imgs:
        if im.ndim == 3:
            # convert to gray; use luminosity
            im = im[..., :3]  # drop alpha if present
            im = (0.2989*im[...,0] + 0.5870*im[...,1] + 0.1140*im[...,2])
        stack.append(im)
    vol = np.stack(stack, axis=0)  # shape (Z, H, W)
    return vol.astype(np.float32)

def normalize_clip(vol, pmin=1, pmax=99):
    lo = np.percentile(vol, pmin)
    hi = np.percentile(vol, pmax)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-12)
    return vol

def resample_to_isotropic(vol, spacing=None, target_size=256):
    # vol shape (Z,H,W)
    if spacing is None:
        # assume isotropic or unknown — just do resize to cube
        vol_resized = resize(vol, (target_size, target_size, target_size), order=1, preserve_range=True, anti_aliasing=True)
        return vol_resized
    # spacing = (sz, sy, sx) in mm: resample voxels so mm per voxel matches
    sz, sy, sx = spacing
    # desired isotropic voxel mm = average physical size (or choose min)
    target_mm = min(sx, sy, sz)
    zoom_f = (sz/target_mm, sy/target_mm, sx/target_mm)
    vol_iso = zoom(vol, zoom_f, order=1)  # may produce large array
    # now crop/pad/rescale to target_size
    vol_resized = resize(vol_iso, (target_size, target_size, target_size), order=1, preserve_range=True, anti_aliasing=True)
    return vol_resized

def main(in_folder, out_npy, target_size=256, spacing=None):
    vol = load_png_stack(in_folder)  # (Z,H,W)
    print("Loaded volume shape (Z,H,W):", vol.shape)
    vol = normalize_clip(vol, pmin=1, pmax=99)
    vol = resample_to_isotropic(vol, spacing=spacing, target_size=target_size)
    # reorder to (Z,Y,X) -> pipeline probably expects (D,H,W). Save as float32
    vol = vol.astype(np.float32)
    np.save(out_npy, vol)
    print("Saved processed volume:", out_npy, "shape:", vol.shape, "min/max:", vol.min(), vol.max())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python png_stack_to_npy.py <png_folder> <out.npy> [target_size] [spacing_z spacing_y spacing_x]")
        sys.exit(1)
    folder = sys.argv[1]
    out = sys.argv[2]
    ts = int(sys.argv[3]) if len(sys.argv) >= 4 else 256
    spacing = None
    if len(sys.argv) == 7:
        spacing = (float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
    main(folder, out, target_size=ts, spacing=spacing)
