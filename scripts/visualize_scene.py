import os
import os.path as osp
import torch
import sys
from argparse import ArgumentParser
import numpy as np
import open3d as o3d
import matplotlib
from scipy.ndimage import zoom  # for volume upsampling

sys.path.append("./")
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.plot_utils import create_textured_camera, create_vol_mesh
from r2_gaussian.utils.graphics_utils import fov2focal
from r2_gaussian.utils.general_utils import t2a


def main(dataset: ModelParams, args):
    # Set up dataset
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg

    # Load volume and optionally upsample
    vol = np.load(osp.join(dataset.source_path, "vol_gt.npy"))
    if args.upsample_factor > 1:
        vol = zoom(vol, args.upsample_factor, order=1)  # linear interpolation

    # Create mesh using marching cubes
    vol_mesh = create_vol_mesh(
        vol,
        np.array(scanner_cfg["offOrigin"]),
        np.array(scanner_cfg["dVoxel"]),
        np.eye(3),
        level=args.mc_thresh,
    )

    # Optional: densify mesh with loop subdivision
    if args.subdivide > 0:
        vol_mesh = vol_mesh.subdivide_loop(number_of_iterations=args.subdivide)

    # Coordinate frame and bounding boxes
    vol_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scanner_cfg["sVoxel"][0] / 2,
        origin=scanner_cfg["offOrigin"],
    )
    vol_bbox = o3d.geometry.OrientedBoundingBox(
        center=scanner_cfg["offOrigin"],
        R=np.eye(3),
        extent=scanner_cfg["sVoxel"],
    )
    vol_bbox.color = np.array([1, 0, 0])

    unit_bbox = o3d.geometry.OrientedBoundingBox(
        center=[0, 0, 0], R=np.eye(3), extent=[2, 2, 2]
    )
    unit_bbox.color = np.array([0, 0, 1])

    # Cameras
    cams = []
    cmap = matplotlib.colormaps["viridis"]
    cam_scale = args.cam_scale
    n_proj = len(scene.train_cameras)
    for i_proj, camera in enumerate(scene.train_cameras):
        proj = t2a(camera.original_image)[0]
        K = np.array(
            [
                [fov2focal(camera.FoVx, proj.shape[1]), 0, proj.shape[1] / 2],
                [0, fov2focal(camera.FoVy, proj.shape[0]), proj.shape[0] / 2],
                [0, 0, 1],
            ]
        )
        w2c = np.eye(4)
        w2c[:3, :3] = t2a(camera.R.T)
        w2c[:3, 3] = t2a(camera.T)
        cam = create_textured_camera(
            K,
            w2c,
            cam_scale,
            cmap(i_proj / n_proj)[:3],
            proj.shape[1],
            proj.shape[0],
            f"{i_proj:03d}",
            proj,
        )
        cams += cam

    vis_assets = cams + [vol_mesh, vol_bbox, vol_coord, unit_bbox]
    o3d.visualization.draw_geometries(vis_assets, mesh_show_back_face=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualization script parameters")
    lp = ModelParams(parser)
    parser.add_argument("--mc_thresh", type=float, default=0.5,
                        help="Threshold for marching cubes; lower = denser mesh")
    parser.add_argument("--cam_scale", type=float, default=1.0,
                        help="Size of camera model for visualization")
    parser.add_argument("--upsample_factor", type=int, default=1,
                        help="Upsample factor for volume before meshing")
    parser.add_argument("--subdivide", type=int, default=0,
                        help="Number of Open3D loop subdivisions to densify mesh")
    args = parser.parse_args(sys.argv[1:])
    main(lp.extract(args), args)
