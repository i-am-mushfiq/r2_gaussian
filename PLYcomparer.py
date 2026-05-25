import open3d as o3d
import numpy as np
from pathlib import Path

def compare_ply_files(file1, file2, tolerance=1e-6):
    report = {}

    # Load point clouds / meshes
    ply1 = o3d.io.read_triangle_mesh(file1)
    ply2 = o3d.io.read_triangle_mesh(file2)

    # --- Vertices ---
    verts1 = np.asarray(ply1.vertices)
    verts2 = np.asarray(ply2.vertices)
    report["num_vertices_file1"] = verts1.shape[0]
    report["num_vertices_file2"] = verts2.shape[0]

    if verts1.shape == verts2.shape:
        verts_equal = np.allclose(verts1, verts2, atol=tolerance)
        report["vertices_match"] = verts_equal
        if not verts_equal:
            diff = np.abs(verts1 - verts2)
            report["max_vertex_diff"] = np.max(diff)
            report["mean_vertex_diff"] = np.mean(diff)
    else:
        report["vertices_match"] = False
        report["vertex_shape_file1"] = verts1.shape
        report["vertex_shape_file2"] = verts2.shape

    # --- Faces ---
    faces1 = np.asarray(ply1.triangles)
    faces2 = np.asarray(ply2.triangles)
    report["num_faces_file1"] = faces1.shape[0]
    report["num_faces_file2"] = faces2.shape[0]

    if faces1.shape == faces2.shape:
        report["faces_match"] = np.array_equal(faces1, faces2)
    else:
        report["faces_match"] = False
        report["face_shape_file1"] = faces1.shape
        report["face_shape_file2"] = faces2.shape

    # --- Colors ---
    has_colors1 = ply1.has_vertex_colors()
    has_colors2 = ply2.has_vertex_colors()
    report["has_colors_file1"] = has_colors1
    report["has_colors_file2"] = has_colors2

    if has_colors1 and has_colors2:
        colors1 = np.asarray(ply1.vertex_colors)
        colors2 = np.asarray(ply2.vertex_colors)
        if colors1.shape == colors2.shape:
            colors_equal = np.allclose(colors1, colors2, atol=tolerance)
            report["colors_match"] = colors_equal
            if not colors_equal:
                color_diff = np.abs(colors1 - colors2)
                report["max_color_diff"] = np.max(color_diff)
                report["mean_color_diff"] = np.mean(color_diff)
        else:
            report["colors_match"] = False
            report["color_shape_file1"] = colors1.shape
            report["color_shape_file2"] = colors2.shape
    else:
        report["colors_match"] = has_colors1 == has_colors2

    # --- Normals ---
    has_normals1 = ply1.has_vertex_normals()
    has_normals2 = ply2.has_vertex_normals()
    report["has_normals_file1"] = has_normals1
    report["has_normals_file2"] = has_normals2

    if has_normals1 and has_normals2:
        normals1 = np.asarray(ply1.vertex_normals)
        normals2 = np.asarray(ply2.vertex_normals)
        if normals1.shape == normals2.shape:
            normals_equal = np.allclose(normals1, normals2, atol=tolerance)
            report["normals_match"] = normals_equal
            if not normals_equal:
                normal_diff = np.abs(normals1 - normals2)
                report["max_normal_diff"] = np.max(normal_diff)
                report["mean_normal_diff"] = np.mean(normal_diff)
        else:
            report["normals_match"] = False
            report["normal_shape_file1"] = normals1.shape
            report["normal_shape_file2"] = normals2.shape
    else:
        report["normals_match"] = has_normals1 == has_normals2

    # --- Optional: Hausdorff distance ---
    try:
        d1 = ply1.compute_point_cloud_distance(ply2)
        d2 = ply2.compute_point_cloud_distance(ply1)
        report["hausdorff_distance"] = max(max(d1), max(d2))
    except:
        report["hausdorff_distance"] = None

    return report


# --- Example usage ---
file1 = Path(__file__).parent / "output/Brain2_training/point_cloud/iteration_30000/point_cloud.ply"
file2 = Path(__file__).resolve().parent.parent / "gaussian-splatting" / "output" / "room" / "input.ply"

comparison_report = compare_ply_files(file1, file2)
for k, v in comparison_report.items():
    print(f"{k}: {v}")
