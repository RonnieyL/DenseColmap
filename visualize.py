#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from scene.colmap_loader import (
    read_intrinsics_binary,
    read_extrinsics_binary,
    read_points3D_binary,
    qvec2rotmat,
    rotmat2qvec,
)

def find_sparse0(dir_path: Path) -> Path:
    # if dir_path is already sparse/0
    if (dir_path / "cameras.bin").exists() and (dir_path / "images.bin").exists():
        return dir_path
    # otherwise look for dir_path/sparse/0
    candidate = dir_path / "sparse" / "0"
    if (candidate / "cameras.bin").exists():
        return candidate
    raise FileNotFoundError(f"Could not find sparse/0 under {dir_path!r}")

def make_camera_frustum(K: np.ndarray, scale: float=0.1) -> o3d.geometry.LineSet:
    # build a tiny frustum in camera coordinates
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    # assume image width/height = 2*cx,2*cy
    w, h = 2*cx, 2*cy
    z = scale
    corners = np.array([
        [(0-cx)/fx*z, (0-cy)/fy*z, z],
        [(w-cx)/fx*z, (0-cy)/fy*z, z],
        [(w-cx)/fx*z, (h-cy)/fy*z, z],
        [(0-cx)/fx*z, (h-cy)/fy*z, z],
    ])
    center = np.zeros((1,3))
    pts = np.vstack([center, corners])
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    fr = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    fr.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in lines])
    return fr

def visualize(model_dir: Path, frustum_scale: float):
    sparse0 = find_sparse0(model_dir)
    cams = read_intrinsics_binary(sparse0/"cameras.bin")
    imgs = read_extrinsics_binary(sparse0/"images.bin")
    xyz, rgb, _ = read_points3D_binary(sparse0/"points3D.bin")

    # build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64)/255.0)

    # build frustums
    frustums = []
    for img in imgs.values():
        cam = cams[img.camera_id]
        K = np.array([[cam.params[0],0,cam.params[2]],
                      [0,cam.params[1],cam.params[3]],
                      [0,0,1]],dtype=np.float64)
        fr = make_camera_frustum(K, scale=frustum_scale)

        # COLMAP qvec/tvec give world→cam, so invert to get cam→world:
        R_cam2w = qvec2rotmat(img.qvec).T
        t_cam2w = -R_cam2w @ img.tvec
        T = np.eye(4)
        T[:3,:3] = R_cam2w
        T[:3, 3] = t_cam2w
        fr.transform(T)
        frustums.append(fr)

    o3d.visualization.draw_geometries([pcd, *frustums])

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visualize COLMAP sparse model (cameras, images, points)"
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to scene root or the sparse/0 folder"
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.2,
        help="Size of frusta (distance to image plane)"
    )
    args = p.parse_args()
    visualize(args.model_dir, args.scale)
