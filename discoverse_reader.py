#     XMY    25-03-26
#     discoverse loader

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scipy.spatial.transform import Rotation 
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

def get_colmap_qvec_tvec(position, quat_xyzw):
    """
    将 Mujoco 坐标下的 T_c2w 和 R_c2w（quat_xyzw）转换为 COLMAP / 3DGS 格式的
    R_w2c 四元数 (qvec: [qw, qx, qy, qz]) 和 T_w2c (tvec)

    :param position: np.array([x, y, z])，相机在 Mujoco 世界坐标下的位置（T_c2w）
    :param quat_xyzw: np.array([x, y, z, w])，Mujoco 的四元数，表示 R_c2w
    :return: (qvec [qw, qx, qy, qz], tvec)
    """
    # 坐标轴变换矩阵：Mujoco → COLMAP
    R_m2c = np.diag([1, -1, -1])

    # Step 1: 获取 R_c2w（Mujoco）
    R_c2w_mujoco = Rotation.from_quat(quat_xyzw).as_matrix()

    # Step 2: 转换到 COLMAP 坐标系下
    R_c2w_colmap = R_m2c @ R_c2w_mujoco @ R_m2c.T
    R_w2c_colmap = R_c2w_colmap.T

    # Step 3: 转换相机位置（T_c2w）到 COLMAP 坐标系
    T_c2w_colmap = R_m2c @ position

    # Step 4: 计算 T_w2c
    T_w2c_colmap = - R_w2c_colmap @ T_c2w_colmap

    # Step 5: 获取四元数（wxyz 顺序）
    q_xyzw = Rotation.from_matrix(R_w2c_colmap).as_quat()
    qvec = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # xyzw → wxyz

    return qvec, T_w2c_colmap


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# 读取并处理Colmap中的相机信息
def readDiscoverseColmap(images_folder, test_cam_names_list):

    parent_dir = os.path.dirname(images_folder)
    json_path = os.path.join(parent_dir, "mujoco_cam_infos.json")  

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            mujoco_cam_infos = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"JsonNotFound: {json_path}")
    
    uid = 1
    width = 640
    height = 480
    FovY = 72.02
    FovX = 2 * np.arctan(np.tan(np.radians(FovY) / 2) * (width / height))
    depth_params = None
    depth_path = ""

    cam_infos = []
    for cam in mujoco_cam_infos["cameras"]:
        position = cam["cam_pos"]
        quat_xyzw = cam["quat_xyzw"]
        image_name = cam["image_name"]
        
        
        # 转换为3DGS坐标系
        qvec, tvec = get_colmap_qvec_tvec(position, quat_xyzw)
        R_1 = np.transpose(qvec2rotmat(qvec))
        T_1 = np.array(tvec)

        image_path = os.path.join(images_folder, image_name)
        cam_info = CameraInfo(uid=uid, R=R_1, T=T_1, FovY=np.radians(FovY), FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)
    
    return cam_infos

def readDiscoverseScene(path,eval):
    
    ply_path = os.path.join(path, "points3D.ply")
    num_pts = 100_000
    print(f"Generating random point cloud ({num_pts})...")  
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    json_path = os.path.join(path, "mujoco_cam_infos.json")  

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            mujoco_cam_infos = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"JsonNotFound: {json_path}")

    # 直接从JSON获取所有相机对应的图片名并排序
    all_img_names = sorted([cam["image_name"] for cam in mujoco_cam_infos["cameras"]])  # 假设字段是img_name

    if eval:
        llffhold = 7  
        test_img_names = [name for idx, name in enumerate(all_img_names) if idx % llffhold == 0]
        print(f"Selected {len(test_img_names)} test images from {len(all_img_names)} total")
    else:
        test_img_names = []
    
    images_folder=os.path.join(path,"input")
    cam_infos_unsorted=readDiscoverseColmap(images_folder=images_folder,test_cam_names_list=test_img_names)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info




############################################################################
##############################################################################