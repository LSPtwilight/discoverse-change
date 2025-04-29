import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import time

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

def get_camera_quaternion(camera_position, object_position):
    """
    计算相机看向物体的旋转四元数 (xyzw 格式)
    
    :param camera_position: np.array([x, y, z])，相机位置
    :param object_position: np.array([x, y, z])，物体位置
    :return: np.array([x, y, z, w])，相机旋转的四元数
    """
    # 计算相机的朝向向量（物体 - 相机）
    forward_vector = np.array(object_position) - np.array(camera_position)
    forward_vector /= np.linalg.norm(forward_vector)  # 归一化

    # 定义参考的“前向”方向（Mujoco 期望的方向是世界坐标系的 -Z 轴）
    world_forward = np.array([0, 0, -1])  

    # 计算旋转矩阵，使 world_forward 旋转到 forward_vector
    rotation_matrix = Rotation.from_rotvec(np.cross(world_forward, forward_vector) *
                                    np.arccos(np.dot(world_forward, forward_vector)))

    # 转换为四元数（Scipy 默认格式是 xyzw）
    quaternion_xyzw = rotation_matrix.as_quat()

    return quaternion_xyzw
#################################################################################
#################################################################################




class AirbotPlayCfg(BaseConfig):
    mjcf_file_path = "mjcf/airbot_play_floor.xml"
    decimation     = 4
    timestep       = 0.005
    sync           = True
    headless       = False
    init_key       = "home"
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    obs_rgb_cam_id  = None
    rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict = {
        "arm_base"  : "airbot_play/arm_base.ply",
        "link1"     : "airbot_play/link1.ply",
        "link2"     : "airbot_play/link2.ply",
        "link3"     : "airbot_play/link3.ply",
        "link4"     : "airbot_play/link4.ply",
        "link5"     : "airbot_play/link5.ply",
        "link6"     : "airbot_play/link6.ply",
        "left"      : "airbot_play/left.ply",
        "right"     : "airbot_play/right.ply",
    }

class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        self.nj = 7
        super().__init__(config)

    def post_load_mjcf(self):
        try:
            self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.nj]
            self.init_joint_ctrl = self.mj_model.key(self.config.init_key).ctrl[:self.nj]
        except KeyError as e:
            self.init_joint_pose = np.zeros(self.nj)
            self.init_joint_ctrl = np.zeros(self.nj)

        self.sensor_joint_qpos = self.mj_data.sensordata[:self.nj]
        self.sensor_joint_qvel = self.mj_data.sensordata[self.nj:2*self.nj]
        self.sensor_joint_force = self.mj_data.sensordata[2*self.nj:3*self.nj]
        self.sensor_endpoint_posi_local = self.mj_data.sensordata[3*self.nj:3*self.nj+3]
        self.sensor_endpoint_quat_local = self.mj_data.sensordata[3*self.nj+3:3*self.nj+7]
        self.sensor_endpoint_linear_vel_local = self.mj_data.sensordata[3*self.nj+7:3*self.nj+10]
        self.sensor_endpoint_gyro = self.mj_data.sensordata[3*self.nj+10:3*self.nj+13]
        self.sensor_endpoint_acc = self.mj_data.sensordata[3*self.nj+13:3*self.nj+16]

        if "new_object" in self.config.gs_model_dict:
            new_object_path = self.config.gs_model_dict["new_object"]
            print(f"新物体 {new_object_path} 已加载")

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        print("    arm .qpos  = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("    arm .qvel  = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))
        print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))
        print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))
        print("    sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("    sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0
        self.mj_data.ctrl[:self.nj] = np.clip(action[:self.nj], self.mj_model.actuator_ctrlrange[:self.nj,0], self.mj_model.actuator_ctrlrange[:self.nj,1])

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time" : self.mj_data.time,
            "jq"   : self.sensor_joint_qpos.tolist(),
            "jv"   : self.sensor_joint_qvel.tolist(),
            "jf"   : self.sensor_joint_force.tolist(),
            "ep"   : self.sensor_endpoint_posi_local.tolist(),
            "eq"   : self.sensor_endpoint_quat_local.tolist(),
            "img"  : self.img_rgb_obs_s,
            "depth" : self.img_depth_obs_s
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

    
    def get_camera_image_direct(self, camera_name, changed_xyz=[-0.324, 0.697, 1.02], lookat_position=[0, 0, 0], width=640, height=480):
        # 确保 Gaussian Renderer 被正确启用
        if self.config.use_gaussian_renderer:
            if not hasattr(self, "gs_renderer"):
                raise RuntimeError("Gaussian Renderer is not properly initialized.")

            # 获取相机 ID
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id == -1:
                raise ValueError(f"Camera '{camera_name}' not found in the Mujoco model.")

            # 设置新的相机位置
            new_camera_position = np.array(changed_xyz)
            
            # 计算相机朝向（四元数）
            quat_xyzw = get_camera_quaternion(new_camera_position, np.array(lookat_position))
            
            # 设置相机位置和方向
            self.gs_renderer.set_camera_pose(new_camera_position, quat_xyzw)

            # 设置相机视野（FOV）
            fovy_radians = np.deg2rad(self.mj_model.cam_fovy[cam_id])
            self.gs_renderer.set_camera_fovy(fovy_radians)

            # 渲染图像
            image = self.gs_renderer.render(render_depth=False)
            #print(f"[Gaussian Renderer] Captured image from camera '{camera_name}' with resolution ({width}x{height})")

            return image,new_camera_position,quat_xyzw
        
    def get_camera_depth_direct(self, camera_name, changed_xyz=[-0.324, 0.697, 1.02], lookat_position=[0, 0, 0], width=640, height=480):
        # 确保 Gaussian Renderer 被正确启用
        if self.config.use_gaussian_renderer:
            if not hasattr(self, "gs_renderer"):
                raise RuntimeError("Gaussian Renderer is not properly initialized.")

            # 获取相机 ID
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id == -1:
                raise ValueError(f"Camera '{camera_name}' not found in the Mujoco model.")

            # 设置新的相机位置
            new_camera_position = np.array(changed_xyz)
            
            # 计算相机朝向（四元数）
            quat_xyzw = get_camera_quaternion(new_camera_position, np.array(lookat_position))
            
            # 设置相机位置和方向
            self.gs_renderer.set_camera_pose(new_camera_position, quat_xyzw)

            # 设置相机视野（FOV）
            fovy_radians = np.deg2rad(self.mj_model.cam_fovy[cam_id])
            self.gs_renderer.set_camera_fovy(fovy_radians)

            # 渲染图像
            image_depth = self.gs_renderer.render(render_depth=True)
            #print(f"[Gaussian Renderer] Captured image from camera '{camera_name}' with resolution ({width}x{height})")

            return image_depth,new_camera_position,quat_xyzw


    
    def close(self):
        # 销毁渲染器上下文
        if hasattr(self, "render_context"):
            self.render_context.close()
        
        # 如果有其他需要清理的资源，可以在这里加入
        # 例如：关闭模型、清理模拟器等
        
        print("Simulation resources cleaned up.")


if __name__ == "__main__":
    cfg = AirbotPlayCfg()
    exec_node = AirbotPlayBase(cfg)

    exec_node.reset()

    # 定义保存图片的目录
    output_dir = 'output_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # MJCF文件路径
    mjcf_file_path = '/home/bigai/DISCOVERSE/models/mjcf/airbot_play_floor.xml'

    # 定义位置变化范围
    y_start = -2.5
    y_end = -0.5
    step = 0.1


# #########################################################################
#     ##在每次修改位置时，拍照并保存
#     for y in np.arange(y_start, y_end + step, step):

#                 # 初始化并重置模拟器
#         cfg = AirbotPlayCfg()
#         exec_node = AirbotPlayBase(cfg)
#         exec_node.reset()

#         # 载入 MJCF 文件
#         tree = ET.parse(mjcf_file_path)
#         root = tree.getroot()

#         # 获取相机元素
#         camera = root.find(".//camera[@name='custom_camera']")
#         # 动态更新相机位置
#         camera.set('pos', f"0 {y} 1")  # pos 属性值为 "x y z"

#         print(f"Updated camera position to: 0 {y} 1")
        
#         # 保存更新后的 MJCF 文件
#         tree.write(mjcf_file_path)

#         # 重置模拟器
#         exec_node.resetState()
#         time.sleep(0.1)

#         # 每次改变位置后，拍摄并保存图片
#         img = exec_node.get_camera_image(camera_name="custom_camera", width=640, height=480)
        
#         # 保存图像
#         img_filename = f"{output_dir}/camera_image_y{y:.1f}.png"
#         plt.imsave(img_filename, img)


###################################################################
    # # 获取自定义相机的图片
    # img = exec_node.get_camera_image("custom_camera")

    # ######################
    # # 显示图片
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()

    obs = exec_node.reset()
    action = exec_node.init_joint_pose[:exec_node.nj]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)