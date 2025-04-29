import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco.viewer
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import json
import os
import shutil
import argparse
import multiprocessing as mp
import imageio
import cv2

from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, copypy2

#################
###  XMY
output_dir = 'output_images'
test_count=0

def get_random_camera_positions(arm_pose_center, num_samples=80, min_radius=0.15, max_radius=0.22, min_z=0.3):
    """
    均匀分布相机在上半球（避免靠近赤道）
    
    :param arm_pose_center: np.array([x, y, z])
    :param num_samples: 要生成的相机数
    :param min_radius: 最小半径（距离中心）
    :param max_radius: 最大半径
    :param min_z: 单位球面 z 的下限，用于避免相机太低（z ∈ [min_z, 1]）
    :return: List[np.array([x, y, z])]
    """
    camera_positions = []

    for _ in range(num_samples):
        # 均匀采样 z（控制采样范围在上半球的靠上区域）
        z_unit = np.random.uniform(min_z, 1.0)
        phi = np.random.uniform(0, 2 * np.pi)
        r_unit = np.sqrt(1 - z_unit ** 2)

        # 随机半径（控制相机远近）
        radius = np.random.uniform(min_radius, max_radius)

        # 转为笛卡尔坐标
        x = radius * r_unit * np.cos(phi)
        y = radius * r_unit * np.sin(phi)
        z = radius * z_unit

        camera_pos = arm_pose_center + np.array([x, y, z])
        camera_positions.append(camera_pos)

    return camera_positions

arm_pose_center = np.array([-0.34, 0.90 ,0.7195])
random_camera_positions = get_random_camera_positions(arm_pose_center, num_samples=120)

##########################################################################################################

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())

    def domain_randomization(self):
        # 随机 方块位置
        # self.mj_data.qpos[self.nj+1+0] += 2.*(np.random.random() - 0.5) * 0.12
        # self.mj_data.qpos[self.nj+1+1] += 2.*(np.random.random() - 0.5) * 0.08

        # # 随机 杯子位置
        # self.mj_data.qpos[self.nj+1+7+0] += 2.*(np.random.random() - 0.5) * 0.1
        # self.mj_data.qpos[self.nj+1+7+1] += 2.*(np.random.random() - 0.5) * 0.05

        # 随机 eye side 视角
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]
        print("1")

    def check_success(self):
        tmat_block = get_body_tmat(self.mj_data, "block_green")
        tmat_bowl = get_body_tmat(self.mj_data, "bowl_pink")
        return (abs(tmat_bowl[2, 2]) > 0.99) and np.hypot(tmat_block[0, 3] - tmat_bowl[0, 3], tmat_block[1, 3] - tmat_bowl[1, 3]) < 0.02


cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "ready"
cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"]   = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"

# #  添加物体
cfg.gs_model_dict["GGbond"] = "object/GGbond.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/block_place.xml"
# cfg.obj_list     = ["drawer_1", "drawer_2", "bowl_pink", "block_green"]
cfg.obj_list     = ["drawer_1", "drawer_2", "bowl_pink", "block_green","GGbond"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument("--save_segment", action="store_true", help="save segment videos")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/block_place")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.save_segment:
        cfg.obs_depth_cam_id = list(set(cfg.obs_rgb_cam_id + ([] if cfg.obs_depth_cam_id is None else cfg.obs_depth_cam_id)))
        from discoverse.randomain.utils import SampleforDR
        samples = SampleforDR(objs=cfg.obj_list[2:], robot_parts=cfg.rb_link_list, cam_ids=cfg.obs_rgb_cam_id, save_dir=os.path.join(save_dir, "segment"), fps=cfg.render_set["fps"])

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))
        
    arm_fik = AirbotPlayFIK(os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))

    trmat = Rotation.from_euler("xyz", [0., np.pi/2, 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 9
    max_time = 10.0 # seconds
    
    action = np.zeros(7)
    process_list = []

    move_speed = 0.75
    sim_node.reset()
    while sim_node.running:
        test_count+=1
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            if args.save_segment:
                samples.reset()

        try:
            if stm.trigger():
                if stm.state_idx == 0: # 伸到方块上方
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.1 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 1: # 伸到方块
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.028 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 2: # 抓住方块
                    sim_node.target_control[6] = 0.0
                elif stm.state_idx == 3: # 抓稳方块
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 4: # 提起来方块
                    tmat_tgt_local[2,3] += 0.07
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 5: # 把方块放到碗上空
                    tmat_plate = get_body_tmat(sim_node.mj_data, "bowl_pink")
                    tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0.0, 0.0, 0.13])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 6: # 降低高度 把方块放到碗上
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 7: # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 8: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            # traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

################################################
#########   XMY      ###########################
#       prograss  rendering
        # if test_count%2 == 0 :  
        #     image = sim_node.get_move_camera_image("testcamera",lookat_object_name="arm_pose")  
        #     img_filename = f"{output_dir}/camera_image_{test_count}.png"
            
        #     plt.imsave(img_filename, image)  
        #     print(f"Image saved: {img_filename}")  
        

        obs, _, _, _, _ = sim_node.step(action)

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)
            if args.save_segment:
                samples.sampling(sim_node)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(target=recoder_airbot_play, args=(save_path, act_lst, obs_lst, cfg))
                process.start()
                process_list.append(process)
                if args.save_segment:
                    seg_process = mp.Process(target=samples.save)
                    seg_process.start()
                    process_list.append(seg_process)

                data_idx += 1
                ori_cam_pos=np.array([-0.924,0.617,1.42])
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:

                    if data_idx >= data_set_size:

                        #### XMY #################
                        cameras_info = []
                        image_RBG_dir   = os.path.join(output_dir, "input")
                        image_depth_dir = os.path.join(output_dir, "depth")
                        depth_png_dir = os.path.join(output_dir, "depth_png")
                        os.makedirs(image_RBG_dir, exist_ok=True) 
                        os.makedirs(image_depth_dir, exist_ok=True) 
                        os.makedirs(depth_png_dir, exist_ok=True)

                        #### main loop #################                        
                        for idx, cam_pos in enumerate(random_camera_positions):

                            print("get image number:",idx)
                            ##########
                            ## get RGB image 
                            img_RGB, cam_pos, quat_xyzw = sim_node.get_camera_image_direct(
                                camera_name="testcamera", 
                                changed_xyz=cam_pos,
                                lookat_position=[-0.38, 0.90, 0.7845]
                            )

                            ## get depth image
                            img_depth, _ , _ = sim_node.get_camera_depth_direct(
                                camera_name="testcamera", 
                                changed_xyz=cam_pos,
                                lookat_position=[-0.38, 0.90, 0.7845]
                            )

                            ## save # format name : image{i}
                            img_RGB_filename = f"{output_dir}/input/image{idx}.png"  
                            img_depth_filename = f"{output_dir}/depth/image{idx}.npy" 
                            plt.imsave(img_RGB_filename, img_RGB)
                            np.save(img_depth_filename, img_depth)
                            
                            ## camera data collect 
                            camera_data = {
                                "cam_pos": cam_pos.tolist() if hasattr(cam_pos, 'tolist') else list(cam_pos),
                                "image_name": f"image{idx}.png",
                                "quat_xyzw": quat_xyzw.tolist() if hasattr(quat_xyzw, 'tolist') else list(quat_xyzw)
                            }
                            cameras_info.append(camera_data)
                        
                    ## save json of camera info
                    json_filename = os.path.join(output_dir, "mujoco_cam_infos.json")
                    with open(json_filename, 'w') as f:
                        json.dump({"cameras": cameras_info}, f, indent=4)
                    print(f"Saved camera info to {json_filename}")

                    ## get depth image
                    for fname in os.listdir(image_depth_dir):
                        if fname.endswith(".npy"):
                            npy_path = os.path.join(image_depth_dir, fname)
                            depth = np.load(npy_path)  # 原始 float32 深度

                            # 转换为 16-bit PNG 图像
                            depth_mm = (depth * 1000).astype(np.uint16)

                            # 保存为 PNG，改文件名为 image{i}.png（和原 RGB 对应）
                            idx = os.path.splitext(fname)[0].replace("image", "")
                            png_name = f"image{idx}.png"
                            png_path = os.path.join(depth_png_dir, png_name)
                            cv2.imwrite(png_path, depth_mm)

                    print(f"Converted npy -> png")                    
                        
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()

    for p in process_list:
        p.join()