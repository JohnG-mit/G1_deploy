import sys
import os
import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# 添加路径以导入 deploy_real 下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from g1_ik_control import G1HighlevelArmController
from utils.coordinate_trans import get_camera_pose_in_base_frame, get_target_pose_in_base_frame
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

def load_pose_from_txt(path):
    """从txt文件加载4x4位姿矩阵"""
    return np.loadtxt(path).reshape(4, 4)

def main():
    # 0. 初始化 Unitree SDK (如果尚未初始化)
    # 注意：如果 g1_ik_control 内部没有全局初始化，这里需要初始化
    ChannelFactoryInitialize(0, "lo") # 使用 lo 或实际网卡

    # 1. 加载物体在相机坐标系下的位姿 (T_camera_object)
    txt_path = os.path.join(current_dir, "pose/test.txt")
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
        return
    T_camera_object = load_pose_from_txt(txt_path)
    print("Loaded Object Pose in Camera Frame:\n", T_camera_object)

    # 2. 获取相机在基坐标系下的位姿 (T_base_camera)
    # 注意：这里假设腰部关节角度为0。如果在运动中腰部有转动，需要读取实际关节角度传入。
    # 例如：waist_yaw, waist_roll, waist_pitch = ctrl.get_waist_angles()
    T_base_camera = get_camera_pose_in_base_frame(0, 0, 0)
    print("Camera Pose in Base Frame:\n", T_base_camera)

    # 3. 定义抓取偏移 (T_grasp_offset: T_object_grasp)
    # 根据 stool-leg-scaled.obj 分析：
    # 物体中心在 [0, 0, 0]，尺寸约为 4.3cm x 26cm x 3.4cm (Y轴为长轴)
    # 假设我们想要抓取物体中心，并且让夹爪的接近方向 (Z轴) 垂直于物体长轴 (Y轴)
    # 设定：
    #   Grasp Z (接近方向) -> Object -Z (从物体上方接近)
    #   Grasp Y (手指开合) -> Object X (沿短轴抓取)
    #   Grasp X           -> Object Y (沿长轴)
    
    R_grasp = np.array([
        [1, 0, 0],  # Grasp X = Object X
        [0, 1, 0],  # Grasp Y = Object Y
        [0, 0, 1]  # Grasp Z = Object Z
    ])
    
    T_grasp_offset = np.eye(4)
    T_grasp_offset[:3, :3] = R_grasp
    T_grasp_offset[:3, 3] = [0, 0, 0] # 抓取中心
    print("Grasp Offset (Object -> Grasp):\n", T_grasp_offset)

    # 4. 计算基坐标系下的抓取位姿 (T_base_grasp)
    T_base_grasp = get_target_pose_in_base_frame(T_base_camera, T_camera_object, T_grasp_offset)
    print("Grasp Pose in Base Frame:\n", T_base_grasp)

    # 5. 计算预抓取位姿 (Pre-Grasp)
    # 在抓取位姿的基础上，沿夹爪 Z 轴 (接近方向) 后退 10cm
    T_pre_offset = np.eye(4)
    T_pre_offset[2, 3] = -0.10  # 后退 10cm
    T_base_pre_grasp = T_base_grasp @ T_pre_offset
    print("Pre-Grasp Pose in Base Frame:\n", T_base_pre_grasp)

    # 转换为 Pinocchio SE3 对象
    pose_pre_grasp = pin.SE3(T_base_pre_grasp[:3, :3], T_base_pre_grasp[:3, 3])
    pose_grasp = pin.SE3(T_base_grasp[:3, :3], T_base_grasp[:3, 3])

    # 6. 初始化控制器并执行运动
    print("Initializing Controller...")
    ctrl = G1HighlevelArmController()
    ctrl.start()
    
    # 定义左手保持的姿态 (Home Pose)
    # 假设左手保持在胸前
    poseL_home = pin.SE3(np.eye(3), np.array([0.3, 0.25, 0.1])) 

    try:
        print("Moving Right Hand to Pre-Grasp Pose...")
        # 使用 move_to 函数
        # 注意：move_to 需要左右手的位姿。这里右手去预抓取点，左手保持 Home。
        ctrl.move_to(poseL_home, pose_pre_grasp, ws_steps=100)
        time.sleep(1.0)

        print("Moving Right Hand to Grasp Pose...")
        ctrl.move_to(poseL_home, pose_grasp, ws_steps=50)
        time.sleep(1.0)
        
        # 这里可以添加闭合夹爪的指令
        # ctrl.close_gripper('r') 
        
        print("Done.")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        ctrl.stop()

if __name__ == "__main__":
    main()
