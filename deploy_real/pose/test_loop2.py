import sys
import os
import time
import glob
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from g1_arm_IK import G1_29_ArmIK
from utils.coordinate_trans import get_camera_pose_in_base_frame, get_target_pose_in_base_frame

def load_pose_from_txt(path):
    """Load single 4x4 pose matrix from txt file"""
    arr = np.loadtxt(path).reshape(4, 4)
    return arr

def load_pose_sequence(folder_path):
    """Load all pose files from folder in sorted order"""
    pose_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    poses = []
    for file in pose_files:
        pose = load_pose_from_txt(file)
        poses.append(pose)
    print(f"Loaded {len(poses)} poses from {folder_path}")
    return poses

def mat_to_se3(mat):
    """Convert 4x4 numpy matrix to pin.SE3"""
    return pin.SE3(mat[:3, :3], mat[:3, 3])

def interpolate_pose(T_start, T_end, alpha):
    """
    Interpolate between two SE3 poses
    Args:
        T_start: pin.SE3, starting pose
        T_end: pin.SE3, ending pose
        alpha: float in [0,1], interpolation parameter
    Returns:
        pin.SE3: interpolated pose
    """
    # Linear interpolation of translation
    p_start = T_start.translation
    p_end = T_end.translation
    p_interp = (1 - alpha) * p_start + alpha * p_end
    
    # Slerp for rotation
    R_start = T_start.rotation
    R_end = T_end.rotation
    q_start = pin.Quaternion(R_start)
    q_end = pin.Quaternion(R_end)
    q_interp = q_start.slerp(alpha, q_end)
    
    return pin.SE3(q_interp, p_interp)

def _normalize(v, eps=1e-9):
    """Normalize vector with safety check"""
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize zero vector")
    return v / n

def make_grasp_pose_from_object(T_base_obj,
                                grasp_axis_obj=np.array([0.0, 0.0, 1.0]),
                                approach_base=np.array([0.0, 1.0, 0.0]),
                                offset_base=np.zeros(3)):
    """
    从物体位姿构造末端执行器的抓取位姿
    
    设计原则：
    - 末端执行器的 Z 轴（手腕卷动轴）对齐物体的抓取轴
    - 末端执行器的 Y 轴（手掌法向）对齐接近方向
    - 末端执行器的 X 轴通过右手定则确定
    
    Args:
        T_base_obj: (4,4) numpy array, 物体在基座系下的位姿
        grasp_axis_obj: (3,) 物体坐标系中的抓取轴方向（例如长边方向）
        approach_base: (3,) 基座系中的接近方向（例如从侧面接近）
        offset_base: (3,) 基座系中的额外位置偏移
    
    Returns:
        pin.SE3: 末端执行器的目标位姿
    """
    R_bo = T_base_obj[:3, :3]
    t_bo = T_base_obj[:3, 3]

    # 末端 Z 轴：对齐物体抓取轴（在基座系中）
    z_ee = _normalize(R_bo @ _normalize(grasp_axis_obj))
    
    # 末端 Y 轴：对齐接近方向
    y_ee = _normalize(approach_base)

    # 防止近似平行导致叉积退化
    if abs(np.dot(y_ee, z_ee)) > 0.95:
        # 如果接近方向和抓取轴平行，选择一个默认接近方向
        y_ee = np.array([0.0, 0.0, 1.0])  # fallback到向上
        if abs(np.dot(y_ee, z_ee)) > 0.95:
            y_ee = np.array([1.0, 0.0, 0.0])  # 再fallback

    # 末端 X 轴：通过叉积确定（右手定则）
    x_ee = _normalize(np.cross(y_ee, z_ee))
    
    # 重新正交化 Y 轴
    y_ee = np.cross(z_ee, x_ee)

    R_be = np.stack([x_ee, y_ee, z_ee], axis=1)
    t_be = t_bo + offset_base
    
    return pin.SE3(R_be, t_be)

def FK(ik, q):
    """
    Forward kinematics
    Args:
        ik: G1_29_ArmIK instance
        q: arm motor joint state (n-joints)
    Returns:
        tuple: (left_hand_pose, right_hand_pose) as pin.SE3
    """
    pin.forwardKinematics(ik.reduced_robot.model, ik.reduced_robot.data, q)
    pin.updateFramePlacements(ik.reduced_robot.model, ik.reduced_robot.data)
    d = ik.reduced_robot.data
    return d.oMf[ik.L_hand_id], d.oMf[ik.R_hand_id]

def IK(ik, q, poseL: pin.SE3, poseR: pin.SE3):
    """
    Inverse kinematics
    Args:
        ik: G1_29_ArmIK instance
        q: current arm motor joint state (n-joints)
        poseL: left hand end-effector pose
        poseR: right hand end-effector pose
    Returns:
        q_cmd: commanded arm motor joint state (n-joints)
    """
    q_cmd, _ = ik.solve_ik(poseL.homogeneous, poseR.homogeneous, current_lr_arm_motor_q=q)
    return q_cmd

def transform_poses_to_base(poses_camera, waist_yaw=0, waist_pitch=0, waist_roll=0):
    """
    Transform pose sequence from camera frame to base frame
    Args:
        poses_camera: list of (4,4) numpy arrays in camera frame
        waist_yaw, waist_pitch, waist_roll: waist joint angles
    Returns:
        list of (4,4) numpy arrays in base frame
    """
    T_base_camera = get_camera_pose_in_base_frame(waist_yaw, waist_pitch, waist_roll)
    poses_base = []
    for T_cam_obj in poses_camera:
        T_base_obj = get_target_pose_in_base_frame(T_base_camera, T_cam_obj)
        poses_base.append(T_base_obj)
    return poses_base

def generate_ee_trajectory(poses_base, 
                          grasp_axis_obj=np.array([0.0, 0.0, 1.0]),
                          approach_base=np.array([0.0, 1.0, 0.0]),
                          offset_base=np.zeros(3)):
    """
    Generate end-effector trajectory from object poses
    Args:
        poses_base: list of (4,4) numpy arrays, object poses in base frame
        grasp_axis_obj: grasp axis in object frame
        approach_base: approach direction in base frame
        offset_base: position offset in base frame
    Returns:
        list of pin.SE3, end-effector target poses
    """
    ee_trajectory = []
    for T_base_obj in poses_base:
        T_ee = make_grasp_pose_from_object(T_base_obj, grasp_axis_obj, approach_base, offset_base)
        ee_trajectory.append(T_ee)
    return ee_trajectory

def main():
    # ========== 初始化 ==========
    print("=" * 60)
    print("初始化 IK 求解器...")
    print("=" * 60)
    ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)
    
    # ========== 加载位姿序列 ==========
    print("\n" + "=" * 60)
    print("加载物体位姿序列...")
    print("=" * 60)
    pose_folder = os.path.join(current_dir, "ob_in_cam")
    if not os.path.exists(pose_folder):
        print(f"错误：文件夹不存在 {pose_folder}")
        return
    
    poses_camera = load_pose_sequence(pose_folder)
    if len(poses_camera) == 0:
        print("错误：未找到位姿文件")
        return
    
    # ========== 转换到基座坐标系 ==========
    print("\n" + "=" * 60)
    print("转换位姿到基座坐标系...")
    print("=" * 60)
    waist_yaw, waist_pitch, waist_roll = 0, 0, 0  # 假设腰部关节为0
    poses_base = transform_poses_to_base(poses_camera, waist_yaw, waist_pitch, waist_roll)
    
    # ========== 生成末端执行器轨迹 ==========
    print("\n" + "=" * 60)
    print("生成末端执行器轨迹...")
    print("=" * 60)
    # 参数说明：
    # - grasp_axis_obj: 物体坐标系中的抓取轴（例如 [0,0,1] 表示Z轴，物体的长边）
    # - approach_base: 基座系中的接近方向（例如 [0,1,0] 表示从左侧接近）
    # - offset_base: 额外的位置偏移（例如 [0, -0.05, 0] 表示向右偏移5cm）
    
    ee_trajectory_R = generate_ee_trajectory(
        poses_base,
        grasp_axis_obj=np.array([0.0, 0.0, 1.0]),
        approach_base=np.array([0.0, -1.0, 0.0]),
        offset_base=np.array([0.0, -0.05, 0.0]) 
    )
    
    print(f"生成了 {len(ee_trajectory_R)} 个目标位姿")
    
    # ========== 定义Home位姿 ==========
    print("\n" + "=" * 60)
    print("定义Home位姿...")
    print("=" * 60)
    # 使用一个合理的初始关节配置
    q_home = np.array([
        -0.063, 1.636, 0.063, 1.434,  # 左臂
        -0.069, -0.016, -0.024, -1.577,  # 右臂
        -0.064, 1.411, 0.049, 0.044   # 剩余关节
    ])
    
    T_home_L, T_home_R = FK(ik, q_home)
    print(f"左手Home位置: {T_home_L.translation}")
    print(f"右手Home位置: {T_home_R.translation}")
    
    # ========== 初始化关节状态 ==========
    q_current = q_home.copy()
    
    # ========== 移动到Home位姿 ==========
    print("\n" + "=" * 60)
    print("移动到Home位姿...")
    print("=" * 60)
    steps_init = 50
    L_start, R_start = FK(ik, q_current)
    
    for i in range(steps_init):
        alpha = (i + 1) / steps_init
        T_curr_L = interpolate_pose(L_start, T_home_L, alpha)
        T_curr_R = interpolate_pose(R_start, T_home_R, alpha)
        
        q_current = IK(ik, q_current, T_curr_L, T_curr_R)
        time.sleep(0.02)
    
    print("已到达Home位姿")
    
    # ========== 轨迹跟踪循环 ==========
    print("\n" + "=" * 60)
    print("开始轨迹跟踪，按 Ctrl+C 停止")
    print("=" * 60)
    
    try:
        loop_count = 0
        while True:
            loop_count += 1
            print(f"\n>>> 第 {loop_count} 次循环")
            
            # Phase 1: Home -> 第一个目标
            print("  阶段1: Home -> 轨迹起点")
            steps = 100
            for i in range(steps):
                alpha = (i + 1) / steps
                T_curr_R = interpolate_pose(T_home_R, ee_trajectory_R[0], alpha)
                T_curr_L = T_home_L  # 左手保持不动
                
                q_current = IK(ik, q_current, T_curr_L, T_curr_R)
                time.sleep(0.02)
            
            time.sleep(0.5)  # 暂停观察
            
            # Phase 2: 沿轨迹移动
            print("  阶段2: 沿轨迹移动")
            steps_per_segment = 20  # 每两个位姿之间的插值步数
            for idx in range(len(ee_trajectory_R) - 1):
                T_start = ee_trajectory_R[idx]
                T_end = ee_trajectory_R[idx + 1]
                
                for i in range(steps_per_segment):
                    alpha = (i + 1) / steps_per_segment
                    T_curr_R = interpolate_pose(T_start, T_end, alpha)
                    T_curr_L = T_home_L
                    
                    q_current = IK(ik, q_current, T_curr_L, T_curr_R)
                    time.sleep(0.02)
                
                if (idx + 1) % 10 == 0:
                    print(f"    已完成 {idx + 1}/{len(ee_trajectory_R) - 1} 个轨迹点")
            
            time.sleep(0.5)  # 暂停观察
            
            # Phase 3: 最后一个目标 -> Home
            print("  阶段3: 轨迹终点 -> Home")
            steps = 100
            for i in range(steps):
                alpha = (i + 1) / steps
                T_curr_R = interpolate_pose(ee_trajectory_R[-1], T_home_R, alpha)
                T_curr_L = T_home_L
                
                q_current = IK(ik, q_current, T_curr_L, T_curr_R)
                time.sleep(0.02)
            
            time.sleep(1.0)  # 循环间隔
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("用户终止程序")
        print("=" * 60)

if __name__ == "__main__":
    main()
