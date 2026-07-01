"""
轨迹跟踪改进版 - 支持不同的抓取策略

根据分析发现：
1. 物体Z轴方向变化很大（从指向左侧到指向前上方）
2. 如果强制对齐物体Z轴，会导致机器人手腕大幅度旋转

改进策略：
- 策略1：固定接近方向 - 手腕方向保持相对稳定，只跟踪位置
- 策略2：完全跟随物体姿态 - 原方案，手腕会大幅度旋转
- 策略3：部分跟随 - 只跟随物体的平面旋转，忽略翻滚
- 策略4：固定相对位姿（推荐）- 首帧确定抓取姿态后，保持末端与物体相对关系不变
"""

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
project_root = os.path.dirname(parent_dir)
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

# ==================== 策略1：固定接近方向 ====================
def make_ee_pose_fixed_orientation(obj_position,
                                   ee_orientation=None,
                                   offset_base=np.zeros(3)):
    """
    策略1：固定末端执行器方向，只跟踪物体位置
    
    适用场景：物体姿态变化大，希望保持手腕稳定
    
    Args:
        obj_position: (3,) 物体位置
        ee_orientation: (3,3) 固定的末端执行器旋转矩阵，None则使用默认
        offset_base: (3,) 位置偏移
    
    Returns:
        pin.SE3
    """
    if ee_orientation is None:
        # 默认姿态：
        # X轴：向前（接近方向）
        # Y轴：向左（手掌开口方向）
        # Z轴：向上（手腕卷动轴）
        ee_orientation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    
    position = obj_position + offset_base
    return pin.SE3(ee_orientation, position)

# ==================== 策略2：完全跟随物体姿态 ====================
def make_ee_pose_follow_object(T_base_obj,
                               grasp_axis_obj=np.array([0.0, 0.0, 1.0]),
                               approach_base=np.array([0.0, 1.0, 0.0]),
                               offset_base=np.zeros(3)):
    """
    策略2：完全跟随物体姿态
    
    适用场景：物体姿态相对稳定，需要精确对齐
    
    Args:
        T_base_obj: (4,4) 物体位姿
        grasp_axis_obj: (3,) 物体坐标系中的抓取轴
        approach_base: (3,) 基座系中的接近方向
        offset_base: (3,) 位置偏移
    
    Returns:
        pin.SE3
    """
    R_bo = T_base_obj[:3, :3]
    t_bo = T_base_obj[:3, 3]

    # 末端 Z 轴：对齐物体抓取轴
    z_ee = _normalize(R_bo @ _normalize(grasp_axis_obj))
    
    # 末端 Y 轴：对齐接近方向
    y_ee = _normalize(approach_base)

    # 防止近似平行
    if abs(np.dot(y_ee, z_ee)) > 0.95:
        y_ee = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y_ee, z_ee)) > 0.95:
            y_ee = np.array([1.0, 0.0, 0.0])

    # 末端 X 轴
    x_ee = _normalize(np.cross(y_ee, z_ee))
    y_ee = np.cross(z_ee, x_ee)

    R_be = np.stack([x_ee, y_ee, z_ee], axis=1)
    t_be = t_bo + offset_base
    
    return pin.SE3(R_be, t_be)

# ==================== 策略3：部分跟随（推荐）====================
def make_ee_pose_partial_follow(T_base_obj,
                                fixed_approach=np.array([0.0, 1.0, 0.0]),
                                offset_base=np.zeros(3)):
    """
    策略3：部分跟随 - 跟踪位置，固定接近方向，根据物体在水平面的投影调整手腕
    
    适用场景：平衡稳定性和适应性（推荐）
    
    原理：
    - 位置跟随物体位置
    - 接近方向固定（如从侧面）
    - 手腕方向根据物体在水平面的投影微调
    
    Args:
        T_base_obj: (4,4) 物体位姿
        fixed_approach: (3,) 固定的接近方向
        offset_base: (3,) 位置偏移
    
    Returns:
        pin.SE3
    """
    R_bo = T_base_obj[:3, :3]
    t_bo = T_base_obj[:3, 3]
    
    # 获取物体Z轴（假设是长边）
    obj_z = R_bo[:, 2]
    
    # 投影到水平面（XY平面）
    obj_z_horizontal = np.array([obj_z[0], obj_z[1], 0.0])
    obj_z_horizontal_norm = np.linalg.norm(obj_z_horizontal)
    
    if obj_z_horizontal_norm > 0.1:
        # 如果水平分量足够大，使用投影方向
        z_ee = _normalize(obj_z_horizontal)
    else:
        # 如果物体几乎竖直，使用默认方向
        z_ee = np.array([1.0, 0.0, 0.0])
    
    # Y轴：固定接近方向
    y_ee = _normalize(fixed_approach)
    
    # X轴：叉积
    x_ee = _normalize(np.cross(y_ee, z_ee))
    y_ee = np.cross(z_ee, x_ee)
    
    R_be = np.stack([x_ee, y_ee, z_ee], axis=1)
    t_be = t_bo + offset_base
    
    return pin.SE3(R_be, t_be)

# ==================== 策略4：固定相对位姿（推荐）====================
def compute_initial_ee_pose_in_base(T_base_obj_first, grasp_style='side', offset_base=np.array([0.0, -0.05, 0.0])):
    """
    根据首帧物体位姿和抓取方式，计算末端执行器在基座坐标系下的初始位姿
    
    这一步定义：在基座坐标系下，末端执行器应该处于什么位姿来抓取物体
    
    Args:
        T_base_obj_first: (4,4) 首帧物体在基座系下的位姿
        grasp_style: 抓取方式
        offset_base: (3,) 在基座坐标系下的位置偏移
    
    Returns:
        pin.SE3 - T_base_ee_first (末端执行器在基座系下的初始位姿)
    """
    obj_position = T_base_obj_first[:3, 3]
    obj_rotation = T_base_obj_first[:3, :3]
    
    if grasp_style == 'side':
        # 从侧面抓取：末端执行器从Y方向接近物体
        # 末端执行器姿态：保持自然抓取姿态（手掌开口朝向物体）
        
        # 末端执行器的Y轴应指向物体中心
        ee_y = _normalize(-offset_base)
        if np.linalg.norm(ee_y) < 0.01:
            ee_y = np.array([0.0, 1.0, 0.0])
        
        # 末端执行器的Z轴尽量跟随物体Z轴（在水平面的投影）
        obj_z = obj_rotation[:, 2]
        obj_z_horizontal = np.array([obj_z[0], obj_z[1], 0.0])
        if np.linalg.norm(obj_z_horizontal) > 0.1:
            ee_z = _normalize(obj_z_horizontal)
        else:
            ee_z = np.array([0.0, 0.0, 1.0])
        
        # X轴通过叉积得到
        ee_x = _normalize(np.cross(ee_y, ee_z))
        ee_z = np.cross(ee_x, ee_y)
        
        R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        t_base_ee = obj_position + offset_base
        
    elif grasp_style == 'top':
        # 从上方抓取：末端执行器从上方接近
        # Z轴向下，X轴指向前方，Y轴根据物体方向调整
        obj_z = obj_rotation[:, 2]
        obj_z_horizontal = np.array([obj_z[0], obj_z[1], 0.0])
        
        if np.linalg.norm(obj_z_horizontal) > 0.1:
            ee_y = _normalize(obj_z_horizontal)
        else:
            ee_y = np.array([1.0, 0.0, 0.0])
        
        ee_z = np.array([0.0, 0.0, -1.0])  # Z轴向下
        ee_x = _normalize(np.cross(ee_y, ee_z))
        ee_y = np.cross(ee_z, ee_x)
        
        R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        t_base_ee = obj_position + np.array([0.0, 0.0, 0.15])  # 在物体上方
        
    elif grasp_style == 'front':
        # 从前方抓取：末端执行器从X方向接近
        ee_x = np.array([1.0, 0.0, 0.0])
        
        obj_z = obj_rotation[:, 2]
        obj_z_horizontal = np.array([obj_z[0], obj_z[1], 0.0])
        if np.linalg.norm(obj_z_horizontal) > 0.1:
            ee_z = _normalize(obj_z_horizontal)
        else:
            ee_z = np.array([0.0, 0.0, 1.0])
        
        ee_y = _normalize(np.cross(ee_z, ee_x))
        ee_x = np.cross(ee_y, ee_z)
        
        R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        t_base_ee = obj_position + np.array([-0.08, 0.0, 0.0])
        
    else:
        raise ValueError(f"Unknown grasp_style: {grasp_style}")
    
    return pin.SE3(R_base_ee, t_base_ee)

def compute_ee_to_obj_transform(T_base_obj_first, T_base_ee_first):
    """
    计算末端执行器相对于物体的变换
    
    T_obj_ee = inv(T_base_obj) * T_base_ee
    
    这个变换表示：末端执行器在物体坐标系中的位姿
    
    Args:
        T_base_obj_first: pin.SE3 - 首帧物体位姿
        T_base_ee_first: pin.SE3 - 首帧末端执行器位姿
    
    Returns:
        pin.SE3 - T_obj_ee
    """
    T_obj_ee = T_base_obj_first.inverse() * T_base_ee_first
    return T_obj_ee

def make_ee_pose_fixed_relative(T_base_obj, T_obj_ee):
    """
    策略4：固定相对位姿 - 保持末端执行器与物体的相对关系不变
    
    适用场景：最推荐的方案 - 自然跟随物体运动，同时保持稳定的抓取姿态
    
    原理：
    - 在首帧确定末端执行器相对于物体的位姿 T_obj_ee
    - 后续帧中，末端执行器位姿 T_base_ee = T_base_obj * T_obj_ee
    - 相当于将物体和末端执行器"绑定"在一起
    
    优点：
    - 完全跟随物体的平移和旋转
    - 末端执行器与物体保持固定相对位置和方向
    - 不会出现突变或大幅度手腕旋转
    - 轨迹平滑自然
    
    Args:
        T_base_obj: (4,4) numpy array - 当前帧物体位姿
        T_obj_ee: pin.SE3 - 物体到末端执行器的固定变换
    
    Returns:
        pin.SE3 - T_base_ee
    """
    T_base_obj_se3 = mat_to_se3(T_base_obj)
    T_base_ee = T_base_obj_se3 * T_obj_ee
    return T_base_ee

def FK(ik, q):
    """Forward kinematics"""
    pin.forwardKinematics(ik.reduced_robot.model, ik.reduced_robot.data, q)
    pin.updateFramePlacements(ik.reduced_robot.model, ik.reduced_robot.data)
    d = ik.reduced_robot.data
    return d.oMf[ik.L_hand_id], d.oMf[ik.R_hand_id]

def pack_trajectory_frame(ik, q):
    """
    将关节角度打包成轨迹帧格式，与g1_ik_control.py的_pack_frame()一致
    
    Args:
        ik: IK求解器
        q: (12,) 关节角度
    
    Returns:
        (26,) numpy array: [q(12) | pL(3) | qL(4) | pR(3) | qR(4)]
    """
    TL, TR = FK(ik, q)
    
    # 提取位置
    pL = TL.translation
    pR = TR.translation
    
    # 提取四元数 (scipy格式: x, y, z, w)
    quatL = R.from_matrix(TL.rotation).as_quat()
    quatR = R.from_matrix(TR.rotation).as_quat()
    
    # 打包: q(12) + pL(3) + qL(4) + pR(3) + qR(4) = 26
    return np.hstack([q, pL, quatL, pR, quatR])

def save_trajectory_to_records(traj_frames, dt=0.02, output_dir=None):
    """
    保存轨迹到records文件夹
    
    Args:
        traj_frames: list of (26,) arrays 或 (n, 26) numpy array
        dt: 控制周期
        output_dir: 输出目录，默认为项目根目录的records/
    
    Returns:
        保存的文件路径
    """
    if output_dir is None:
        # 默认保存到项目根目录的records文件夹
        output_dir = os.path.join(project_root, 'records')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为numpy数组
    traj_array = np.array(traj_frames)
    
    # 生成文件名
    timestamp = int(time.time())
    filename = os.path.join(output_dir, f'traj_{timestamp}.npz')
    
    # 保存，格式与g1_ik_control.py一致
    np.savez_compressed(
        filename,
        traj=traj_array,
        dt=dt,
        note="cols=[q(12)|pL(3)|qL(4)|pR(3)|qR(4)]"
    )
    
    print(f"[SAVE] 轨迹已保存: {filename}")
    print(f"       形状: {traj_array.shape}")
    print(f"       帧数: {traj_array.shape[0]}, 维度: {traj_array.shape[1]}")
    
    return filename

def IK(ik, q, poseL: pin.SE3, poseR: pin.SE3):
    """Inverse kinematics"""
    q_cmd, _ = ik.solve_ik(poseL.homogeneous, poseR.homogeneous, current_lr_arm_motor_q=q)
    return q_cmd

def transform_poses_to_base(poses_camera, waist_yaw=0, waist_pitch=0, waist_roll=0):
    """Transform pose sequence from camera frame to base frame"""
    T_base_camera = get_camera_pose_in_base_frame(waist_yaw, waist_pitch, waist_roll)
    poses_base = []
    for T_cam_obj in poses_camera:
        T_base_obj = get_target_pose_in_base_frame(T_base_camera, T_cam_obj)
        poses_base.append(T_base_obj)
    return poses_base

def generate_ee_trajectory(poses_base, strategy='fixed_relative', **kwargs):
    """
    生成末端执行器轨迹
    
    Args:
        poses_base: list of (4,4) numpy arrays
        strategy: 'fixed', 'follow', 'partial', or 'fixed_relative'
        **kwargs: 策略相关参数
    
    Returns:
        list of pin.SE3
    """
    ee_trajectory = []
    
    if strategy == 'fixed':
        # 策略1：固定方向
        ee_orientation = kwargs.get('ee_orientation', None)
        offset = kwargs.get('offset_base', np.zeros(3))
        for T in poses_base:
            pose = make_ee_pose_fixed_orientation(T[:3, 3], ee_orientation, offset)
            ee_trajectory.append(pose)
            
    elif strategy == 'follow':
        # 策略2：完全跟随
        grasp_axis = kwargs.get('grasp_axis_obj', np.array([0, 0, 1]))
        approach = kwargs.get('approach_base', np.array([0, 1, 0]))
        offset = kwargs.get('offset_base', np.zeros(3))
        for T in poses_base:
            pose = make_ee_pose_follow_object(T, grasp_axis, approach, offset)
            ee_trajectory.append(pose)
            
    elif strategy == 'partial':
        # 策略3：部分跟随
        approach = kwargs.get('fixed_approach', np.array([0, 1, 0]))
        offset = kwargs.get('offset_base', np.zeros(3))
        for T in poses_base:
            pose = make_ee_pose_partial_follow(T, approach, offset)
            ee_trajectory.append(pose)
    
    elif strategy == 'fixed_relative':
        # 策略4：固定相对位姿（推荐）
        # 步骤1：根据首帧物体位姿和抓取方式，计算末端执行器的初始位姿
        T_base_obj_first = poses_base[0]
        grasp_style = kwargs.get('grasp_style', 'side')
        offset_base = kwargs.get('offset_base', np.array([0.0, -0.05, 0.0]))
        
        # 在基座系下计算末端执行器的初始位姿
        T_base_ee_first = compute_initial_ee_pose_in_base(T_base_obj_first, grasp_style, offset_base)
        
        print(f"  首帧末端执行器位姿 T_base_ee_first:")
        print(f"    位置: {T_base_ee_first.translation}")
        print(f"    旋转矩阵:\n{T_base_ee_first.rotation}")
        
        # 步骤2：计算末端执行器相对于物体的固定变换
        T_base_obj_first_se3 = mat_to_se3(T_base_obj_first)
        T_obj_ee = compute_ee_to_obj_transform(T_base_obj_first_se3, T_base_ee_first)
        
        print(f"\n  相对变换 T_obj_ee (末端执行器在物体坐标系中):")
        print(f"    位置: {T_obj_ee.translation}")
        print(f"    旋转矩阵:\n{T_obj_ee.rotation}")
        
        # 步骤3：对每一帧，应用固定的相对变换
        for T in poses_base:
            pose = make_ee_pose_fixed_relative(T, T_obj_ee)
            ee_trajectory.append(pose)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return ee_trajectory

def main():
    # ========== 选择策略 ==========
    print("=" * 60)
    print("选择抓取策略:")
    print("  1 - 固定方向（手腕不旋转，仅跟踪位置）")
    print("  2 - 完全跟随（原方案，手腕大幅度旋转）")
    print("  3 - 部分跟随（跟随水平方向，忽略俯仰）")
    print("  4 - 固定相对位姿（推荐！首帧绑定后保持相对关系）")
    print("=" * 60)
    
    strategy_choice = input("请输入选择 (1/2/3/4，默认4): ").strip() or '4'
    strategy_map = {'1': 'fixed', '2': 'follow', '3': 'partial', '4': 'fixed_relative'}
    strategy = strategy_map.get(strategy_choice, 'fixed_relative')
    
    print(f"\n选择了策略: {strategy}")
    
    # 如果选择策略4，询问抓取方式
    grasp_style = 'side'
    if strategy == 'fixed_relative':
        print("\n" + "=" * 60)
        print("选择抓取方式:")
        print("  1 - 从侧面抓取 (默认)")
        print("  2 - 从上方抓取")
        print("  3 - 从前方抓取")
        print("=" * 60)
        grasp_choice = input("请输入选择 (1/2/3，默认1): ").strip() or '1'
        grasp_map = {'1': 'side', '2': 'top', '3': 'front'}
        grasp_style = grasp_map.get(grasp_choice, 'side')
        print(f"选择了抓取方式: {grasp_style}")
    
    # ========== 初始化 ==========
    print("\n" + "=" * 60)
    print("初始化 IK 求解器...")
    print("=" * 60)
    ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    
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
    poses_camera = poses_camera[15:45]
    
    # ========== 转换到基座坐标系 ==========
    print("\n" + "=" * 60)
    print("转换位姿到基座坐标系...")
    print("=" * 60)
    waist_yaw, waist_pitch, waist_roll = 0, 0, 0
    poses_base = transform_poses_to_base(poses_camera, waist_yaw, waist_pitch, waist_roll)
    
    # ========== 生成末端执行器轨迹 ==========
    print("\n" + "=" * 60)
    print("生成末端执行器轨迹...")
    print("=" * 60)
    
    # 根据策略生成轨迹
    if strategy == 'fixed':
        ee_trajectory_R = generate_ee_trajectory(
            poses_base,
            strategy='fixed',
            offset_base=np.array([0.0, -0.05, 0.0])
        )
    elif strategy == 'follow':
        ee_trajectory_R = generate_ee_trajectory(
            poses_base,
            strategy='follow',
            grasp_axis_obj=np.array([0.0, 0.0, 1.0]),
            approach_base=np.array([0.0, 1.0, 0.0]),
            offset_base=np.array([0.0, -0.05, 0.0])
        )
    elif strategy == 'partial':
        ee_trajectory_R = generate_ee_trajectory(
            poses_base,
            strategy='partial',
            fixed_approach=np.array([0.0, 1.0, 0.0]),
            offset_base=np.array([0.0, -0.05, 0.0])
        )
    else:  # fixed_relative
        ee_trajectory_R = generate_ee_trajectory(
            poses_base,
            strategy='fixed_relative',
            grasp_style=grasp_style,
            offset_base=np.array([0.0, -0.05, 0.0])  # 在基座系中的偏移
        )
    
    print(f"生成了 {len(ee_trajectory_R)} 个目标位姿")
    
    # ========== 定义Home位姿 ==========
    print("\n" + "=" * 60)
    print("定义Home位姿...")
    print("=" * 60)
    q_home = np.array([
        -0.063, 1.636, 0.063, 1.434,
        -0.069, -0.016, -0.024, -1.577,
        -0.064, 1.411, 0.049, 0.044
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
    
    # 打开文件保存q值
    q_debug_file = open(os.path.join(current_dir, "q_for_debug.txt"), 'w')
    
    # 用于保存轨迹数据
    trajectory_frames = []
    
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
                T_curr_L = T_home_L
                
                q_current = IK(ik, q_current, T_curr_L, T_curr_R)
                
                # 保存轨迹数据（只在第一次循环时保存）
                if loop_count == 1:
                    frame = pack_trajectory_frame(ik, q_current)
                    trajectory_frames.append(frame)
                
                time.sleep(0.02)
            
            time.sleep(0.5)
            
            # 调试输出（可选）
            if loop_count == 1:
                q_current_str = ' '.join([f'{q:.6f}' for q in q_current])
                print(f"    当前关节状态 q: {q_current_str}")
                q_debug_file.write(q_current_str + '\n')
            
            # Phase 2: 沿轨迹移动
            print("  阶段2: 沿轨迹移动")
            steps_per_segment = 20
            for idx in range(len(ee_trajectory_R) - 1):
                T_start = ee_trajectory_R[idx]
                T_end = ee_trajectory_R[idx + 1]
                
                for i in range(steps_per_segment):
                    alpha = (i + 1) / steps_per_segment
                    T_curr_R = interpolate_pose(T_start, T_end, alpha)
                    T_curr_L = T_home_L
                    
                    q_current = IK(ik, q_current, T_curr_L, T_curr_R)
                    
                    # 保存q值到文件（只在第一次循环时保存）
                    if loop_count == 1:
                        q_str = ' '.join([f'{q:.6f}' for q in q_current])
                        q_debug_file.write(q_str + '\n')
                        
                        # 打包轨迹帧并保存
                        frame = pack_trajectory_frame(ik, q_current)
                        trajectory_frames.append(frame)
                    
                    time.sleep(0.02)
                
                if (idx + 1) % 10 == 0:
                    print(f"    已完成 {idx + 1}/{len(ee_trajectory_R) - 1} 个轨迹点")
            
            time.sleep(0.5)
            
            # Phase 3: 最后一个目标 -> Home
            print("  阶段3: 轨迹终点 -> Home")
            steps = 100
            for i in range(steps):
                alpha = (i + 1) / steps
                T_curr_R = interpolate_pose(ee_trajectory_R[-1], T_home_R, alpha)
                T_curr_L = T_home_L
                
                q_current = IK(ik, q_current, T_curr_L, T_curr_R)
                time.sleep(0.02)
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("用户终止程序")
        print("=" * 60)
    finally:
        # 关闭文件
        q_debug_file.close()
        print(f"\nq值已保存到: {os.path.join(current_dir, 'q_for_debug.txt')}")
        
        # 保存轨迹数据到records文件夹
        if len(trajectory_frames) > 0:
            print(f"\n正在保存轨迹数据...")
            save_trajectory_to_records(trajectory_frames, dt=0.02)
        else:
            print(f"\n警告：没有轨迹数据可保存")

if __name__ == "__main__":
    main()
