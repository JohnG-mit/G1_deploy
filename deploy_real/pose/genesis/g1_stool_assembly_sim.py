"""
G1机器人 IKEA凳子装配仿真
使用Genesis进行场景构建，OMPL进行运动规划
"""

import os
import sys
import time
import torch
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
project_root = parent_dir.parent.parent  # G1_deploy目录
sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Tuple, Union
from deploy_real.common.utils import *

import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat
from genesis.utils.misc import tensor_to_array

# ========================== 配置参数 ==========================
# 桌子参数
TABLE_SIZE = (0.77, 1.16, 0.70)  # 宽77cm, 长116cm, 高70cm
TABLE_POS = (0.485, 0.0, 0.35)   # 机器人身前5cm处（桌子中心距离机器人约48.5cm）

# 凳面参数（放在桌子上）
STOOL_TOP_POS = (0.285, 0.0, 0.72)  # 凳面放在桌子上方

# 凳腿初始位置（放在桌子上，待抓取）
STOOL_LEG_POSITIONS = [
    (0.15, 0.15, 0.83),   # 左前
    (0.15, -0.15, 0.83),  # 右前
    (0.35, 0.15, 0.83),   # 左后
    (0.35, -0.15, 0.83),  # 右后
]

# 凳面螺孔位置（相对于凳面中心）- 需要根据实际凳子模型调整
STOOL_HOLE_OFFSETS = [
    (0.15, 0.12, -0.02),   # 左前孔
    (0.15, -0.12, -0.02),  # 右前孔
    (-0.15, 0.12, -0.02),  # 左后孔
    (-0.15, -0.12, -0.02), # 右后孔
]

# ========================== 初始化 ==========================
gs.init(
    backend=gs.gpu,
    logging_level='warning',
)
from genesis.engine.entities import RigidEntity

# ========================== 创建场景 ==========================
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(2.0, -2.0, 1.5),
        camera_lookat=(0.5, 0.0, 0.7),
        camera_fov=45,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=0.5,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.3, 0.3, 0.3),
    ),
    sim_options=gs.options.SimOptions(
        dt=0.005,  # 减小时间步长以提高稳定性
        gravity=(0, 0, -9.81),
    ),
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_joint_limit=True,
    ),
    renderer=gs.renderers.Rasterizer(),
)

# ========================== 添加实体 ==========================

# 1. 地面
plane = scene.add_entity(gs.morphs.Plane())

table: RigidEntity = scene.add_entity(
    gs.morphs.Box(
        size=TABLE_SIZE,
        pos=TABLE_POS,
        fixed=True,
    ),
    surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2, 1.0)),  # 木色
)

stool_top: RigidEntity = scene.add_entity(
    gs.morphs.Mesh(
        file='deploy_real/assets_obj/stool/stool-scale.obj',
        pos=STOOL_TOP_POS,
        euler=(90, 0, -10),
        scale=1.0,
        fixed=True,  # 凳面固定在桌子上
    ),
)

stool_legs: List[RigidEntity] = []
for i, leg_pos in enumerate(STOOL_LEG_POSITIONS):
    leg = scene.add_entity(
        gs.morphs.Mesh(
            file='deploy_real/assets_obj/stool-leg/stool-leg-scaled.obj',
            pos=leg_pos,
            euler=(90, 0, 0),
            scale=1.0,
            fixed=False,  # 凳腿可以被抓取移动
        ),
    )
    stool_legs.append(leg)

g1: RigidEntity = scene.add_entity(
    gs.morphs.URDF(
        file='deploy_real/assets/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf',
        pos=(0, 0, 0.75),
        euler=(0, 0, 0),
        scale=1.0,
        fixed=True,  # 固定基座，防止机器人倒下
        prioritize_urdf_material=True,
    ),
)

# ========================== 构建场景 ==========================
scene.build()

# ========================== 获取机器人关节信息 ==========================
print("\n========== G1 Robot Joint Info ==========")
print(f"Number of joints: {g1.n_joints}")
print(f"Number of dofs: {g1.n_dofs}")
print(f"Number of links: {g1.n_links}")

# 打印所有关节名称
print("\nJoint names:")
for i, joint in enumerate(g1.joints):
    print(f"  [{i}] {joint.name} - dof_idx: {joint.dofs_idx_local}")

# 打印所有link名称
print("\nLink names:")
for i, link in enumerate(g1.links):
    print(f"  [{i}] {link.name}")

# ========================== 定义控制参数 ==========================

LEFT_ARM_JOINTS = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
]

RIGHT_ARM_JOINTS = [
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

LEFT_HAND_JOINTS = [
    'left_index_1_joint',
    'left_index_2_joint',
    'left_little_1_joint',
    'left_little_2_joint',
    'left_middle_1_joint',
    'left_middle_2_joint',
    'left_ring_1_joint',
    'left_ring_2_joint',
    'left_thumb_1_joint',
    'left_thumb_2_joint',
    'left_thumb_3_joint',
    'left_thumb_4_joint',
]

RIGHT_HAND_JOINTS = [
    'right_index_1_joint',
    'right_index_2_joint',
    'right_little_1_joint',
    'right_little_2_joint',
    'right_middle_1_joint',
    'right_middle_2_joint',
    'right_ring_1_joint',
    'right_ring_2_joint',
    'right_thumb_1_joint',
    'right_thumb_2_joint',
    'right_thumb_3_joint',
    'right_thumb_4_joint',
]

# ========================== Inspire Hand FTP 控制映射 ==========================
# Inspire Hand FTP 控制接口:
#   ANGLE_SET: [小拇指弯曲, 无名指弯曲, 中指弯曲, 食指弯曲, 大拇指弯曲, 大拇指旋转]
#   范围: 0~1000, 0=闭合(弯曲), 1000=完全张开
#   FORCE_SET: 同样6个值，力控阈值，范围0~1000

# 手指名称到 URDF 关节的映射
# 每个手指在 URDF 中有 2 个关节 (1: 近端, 2: 远端)，大拇指有 4 个关节
INSPIRE_FINGER_MAPPING = {
    'little': ['little_1_joint', 'little_2_joint'],      # 小拇指
    'ring':   ['ring_1_joint', 'ring_2_joint'],          # 无名指
    'middle': ['middle_1_joint', 'middle_2_joint'],      # 中指
    'index':  ['index_1_joint', 'index_2_joint'],        # 食指
    'thumb_bend': ['thumb_2_joint', 'thumb_3_joint', 'thumb_4_joint'],  # 大拇指弯曲
    'thumb_rot':  ['thumb_1_joint'],                     # 大拇指旋转
}

# Inspire Hand 控制顺序 (与 ANGLE_SET/FORCE_SET 数组索引对应)
INSPIRE_CONTROL_ORDER = ['little', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rot']

# 关节角度范围 (弧度) - 根据 URDF 中的 limit 定义
# Inspire 控制值: 0 = 闭合(max_angle), 1000 = 张开(min_angle=0)
# 注意: _2 关节是 mimic 关节，Genesis 可能自动处理联动
FINGER_JOINT_RANGES = {
    # 小拇指: lower=0, upper=1.4381 (mimic: _1 * 1.0843)
    'little_1': (0.0, 1.4381),
    'little_2': (0.0, 1.5595),  # 1.4381 * 1.0843 ≈ 1.5595 (实际 upper=3.14，但联动限制)
    # 无名指: lower=0, upper=1.4381 (mimic: _1 * 1.0843)
    'ring_1':   (0.0, 1.4381),
    'ring_2':   (0.0, 1.5595),
    # 中指: lower=0, upper=1.4381 (mimic: _1 * 1.0843)
    'middle_1': (0.0, 1.4381),
    'middle_2': (0.0, 1.5595),
    # 食指: lower=0, upper=1.4381 (mimic: _1 * 1.0843)
    'index_1':  (0.0, 1.4381),
    'index_2':  (0.0, 1.5595),
    # 大拇指旋转: lower=0, upper=1.1641
    'thumb_1':  (0.0, 1.1641),
    # 大拇指弯曲: thumb_2(0~0.5864), thumb_3(mimic*0.8024), thumb_4(mimic*0.9487)
    'thumb_2':  (0.0, 0.5864),
    'thumb_3':  (0.0, 0.4705),  # 0.5864 * 0.8024 ≈ 0.4705
    'thumb_4':  (0.0, 0.4463),  # 0.4705 * 0.9487 ≈ 0.4463 (实际 upper=3.14)
}

FIXED_JOINTS = [
    # === 两条腿 ===
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    # === 腰部 ===
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    # === 腕部 ===
    # "left_wrist_pitch_joint",
    # "left_wrist_yaw_joint",
    # "right_wrist_pitch_joint",
    # "right_wrist_yaw_joint",
]

# 获取关节索引
def get_joint_dof_indices(robot: RigidEntity, joint_names: List[str]) -> List[int]:
    """获取指定关节的dof索引"""
    def unwarp(array):
        if isinstance(array, list) and len(array) == 1:
            return array[0]
        return array
    return [unwarp(robot.get_joint(name).dofs_idx_local) for name in joint_names]

left_arm_dofs = get_joint_dof_indices(g1, LEFT_ARM_JOINTS)
right_arm_dofs = get_joint_dof_indices(g1, RIGHT_ARM_JOINTS)
fixed_dofs = get_joint_dof_indices(g1, FIXED_JOINTS)

print(f"\nLeft arm dof indices: {left_arm_dofs}")
print(f"Right arm dof indices: {right_arm_dofs}")
print(f"Fixed dof indices: {fixed_dofs}")

# ========================== 设置控制增益 ==========================
# 注意：需要根据G1的实际参数调整
try:
    # 为手臂设置PD增益
    arm_kp = np.array([100, 100, 50, 50, 60, 60, 60])
    arm_kv = np.array([2, 2, 2, 2, 2, 1, 1])
    
    if left_arm_dofs:
        g1.set_dofs_kp(arm_kp, left_arm_dofs)
        g1.set_dofs_kv(arm_kv, left_arm_dofs)
    
    if right_arm_dofs:
        g1.set_dofs_kp(arm_kp, right_arm_dofs)
        g1.set_dofs_kv(arm_kv, right_arm_dofs)

    if fixed_dofs:
        g1.set_dofs_kp(np.full(len(fixed_dofs), 100), fixed_dofs)
        g1.set_dofs_kv(np.full(len(fixed_dofs), 2), fixed_dofs)
        
    print("Control gains set successfully")
except Exception as e:
    print(f"Warning: Could not set control gains: {e}")

# ========================== 初始站立姿态设置 ==========================

default_wb_qpos = [-0.089,-0.018,-0.006, 0.534,-0.422, 0.001,
                -0.09 ,-0.008, 0.013, 0.531,-0.429, 0.004,
                0, 0.   , 0.   ,
                0.053, 0.486, 0.173, 1.249, 0.167,0.121,-0.032,
                0.054,-0.434,-0.173, 1.249,-0.167, 0.121, 0.032]

def fetch_qpos_by_names(q, joint_names: List[str]) -> List[float]:
    return [q[G1JointIndexURDF[name].value] for name in joint_names]

def set_default_pose(robot: RigidEntity):
    default_q = fetch_qpos_by_names(default_wb_qpos, 
                                    FIXED_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
    dofs_idxs = fixed_dofs + left_arm_dofs + right_arm_dofs
    robot.set_dofs_position(np.array(default_q), dofs_idxs)

def control_standing_pose(robot: RigidEntity):
    standing_poses = fetch_qpos_by_names(default_wb_qpos, FIXED_JOINTS)
    robot.control_dofs_position(np.array(standing_poses), fixed_dofs)

# 设置初始站立姿态
print("Setting initial standing pose...")
set_default_pose(g1)

# ========================== Inspire Hand FTP 控制函数 ==========================

def get_hand_joint_dof_indices(robot: RigidEntity, side: str = 'left') -> dict:
    """
    获取指定侧手部各手指关节的 DOF 索引
    返回: {finger_name: [dof_idx1, dof_idx2, ...], ...}
    """
    result = {}
    for finger_key, joint_suffixes in INSPIRE_FINGER_MAPPING.items():
        dof_indices = []
        for suffix in joint_suffixes:
            joint_name = f'{side}_{suffix}'
            try:
                joint = robot.get_joint(joint_name)
                dof_idx = joint.dofs_idx_local
                if isinstance(dof_idx, list):
                    dof_indices.extend(dof_idx)
                else:
                    dof_indices.append(dof_idx)
            except Exception as e:
                print(f"Warning: Joint {joint_name} not found: {e}")
        result[finger_key] = dof_indices
    return result


def inspire_value_to_joint_angles(
    inspire_value: float,
    finger_key: str,
    side: str = 'left'
) -> List[float]:
    """
    将 Inspire Hand 控制值 (0~1000) 转换为关节角度 (弧度)
    
    Args:
        inspire_value: Inspire Hand 控制值, 0=闭合, 1000=张开
        finger_key: 手指键名 ('little', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rot')
        side: 'left' 或 'right'
    
    Returns:
        对应关节的角度列表 (弧度)
    """
    # 将 0~1000 归一化到 0~1 (0=闭合, 1=张开)
    normalized = np.clip(inspire_value / 1000.0, 0.0, 1.0)
    
    joint_suffixes = INSPIRE_FINGER_MAPPING[finger_key]
    angles = []
    
    for suffix in joint_suffixes:
        # 获取关节范围
        range_key = suffix.replace('_joint', '')
        if range_key in FINGER_JOINT_RANGES:
            min_angle, max_angle = FINGER_JOINT_RANGES[range_key]
        else:
            min_angle, max_angle = 0.0, 1.57  # 默认范围
        
        # Inspire: 0=闭合(max_angle), 1000=张开(min_angle)
        angle = max_angle - normalized * (max_angle - min_angle)
        angles.append(angle)
    
    return angles


def control_inspire_hand(
    robot: RigidEntity,
    angle_set: Union[int, float, List[Union[int, float]]],
    force_set: Union[int, float, List[Union[int, float]], None] = None,
    side: str = 'left',
) -> Tuple[np.ndarray, List[int]]:
    """
    控制 Inspire Hand FTP 灵巧手
    
    Args:
        robot: Genesis RigidEntity 机器人实例
        angle_set: 角度控制值，范围 0~1000
            - 单个值: 应用于所有 6 个控制通道
            - 长度为 6 的列表: [小拇指, 无名指, 中指, 食指, 大拇指弯曲, 大拇指旋转]
            - 0 = 闭合(弯曲), 1000 = 完全张开
        force_set: 力控阈值，范围 0~1000 (Genesis 仿真中暂不支持力控，仅作接口兼容)
            - 单个值: 应用于所有通道
            - 长度为 6 的列表: 各手指力控阈值
            - None: 使用默认值 500
        side: 'left' 或 'right'，控制哪只手
    
    Returns:
        (joint_angles, dof_indices): 关节角度数组和对应的 DOF 索引列表
    
    Example:
        # 张开所有手指
        control_inspire_hand(g1, 1000, side='right')
        
        # 闭合所有手指
        control_inspire_hand(g1, 0, side='right')
        
        # 精细控制: 食指和大拇指闭合，其他张开
        control_inspire_hand(g1, [1000, 1000, 1000, 0, 0, 500], side='right')
    """
    # 处理 angle_set 输入
    if isinstance(angle_set, (int, float)):
        angle_values = [float(angle_set)] * 6
    else:
        angle_values = [float(v) for v in angle_set]
        if len(angle_values) != 6:
            raise ValueError(f"angle_set 列表长度必须为 6，当前为 {len(angle_values)}")
    
    # 处理 force_set 输入 (Genesis 仿真中暂不使用，保留接口兼容性)
    if force_set is None:
        force_values = [500.0] * 6
    elif isinstance(force_set, (int, float)):
        force_values = [float(force_set)] * 6
    else:
        force_values = [float(v) for v in force_set]
        if len(force_values) != 6:
            raise ValueError(f"force_set 列表长度必须为 6，当前为 {len(force_values)}")
    
    # 获取手部关节 DOF 索引
    hand_dof_map = get_hand_joint_dof_indices(robot, side)
    
    # 计算所有关节角度
    all_angles = []
    all_dof_indices = []
    
    for i, finger_key in enumerate(INSPIRE_CONTROL_ORDER):
        inspire_val = angle_values[i]
        joint_angles = inspire_value_to_joint_angles(inspire_val, finger_key, side)
        dof_indices = hand_dof_map.get(finger_key, [])
        
        if len(joint_angles) == len(dof_indices):
            all_angles.extend(joint_angles)
            all_dof_indices.extend(dof_indices)
        else:
            print(f"Warning: Mismatch for {finger_key}: {len(joint_angles)} angles vs {len(dof_indices)} dofs")
    
    return np.array(all_angles), all_dof_indices


def set_hand_position(
    robot: RigidEntity,
    angle_set: Union[int, float, List[Union[int, float]]],
    force_set: Union[int, float, List[Union[int, float]], None] = None,
    side: str = 'left',
):
    """
    直接设置手部关节位置 (无平滑过渡)
    
    Args:
        robot: 机器人实例
        angle_set: Inspire Hand 角度控制值 (0~1000)
        force_set: 力控阈值 (仿真中不使用)
        side: 'left' 或 'right'
    """
    angles, dof_indices = control_inspire_hand(robot, angle_set, force_set, side)
    if len(angles) > 0:
        robot.set_dofs_position(angles, dof_indices)


def control_hand_position(
    robot: RigidEntity,
    angle_set: Union[int, float, List[Union[int, float]]],
    force_set: Union[int, float, List[Union[int, float]], None] = None,
    side: str = 'left',
):
    """
    使用 PD 控制器控制手部关节位置 (有平滑过渡)
    
    Args:
        robot: 机器人实例
        angle_set: Inspire Hand 角度控制值 (0~1000)
        force_set: 力控阈值 (仿真中不使用)
        side: 'left' 或 'right'
    """
    angles, dof_indices = control_inspire_hand(robot, angle_set, force_set, side)
    if len(angles) > 0:
        robot.control_dofs_position(angles, dof_indices)


def open_hand(robot: RigidEntity, side: str = 'left', value: int = 1000, force: int = 500):
    """张开手掌 (所有手指)"""
    control_hand_position(robot, value, force, side=side)


def close_hand(robot: RigidEntity, side: str = 'left', value: int = 0, force: int = 500):
    """闭合手掌 (所有手指)"""
    control_hand_position(robot, value, force, side=side)


def pinch_grasp(robot: RigidEntity, side: str = 'left', thumb_rot: int = 500):
    """
    捏取姿势: 食指和大拇指闭合，其他手指张开
    
    Args:
        robot: 机器人实例
        side: 'left' 或 'right'
        thumb_rot: 大拇指旋转角度 (0~1000)
    """
    # [小拇指, 无名指, 中指, 食指, 大拇指弯曲, 大拇指旋转]
    angle_set = [1000, 1000, 1000, 200, 200, thumb_rot]
    control_hand_position(robot, angle_set, side=side)


def power_grasp(robot: RigidEntity, side: str = 'left', value: int = 200, force: int = 500):
    """
    力量抓取姿势: 所有手指闭合
    
    Args:
        robot: 机器人实例
        side: 'left' 或 'right'
        value: 闭合程度 (0=完全闭合, 1000=完全张开)
    """
    control_hand_position(robot, value, force, side=side)

# ========================== IK和运动规划函数 ==========================

def get_end_effector_link(robot: RigidEntity, side='left'):
    """获取末端执行器link"""
    # 根据g1_arm_IK.py，末端执行器是wrist_yaw关节前推15cm
    ee_name = f'{side}_wrist_yaw_link'
    try:
        return robot.get_link(ee_name)
    except:
        # 尝试其他可能的名称
        for link in robot.links:
            if 'wrist' in link.name.lower() and side in link.name.lower():
                print(f"Using link: {link.name} as end effector for {side} arm")
                return link
    return None

def _quat_wxyz_to_rotmat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    if not isinstance(quat_wxyz, torch.Tensor):
        quat_wxyz = torch.tensor(quat_wxyz, dtype=torch.float32)
    q = quat_wxyz.reshape(4)
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return torch.eye(3, dtype=q.dtype, device=q.device)
    s = 2.0 / n

    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    return torch.tensor(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=q.dtype,
        device=q.device,
    )

def palm_target_to_wrist_target(
    palm_pos_w: torch.Tensor,
    palm_quat_wxyz_w: torch.Tensor,
    wrist_to_palm_offset_m: float = 0.15,
    wrist_to_palm_axis_local: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Emulate Pinocchio's extra EE frame (wrist_yaw + 15cm along local +X).

    In Pinocchio defined:  T_wrist_palm = [I, (0.15, 0, 0)].
    So:  T_world_palm = T_world_wrist * T_wrist_palm
    =>   T_world_wrist = T_world_palm * inv(T_wrist_palm)

    Given desired palm pose in world, return the equivalent wrist target.
    Orientation stays the same because the offset is pure translation.
    """
    if not isinstance(palm_pos_w, torch.Tensor):
        palm_pos_w = torch.tensor(palm_pos_w, dtype=torch.float32)
    if not isinstance(palm_quat_wxyz_w, torch.Tensor):
        palm_quat_wxyz_w = torch.tensor(palm_quat_wxyz_w, dtype=torch.float32)
    
    p_palm = palm_pos_w.reshape(3)
    q_palm = palm_quat_wxyz_w.reshape(4)

    if wrist_to_palm_axis_local is None:
        wrist_to_palm_axis_local = torch.tensor([1.0, 0.0, 0.0], dtype=p_palm.dtype, device=p_palm.device)
    elif not isinstance(wrist_to_palm_axis_local, torch.Tensor):
        wrist_to_palm_axis_local = torch.tensor(wrist_to_palm_axis_local, dtype=p_palm.dtype, device=p_palm.device)

    R_wpalm = _quat_wxyz_to_rotmat(q_palm)
    offset_local = float(wrist_to_palm_offset_m) * wrist_to_palm_axis_local.reshape(3)
    offset_world = R_wpalm @ offset_local

    p_wrist = p_palm - offset_world
    q_wrist = q_palm
    return p_wrist, q_wrist

def euler_to_quat_wxyz(euler_deg: Tuple[float, float, float], order: str = 'xyz', device: str = 'cuda') -> torch.Tensor:
    """
    欧拉角 (度) 转四元数 (w, x, y, z)
    
    Args:
        euler_deg: (roll, pitch, yaw) 欧拉角，单位：度
            - roll:  绕 X 轴旋转
            - pitch: 绕 Y 轴旋转  
            - yaw:   绕 Z 轴旋转
        order: 旋转顺序，默认 'xyz' (先绕X，再绕Y，最后绕Z)
        device: torch 设备 ('cuda' 或 'cpu')
    
    Returns:
        四元数 tensor [w, x, y, z]
    """
    euler_rad = torch.deg2rad(torch.tensor(euler_deg, dtype=torch.float32, device=device))
    roll, pitch, yaw = euler_rad
    
    # 计算各轴的半角
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    
    # XYZ 顺序的四元数乘法
    if order == 'xyz':
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    elif order == 'zyx':
        w = cr * cp * cy - sr * sp * sy
        x = sr * cp * cy + cr * sp * sy
        y = cr * sp * cy - sr * cp * sy
        z = cr * cp * sy + sr * sp * cy
    else:
        # 默认 xyz
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z])


# ========================== 预定义抓取方向 ==========================
# 使用欧拉角定义常用的手掌朝向 (roll, pitch, yaw) 单位：度
# 手掌坐标系: X轴指向手指方向(前), Y轴指向拇指方向(左), Z轴垂直手掌(上)

class GraspDirection:
    """常用抓取方向预设 (欧拉角, 度)"""
    
    # === 基本方向 ===
    PALM_DOWN = (180, 0, 0)       # 手掌朝下，手指向前 (俯抓)
    PALM_UP = (0, 0, 0)           # 手掌朝上，手指向前
    PALM_FORWARD = (90, 0, 0)     # 手掌朝前，手指向上
    PALM_BACKWARD = (-90, 0, 0)   # 手掌朝后，手指向上
    PALM_LEFT = (90, 0, 90)       # 手掌朝左
    PALM_RIGHT = (90, 0, -90)     # 手掌朝右
    
    # === 常用抓取姿态 ===
    TOP_DOWN = (180, 0, 0)        # 从上方垂直向下抓取
    SIDE_LEFT = (90, 0, 90)       # 从左侧水平抓取
    SIDE_RIGHT = (90, 0, -90)     # 从右侧水平抓取
    FRONT = (90, 0, 0)            # 从前方水平抓取
    BACK = (90, 0, 180)           # 从后方水平抓取
    
    # === 带倾斜角度 ===
    TILT_DOWN_30 = (150, 0, 0)    # 向下倾斜30度
    TILT_DOWN_45 = (135, 0, 0)    # 向下倾斜45度
    TILT_DOWN_60 = (120, 0, 0)    # 向下倾斜60度


def solve_ik_for_arm(
    robot: RigidEntity,
    target_pos,
    target_orient=None,
    side='left',
    use_palm_frame: bool = True,
    palm_offset_m: float = 0.15,
    orient_type: str = 'auto',
):
    """
    使用Genesis内置IK求解器
    
    Args:
        robot: 机器人实例
        target_pos: 目标位置 [x, y, z]，支持 torch.Tensor, np.ndarray, list
        target_orient: 目标姿态，支持多种格式:
            - None: 使用默认的 TOP_DOWN 方向
            - 四元数 [w, x, y, z]: 长度为4的数组
            - 欧拉角 (roll, pitch, yaw): 长度为3的元组/列表，单位度
            - GraspDirection 预设: 如 GraspDirection.TOP_DOWN
        side: 'left' 或 'right'
        use_palm_frame: 是否使用手掌坐标系 (True) 或腕部坐标系 (False)
        palm_offset_m: 手掌中心距腕部的偏移量 (默认15cm)
        orient_type: 姿态类型 ('auto', 'quat', 'euler')
            - 'auto': 自动检测 (长度3=欧拉角, 长度4=四元数)
            - 'quat': 强制解释为四元数
            - 'euler': 强制解释为欧拉角
    
    Returns:
        qpos: torch.Tensor 关节角度，失败返回 None
    
    Example:
        使用预设方向 - 从上方抓取\n
        solve_ik_for_arm(robot, [0.3, 0, 0.8], GraspDirection.TOP_DOWN, 'right')
        
        使用欧拉角 - 手掌朝下，手指朝右\n
        solve_ik_for_arm(robot, [0.3, 0, 0.8], (180, 0, 90), 'right')
        
        使用四元数\n
        solve_ik_for_arm(robot, [0.3, 0, 0.8], [1, 0, 0, 0], 'right', orient_type='quat')
    """
    ee_link = get_end_effector_link(robot, side)
    if ee_link is None:
        print(f"Error: Could not find end effector for {side} arm")
        return None
    
    try:
        # 将位置转换为 tensor
        if isinstance(target_pos, torch.Tensor):
            pos = target_pos.clone().reshape(3)
        else:
            pos = torch.tensor(target_pos, dtype=torch.float32, device='cuda').reshape(3)
        
        # 处理姿态输入
        if target_orient is None:
            # 默认从上方抓取
            quat = euler_to_quat_wxyz(GraspDirection.TOP_DOWN, device=pos.device.type)
        else:
            if isinstance(target_orient, torch.Tensor):
                orient = target_orient.flatten()
            else:
                orient = torch.tensor(target_orient, dtype=torch.float32, device=pos.device).flatten()
            
            if orient_type == 'euler' or (orient_type == 'auto' and len(orient) == 3):
                # 欧拉角输入
                quat = euler_to_quat_wxyz(tuple(orient.tolist()), device=pos.device.type)
            elif orient_type == 'quat' or (orient_type == 'auto' and len(orient) == 4):
                # 四元数输入
                quat = orient.reshape(4)
            else:
                raise ValueError(f"无法识别姿态格式: {orient}, 请指定 orient_type='euler' 或 'quat'")

        # Genesis IK must target a real link.
        # To match Pinocchio's added EE frame (15cm in front of wrist_yaw joint),
        # we convert the desired palm pose into an equivalent wrist_yaw_link pose.
        if use_palm_frame:
            pos, quat = palm_target_to_wrist_target(
                palm_pos_w=pos,
                palm_quat_wxyz_w=quat,
                wrist_to_palm_offset_m=palm_offset_m,
            )

        print(f"[IK] Solving for {side} arm, target pos: {pos.tolist()}, quat: {quat.tolist()}")
        ik_start = time.time()
        qpos = robot.inverse_kinematics(
            link=ee_link,
            pos=pos,
            quat=quat,
        )
        ik_elapsed = time.time() - ik_start
        print(f"[IK] Solved in {ik_elapsed:.3f}s")
        return qpos
    except Exception as e:
        print(f"IK solve failed: {e}")
        return None

def plan_motion(robot: RigidEntity, qpos_goal, num_waypoints=200):
    """
    使用OMPL进行运动规划
    需要安装OMPL: pip install ompl
    """
    try:
        print(f"[Motion Planning] Planning path with {num_waypoints} waypoints...")
        plan_start = time.time()
        path = robot.plan_path(
            qpos_goal=qpos_goal,
            num_waypoints=num_waypoints,
        )
        plan_elapsed = time.time() - plan_start
        print(f"[Motion Planning] Completed in {plan_elapsed:.3f}s, path length: {len(path) if path is not None else 0}")
        return path
    except Exception as e:
        print(f"Motion planning failed: {e}")
        return None

def execute_path(robot: RigidEntity, scene, path, arm_dofs=None):
    """执行规划的路径"""
    if path is None:
        print("No path to execute")
        return False
    print(f"Executing path with {len(path)} waypoints...")
    for waypoint in path:
        if arm_dofs:
            # 只控制手臂关节
            arm_qpos = waypoint[arm_dofs]
            robot.control_dofs_position(arm_qpos, arm_dofs)
        else:
            robot.control_dofs_position(waypoint)
        # 保持站立姿态
        control_standing_pose(robot)
        scene.step()
    
    # 等待机器人到达最终位置
    for _ in range(100):
        if arm_dofs:
            robot.control_dofs_position(path[-1][arm_dofs], arm_dofs)
        control_standing_pose(robot)
        scene.step()
    
    return True

class StoolAssemblyTask:
    
    def __init__(self, scene, robot: RigidEntity, stool_top: RigidEntity, stool_legs: List[RigidEntity]):
        self.scene = scene
        self.robot = robot
        self.stool_top = stool_top
        self.stool_legs = stool_legs
        
        self.left_ee = get_end_effector_link(robot, 'left')
        self.right_ee = get_end_effector_link(robot, 'right')
        
    def move_to_home_position(self):
        print("Moving to home position...")
        home_qpos = torch.zeros(self.robot.n_dofs, dtype=torch.float32, device='cuda')
        self.robot.set_dofs_position(home_qpos)
        for _ in range(100):
            self.scene.step()
    
    def grasp_leg(self, leg_index):
        print(f"Grasping leg {leg_index}...")
        
        leg: RigidEntity = self.stool_legs[leg_index]
        leg_pos = leg.get_pos()
        
        pre_grasp_pos = leg_pos.clone()
        pre_grasp_pos[1] += 0.05
        
        grasp_euler = GraspDirection.SIDE_RIGHT  # 从右侧水平抓取
        
        # 1. 移动到预抓取位置
        qpos = solve_ik_for_arm(self.robot, pre_grasp_pos, grasp_euler, 'right')
        if qpos is not None:
            path = plan_motion(self.robot, qpos)
            execute_path(self.robot, self.scene, path, arm_dofs=right_arm_dofs)
        
        # 2. 下降抓取
        qpos = solve_ik_for_arm(self.robot, leg_pos, grasp_euler, 'right')
        if qpos is not None:
            # 只控制手臂关节，不要控制整个机器人
            arm_qpos = qpos[right_arm_dofs]
            self.robot.control_dofs_position(arm_qpos, right_arm_dofs)
            for _ in range(200):  # 增加步数以确保到位
                control_standing_pose(self.robot)  # 保持站立姿态
                self.scene.step()
        
        # 3. 闭合手指
        print("Closing gripper...")
        close_hand(self.robot, side='right', value=200, force=500)
        for _ in range(100):
            control_standing_pose(self.robot)
            self.scene.step()
        
        return True
    
    def insert_leg(self, leg_index, hole_index):
        """将凳腿插入指定的孔"""
        print(f"Inserting leg {leg_index} into hole {hole_index}...")
        
        stool_pos = torch.tensor(STOOL_TOP_POS, dtype=torch.float32, device='cuda')
        hole_offset = torch.tensor(STOOL_HOLE_OFFSETS[hole_index], dtype=torch.float32, device='cuda')
        target_pos = stool_pos + hole_offset
        
        # 计算插入前的位置（孔上方）
        pre_insert_pos = target_pos.clone()
        pre_insert_pos[2] += 0.15
        
        insert_orient = GraspDirection.TOP_DOWN  # 垂直朝下 (欧拉角)
        
        # 1. 移动到孔上方
        qpos = solve_ik_for_arm(self.robot, pre_insert_pos, insert_orient, 'right')
        if qpos is not None:
            path = plan_motion(self.robot, qpos)
            execute_path(self.robot, self.scene, path, arm_dofs=right_arm_dofs)
        
        # 2. 插入
        qpos = solve_ik_for_arm(self.robot, target_pos, insert_orient, 'right')
        if qpos is not None:
            arm_qpos = qpos[right_arm_dofs]
            self.robot.control_dofs_position(arm_qpos, right_arm_dofs)
            for _ in range(300):  # 增加步数
                control_standing_pose(self.robot)
                self.scene.step()
        
        # 3. 松开手指
        print("Opening gripper...")
        open_hand(self.robot, side='right', value=1000)
        for _ in range(100):
            control_standing_pose(self.robot)
            self.scene.step()
        
        # 4. 退出
        qpos = solve_ik_for_arm(self.robot, pre_insert_pos, insert_orient, 'right')
        if qpos is not None:
            arm_qpos = qpos[right_arm_dofs]
            self.robot.control_dofs_position(arm_qpos, right_arm_dofs)
            for _ in range(200):
                control_standing_pose(self.robot)
                self.scene.step()
        
        return True
    
    def run_assembly(self):
        print("\n========== Starting Stool Assembly ==========\n")
        
        for i in range(4):
            print(f"\n--- Processing leg {i+1}/4 ---")
            self.grasp_leg(i)
            self.insert_leg(i, i)
            # self.move_to_home_position()
        
        print("\n========== Assembly Complete ==========\n")

# ========================== 主循环 ==========================

def main():
    print("\n========== G1 Stool Assembly Simulation ==========\n")
    
    # 创建装配任务
    task = StoolAssemblyTask(scene, g1, stool_top, stool_legs)
    
    # 简单的仿真循环，用于查看场景
    print("Running simulation...")
    
    try:
        step = 0
        control_standing_pose(g1)
        for _ in range(100):
            scene.step()
        while True:
            scene.step()
            step += 1
            
            # 每1000步打印一次状态
            if step % 1000 == 0:
                print(f"Step: {step}")
                
            if step == 200:
                print("starting assembly task...")
                task.run_assembly()
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

if __name__ == "__main__":
    main()
