"""
G1机器人仿真工具类
集成Genesis仿真与Pinocchio IK解算
"""

import os
import sys
from pathlib import Path
import numpy as np
import genesis as gs

# 路径设置
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.parent  # G1_deploy目录
sys.path.insert(0, str(project_root))

# 导入现有的IK解算器
try:
    from deploy_real.g1_arm_IK import G1_29_ArmIK
    HAS_PINOCCHIO_IK = True
    print("Pinocchio IK solver available")
except ImportError as e:
    HAS_PINOCCHIO_IK = False
    print(f"Pinocchio IK solver not available: {e}")


class G1GenesisController:
    """
    G1机器人Genesis仿真控制器
    结合Genesis内置IK和Pinocchio IK
    """
    
    # G1 29DOF 关节映射
    # 根据URDF和g1_arm_IK.py
    JOINT_MAPPING = {
        # 腿部关节 (12个)
        'left_hip_pitch': 0,
        'left_hip_roll': 1,
        'left_hip_yaw': 2,
        'left_knee': 3,
        'left_ankle_pitch': 4,
        'left_ankle_roll': 5,
        'right_hip_pitch': 6,
        'right_hip_roll': 7,
        'right_hip_yaw': 8,
        'right_knee': 9,
        'right_ankle_pitch': 10,
        'right_ankle_roll': 11,
        
        # 腰部关节 (3个)
        'waist_yaw': 12,
        'waist_roll': 13,
        'waist_pitch': 14,
        
        # 左臂关节 (7个)
        'left_shoulder_pitch': 15,
        'left_shoulder_roll': 16,
        'left_shoulder_yaw': 17,
        'left_elbow': 18,
        'left_wrist_roll': 19,
        'left_wrist_pitch': 20,
        'left_wrist_yaw': 21,
        
        # 右臂关节 (7个)
        'right_shoulder_pitch': 22,
        'right_shoulder_roll': 23,
        'right_shoulder_yaw': 24,
        'right_elbow': 25,
        'right_wrist_roll': 26,
        'right_wrist_pitch': 27,
        'right_wrist_yaw': 28,
    }
    
    # 保留用于IK解算的上肢关节（与g1_arm_IK.py一致）
    ARM_IK_JOINTS = [
        'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
        'left_elbow', 'left_wrist_roll', 'left_wrist_yaw',  # 注意: wrist_pitch被锁定
        'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
        'right_elbow', 'right_wrist_roll', 'right_wrist_yaw',
    ]
    
    def __init__(self, robot_entity, scene, use_pinocchio_ik=True):
        """
        初始化控制器
        
        Args:
            robot_entity: Genesis中加载的G1机器人实体
            scene: Genesis场景
            use_pinocchio_ik: 是否使用Pinocchio IK（更精确但较慢）
        """
        self.robot = robot_entity
        self.scene = scene
        self.use_pinocchio_ik = use_pinocchio_ik and HAS_PINOCCHIO_IK
        
        # 初始化Pinocchio IK解算器
        if self.use_pinocchio_ik:
            self.pin_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
            print("Using Pinocchio IK solver")
        else:
            self.pin_ik = None
            print("Using Genesis built-in IK solver")
        
        # 缓存关节索引
        self._cache_joint_indices()
        
        # 设置控制增益
        self._setup_control_gains()
    
    def _cache_joint_indices(self):
        """缓存Genesis中的关节索引"""
        self.joint_idx_map = {}
        
        for joint in self.robot.joints:
            name = joint.name.replace('_joint', '')
            if name in self.JOINT_MAPPING:
                self.joint_idx_map[name] = joint.dofs_idx_local
        
        # 获取左右臂的dof索引
        self.left_arm_dofs = []
        self.right_arm_dofs = []
        
        for jname in self.ARM_IK_JOINTS:
            if jname in self.joint_idx_map:
                if 'left' in jname:
                    self.left_arm_dofs.append(self.joint_idx_map[jname])
                else:
                    self.right_arm_dofs.append(self.joint_idx_map[jname])
        
        print(f"Left arm DOFs: {self.left_arm_dofs}")
        print(f"Right arm DOFs: {self.right_arm_dofs}")
    
    def _setup_control_gains(self):
        """设置PD控制增益"""
        # 手臂关节增益
        arm_kp = np.array([100, 100, 50, 50, 60, 60])
        arm_kv = np.array([2, 2, 2, 2, 1, 1])
        
        try:
            if self.left_arm_dofs:
                self.robot.set_dofs_kp(arm_kp, self.left_arm_dofs)
                self.robot.set_dofs_kv(arm_kv, self.left_arm_dofs)
            
            if self.right_arm_dofs:
                self.robot.set_dofs_kp(arm_kp, self.right_arm_dofs)
                self.robot.set_dofs_kv(arm_kv, self.right_arm_dofs)
        except Exception as e:
            print(f"Warning: Could not set control gains: {e}")
    
    def get_end_effector_pose(self, side='left'):
        """
        获取末端执行器位姿
        
        Returns:
            pos: [x, y, z] 位置
            quat: [w, x, y, z] 四元数
        """
        ee_link_name = f'{side}_wrist_yaw_link'
        
        try:
            ee_link = self.robot.get_link(ee_link_name)
            pos = ee_link.get_pos()
            quat = ee_link.get_quat()
            return pos, quat
        except Exception as e:
            print(f"Error getting EE pose: {e}")
            return None, None
    
    def solve_arm_ik_pinocchio(self, left_pose, right_pose, current_q=None):
        """
        使用Pinocchio求解双臂IK
        
        Args:
            left_pose: 4x4 homogeneous matrix for left hand
            right_pose: 4x4 homogeneous matrix for right hand
            current_q: 当前关节角度 (12维，上肢)
            
        Returns:
            arm_q: 上肢关节角度 (12维)
            arm_tau: 前馈力矩
        """
        if self.pin_ik is None:
            raise RuntimeError("Pinocchio IK solver not initialized")
        
        return self.pin_ik.solve_ik(left_pose, right_pose, current_q)
    
    def solve_arm_ik_genesis(self, target_pos, target_quat, side='left'):
        """
        使用Genesis内置IK求解单臂
        
        Args:
            target_pos: [x, y, z] 目标位置
            target_quat: [w, x, y, z] 目标四元数
            side: 'left' 或 'right'
            
        Returns:
            qpos: 全身关节角度
        """
        ee_link_name = f'{side}_wrist_yaw_link'
        
        try:
            ee_link = self.robot.get_link(ee_link_name)
            qpos = self.robot.inverse_kinematics(
                link=ee_link,
                pos=np.array(target_pos),
                quat=np.array(target_quat),
            )
            return qpos
        except Exception as e:
            print(f"Genesis IK failed: {e}")
            return None
    
    def solve_dual_arm_ik_genesis(self, left_pos, left_quat, right_pos, right_quat):
        """
        使用Genesis求解双臂IK（使用multilink IK）
        
        Args:
            left_pos, left_quat: 左手目标位姿
            right_pos, right_quat: 右手目标位姿
            
        Returns:
            qpos: 全身关节角度
        """
        try:
            left_ee = self.robot.get_link('left_wrist_yaw_link')
            right_ee = self.robot.get_link('right_wrist_yaw_link')
            
            qpos = self.robot.inverse_kinematics_multilink(
                links=[left_ee, right_ee],
                poss=[np.array(left_pos), np.array(right_pos)],
                quats=[np.array(left_quat), np.array(right_quat)],
            )
            return qpos
        except Exception as e:
            print(f"Dual arm IK failed: {e}")
            return None
    
    def plan_collision_free_path(self, qpos_goal, num_waypoints=200):
        """
        使用OMPL规划无碰撞路径
        
        Args:
            qpos_goal: 目标关节位置
            num_waypoints: 路径点数量
            
        Returns:
            path: 路径点列表
        """
        try:
            path = self.robot.plan_path(
                qpos_goal=qpos_goal,
                num_waypoints=num_waypoints,
            )
            return path
        except Exception as e:
            print(f"Motion planning failed: {e}")
            return None
    
    def execute_path(self, path, arm_dofs=None):
        """
        执行规划的路径
        
        Args:
            path: 路径点列表
            arm_dofs: 如果指定，只控制指定的关节
        """
        if path is None:
            print("No path to execute")
            return False
        
        for waypoint in path:
            if arm_dofs is not None:
                self.robot.control_dofs_position(waypoint[arm_dofs], arm_dofs)
            else:
                self.robot.control_dofs_position(waypoint)
            self.scene.step()
        
        # 等待稳定
        for _ in range(50):
            self.scene.step()
        
        return True
    
    def control_arm(self, arm_q, side='left'):
        """
        控制单臂到指定关节角度
        
        Args:
            arm_q: 手臂关节角度 (6维)
            side: 'left' 或 'right'
        """
        if side == 'left':
            dofs = self.left_arm_dofs
        else:
            dofs = self.right_arm_dofs
        
        if dofs and len(arm_q) == len(dofs):
            self.robot.control_dofs_position(arm_q, dofs)
    
    def control_dual_arms(self, left_q, right_q):
        """
        同时控制双臂
        
        Args:
            left_q: 左臂关节角度 (6维)
            right_q: 右臂关节角度 (6维)
        """
        self.control_arm(left_q, 'left')
        self.control_arm(right_q, 'right')


def create_pose_matrix(pos, quat_wxyz):
    """
    创建4x4齐次变换矩阵
    
    Args:
        pos: [x, y, z] 位置
        quat_wxyz: [w, x, y, z] 四元数
        
    Returns:
        4x4 numpy array
    """
    from scipy.spatial.transform import Rotation
    
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # xyzw格式
    R = r.as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    
    return T


# ========================== 测试代码 ==========================

if __name__ == "__main__":
    # 初始化Genesis
    gs.init(
        backend=gs.gpu,
        logging_level='warning',
    )
    
    # 创建场景
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.7),
            camera_fov=45,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
    )
    
    # 添加地面
    plane = scene.add_entity(gs.morphs.Plane())
    
    # 添加G1机器人
    g1 = scene.add_entity(
        gs.morphs.URDF(
            file='deploy_real/assets/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf',
            pos=(0, 0, 0.75),
            euler=(0, 0, 0),
            scale=1.0,
        ),
    )
    
    # 构建场景
    scene.build()
    
    # 创建控制器
    controller = G1GenesisController(g1, scene, use_pinocchio_ik=True)
    
    # 打印关节信息
    print("\n========== Joint Information ==========")
    for joint in g1.joints:
        print(f"Joint: {joint.name}, DOF idx: {joint.dofs_idx_local}")
    
    # 简单测试
    print("\n========== Running Test ==========")
    
    try:
        for i in range(1000):
            scene.step()
            
            if i % 100 == 0:
                left_pos, left_quat = controller.get_end_effector_pose('left')
                right_pos, right_quat = controller.get_end_effector_pose('right')
                
                if left_pos is not None:
                    print(f"Step {i}: Left EE pos: {left_pos}")
                    
    except KeyboardInterrupt:
        print("\nTest stopped by user")
