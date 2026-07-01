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
    return pin.SE3(mat[:3, :3], mat[:3, 3])

def interpolate_pose(T_start, T_end, alpha):
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
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero vector")
    return v / n

def make_ee_pose_from_object(T_base_obj,
                            v_long_obj=np.array([1.0, 0.0, 0.0]),  # 物体自身坐标系的长轴方向
                            approach_base=np.array([0.0, 0.0, -1.0]),  # 机器人基坐标系下的接近方向
                            offset_base=np.zeros(3)):  # 机器人基坐标系下手的位置偏移
    """从物体位姿构造末端（L_ee/R_ee）目标位姿。
    约定：机器人 base 坐标系为 X前、Y左、Z上。
    目标：
    - 末端 x 轴：指向物体长边（由 v_long_obj 指定，定义在物体坐标系中）
    - 末端 y 轴：与 approach_base 共面且垂直于 x 轴
    - 末端 z 轴：由 x、y 叉积确定
    Args:
        T_base_obj: (4,4) 物体在 base 下的位姿（Base <- Obj）
        v_long_obj: (3,) 物体长边方向在物体坐标系中的单位向量（例如凳子腿 OBJ 长边≈+Y，则 [0,1,0]）
        approach_base: (3,) base 的“接近方向”，例如 [0,0,-1] 表示从上往下抓取
        offset_base: (3,) base 的额外平移偏置，默认 0。
    Returns:
        pin.SE3: 末端目标位姿（世界系 o/base 下的 L_ee/R_ee）
    """
    R_bo = T_base_obj[:3, :3]
    t_bo = T_base_obj[:3, 3]

    z = _normalize(R_bo @ _normalize(v_long_obj))
    y = _normalize(approach_base)

    # 防止近似平行导致叉积退化
    if abs(np.dot(y, z)) > 0.95:
        z = np.array([0.0, 0.0, 1.0])  # fallback

    x = _normalize(np.cross(y, z))
    z = np.cross(x, y)

    R_be = np.stack([x, y, z], axis=1)
    t_be = t_bo + offset_base
    return pin.SE3(R_be, t_be)

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
        # X轴：向前（手指指向），Y轴：向左（接近方向），Z轴：向上（手腕卷动）
        
        # 末端执行器的Y轴应指向物体中心
        # ee_y = _normalize(-offset_base)
        # if np.linalg.norm(ee_y) < 0.01:
        #     ee_y = np.array([0.0, 1.0, 0.0])
        
        # 末端执行器的Z轴尽量跟随物体Z轴（在水平面的投影）
        # obj_z = obj_rotation[:, 2]
        # obj_z_horizontal = np.array([obj_z[0], obj_z[1], 0.0])
        # if np.linalg.norm(obj_z_horizontal) > 0.1:
        #     ee_z = _normalize(obj_z_horizontal)
        # else:
        #     ee_z = np.array([0.0, 0.0, 1.0])
        
        # X轴通过叉积得到
        # ee_x = _normalize(np.cross(ee_y, ee_z))
        # ee_z = np.cross(ee_x, ee_y)
        
        # R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        # t_base_ee = obj_position + offset_base
        R_base_ee = np.eye(3)
        t_base_ee = obj_position + offset_base
        
    elif grasp_style == 'top':
        # 从上方抓取：末端执行器从上方接近
        # obj_z = obj_rotation[:, 2]
        # obj_z_horizontal = np.array([obj_z[0], obj_z[1], 0.0])
        
        # if np.linalg.norm(obj_z_horizontal) > 0.1:
        #     ee_y = _normalize(obj_z_horizontal)
        # else:
        #     ee_y = np.array([1.0, 0.0, 0.0])
        
        # ee_z = np.array([0.0, 0.0, -1.0])  # Z轴向下
        # ee_x = _normalize(np.cross(ee_y, ee_z))
        # ee_y = np.cross(ee_z, ee_x)
        
        # R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        # t_base_ee = obj_position + np.array([0.0, 0.0, 0.15])  # 在物体上方
        
        ee_y = np.array([0.0, 0.0, -1.0])
        ee_x = np.array([0.0, 1.0, 0.0])
        ee_z = np.array([-1.0, 0.0, 0.0])
        R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        t_base_ee = obj_position + offset_base
        
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

def FK(ik, q):
    '''
    forward kinematics
    param:  q: arm motor joint state (n-joints)
    return: tuple: tuple(left_hand_f, right_hand_f) of (pin.SE3)
    '''
    pin.forwardKinematics(ik.reduced_robot.model, ik.reduced_robot.data, q)
    pin.updateFramePlacements(ik.reduced_robot.model, ik.reduced_robot.data)
    d = ik.reduced_robot.data
    return d.oMf[ik.L_hand_id], d.oMf[ik.R_hand_id]

def IK(ik, q, poseL: pin.SE3, poseR: pin.SE3):
    '''
    inverse kinematics
    param:  poseL: left  hand end-effector pose
            poseR: right hand end-effector pose
    return: q   : arm motor joint state (n-joints)
    '''
    q_now = q
    q_cmd, _ = ik.solve_ik(poseL.homogeneous, poseR.homogeneous, current_lr_arm_motor_q=q_now)
    return q_cmd

def main():
    # 1. Initialize IK
    print("Initializing IK...")
    ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)
    
    # 2. Load Target Pose
    txt_path = os.path.join(current_dir, "test.txt")
    txt_path2 = os.path.join(current_dir, "test2.txt")
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
        return

    print(f"Loading target pose from {txt_path}")
    T_camera_object_np = load_pose_from_txt(txt_path)
    T_camera_object_np2 = load_pose_from_txt(txt_path2)
    
    # 3. Transform to Base Frame
    # Assuming default waist angles (0,0,0)
    print("Transforming pose to Base Frame...")
    T_base_camera = get_camera_pose_in_base_frame(0, 0, 0)
    T_base_target_np = get_target_pose_in_base_frame(T_base_camera, T_camera_object_np)
    T_base_target_np2 = get_target_pose_in_base_frame(T_base_camera, T_camera_object_np2)
    
    # T_target_R = make_ee_pose_from_object(
    #                                     T_base_target_np, 
    #                                     v_long_obj=[0,0,1], 
    #                                     approach_base=[0,1,0], 
    #                                     offset_base=[0,-0.05,0]
    #                                     )
    # T_target_R2 = make_ee_pose_from_object(
    #                                     T_base_target_np2, 
    #                                     v_long_obj=[0,0,1], 
    #                                     approach_base=[0,0,-1], 
    #                                     offset_base=[0,0,0]
    #                                     )
    
    T_target_R = pin.SE3(
        np.eye(3),
        np.array([0.2196, -0.2503, 0.94086])
    )
    
    # 4. Define Home Pose
    # "Forearm 90 deg lifted, palms relative"
    # Position: In front of chest.
    # Left Hand: x=0.3, y=0.25, z=0.1
    # Right Hand: x=0.3, y=-0.25, z=0.1
    # Rotation:
    # Left Hand: Palm facing Right (-Y). 
    # Right Hand: Palm facing Left (+Y).
    
    # Construct rotations
    # Identity: X forward, Y left, Z up.
    # Right Hand (Palm Left +Y): Identity is close (Palm is Y).
    # Left Hand (Palm Right -Y): Rotate 180 around X? Y becomes -Y. Z becomes -Z.
    
    R_home_L = np.eye(3)
    R_home_R = np.eye(3)
    P_home_L = np.array([0.3, 0.25, 0.1])
    P_home_R = np.array([0.3, -0.25, 0.1])
    q_home = [-0.06300107389688492, 1.6363739967346191, 0.06309694796800613, 1.433660864830017, -0.06913699209690094, -0.01609145849943161, -0.023728765547275543, -1.5769442319869995, -0.06358829885721207, 1.4105912446975708, 0.04851214215159416, 0.04398690164089203]
    T_home_L, T_home_R = FK(ik, np.array(q_home))
    
    # T_home_L = pin.SE3(R_home_L, P_home_L)
    # T_home_R = pin.SE3(R_home_R, P_home_R)
    
    # 5. Loop
    # q_current = ik.init_data.copy()
    q_current = np.array(q_home)
    
    # Move to Home first (quickly or directly)
    print("Moving to Initial Pose...")
    steps_init = 50
    # We don't know where we start, so let's just interpolate from current q's FK?
    # Or just jump to Home? IK might jump.
    # Let's interpolate from "current" (which is zero/init) to Home.
    
    # Get current FK
    L_start, R_start = ik.forward_kinematics(q_current)
    # Note: forward_kinematics returns pin.SE3 objects
    
    for i in range(steps_init):
        alpha = i / steps_init
        T_curr_L = interpolate_pose(L_start, T_home_L, alpha)
        T_curr_R = interpolate_pose(R_start, T_home_R, alpha)
        
        q_sol, _ = ik.solve_ik(
            left_wrist=T_curr_L.homogeneous,
            right_wrist=T_curr_R.homogeneous,
            current_lr_arm_motor_q=q_current
        )
        q_current = q_sol
        time.sleep(0.01)
        
    print("Reached Initial Pose. Starting Loop. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Move to Target (Right Hand only, Left stays at Home)
            steps = 100
            for i in range(steps):
                alpha = i / steps
                # Interpolate Right Hand
                T_curr_R = interpolate_pose(T_home_R, T_target_R, alpha)
                # Keep Left Hand at Home
                T_curr_L = T_home_L
                
                q_sol, _ = ik.solve_ik(
                    left_wrist=T_curr_L.homogeneous,
                    right_wrist=T_curr_R.homogeneous,
                    current_lr_arm_motor_q=q_current
                )
                q_current = q_sol
                time.sleep(0.02)
            
            time.sleep(1.0)

            # for i in range(steps):
            #     alpha = i / steps
            #     # Interpolate Right Hand back
            #     T_curr_R = interpolate_pose(T_target_R, T_target_R2, alpha)
            #     T_curr_L = T_home_L
                
            #     q_sol, _ = ik.solve_ik(
            #         left_wrist=T_curr_L.homogeneous,
            #         right_wrist=T_curr_R.homogeneous,
            #         current_lr_arm_motor_q=q_current
            #     )
            #     q_current = q_sol
            #     time.sleep(0.02)

            # time.sleep(1.0)
            
            # for i in range(steps):
            #     alpha = i / steps
            #     # Interpolate Right Hand back
            #     T_curr_R = interpolate_pose(T_target_R2, T_home_R, alpha)
            #     T_curr_L = T_home_L
                
            #     q_sol, _ = ik.solve_ik(
            #         left_wrist=T_curr_L.homogeneous,
            #         right_wrist=T_curr_R.homogeneous,
            #         current_lr_arm_motor_q=q_current
            #     )
            #     q_current = q_sol
            #     time.sleep(0.02)
                
            # time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()
