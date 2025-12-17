import sys
import os
import numpy as np
import pinocchio as pin

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.abspath(parent_dir))
from g1_arm_IK import G1_29_ArmIK

def load_pose_from_txt(path):
    arr = np.loadtxt(path).reshape(4, 4)
    R = arr[:3, :3]
    p = arr[:3, 3]
    return pin.SE3(R, p)

ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)

# 当前关节，用于初始化和保持对侧不乱动
q_now = ik.init_data.copy()  # 或者你从真实机器人读一帧 current_q

# 目标位姿（来自 FoundationPose）
T_target = load_pose_from_txt(f"{current_dir}/test.txt")  # pin.SE3
# 左手就固定在现在的位置：先算一下当前 FK
L_now, R_now = ik.forward_kinematics(q_now)

q_sol, _ = ik.solve_ik(
    left_wrist=L_now.homogeneous,
    right_wrist=T_target.homogeneous,
    current_lr_arm_motor_q=q_now,
)
