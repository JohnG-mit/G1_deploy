"""
Genesis仿真入门示例
展示G1机器人、桌子和凳子的基础场景搭建
"""

import os
import sys
from pathlib import Path
import numpy as np
import genesis as gs

# 路径设置 - 确保从项目根目录运行时能找到资源文件
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
project_root = parent_dir.parent.parent  # G1_deploy目录
sys.path.insert(0, str(project_root))

# 切换工作目录到项目根目录（确保相对路径正确）
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

# ========================== 初始化Genesis ==========================
gs.init(backend=gs.gpu)

# ========================== 创建场景 ==========================
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(2.0, -2.0, 1.5),  # 调整相机位置更好地观察场景
        camera_lookat=(0.5, 0.0, 0.7),
        camera_fov=45,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,      # 显示原点坐标系
        world_frame_size=0.5,       # 坐标系长度(米)
        show_link_frame=False,      # 不显示实体链接坐标系
        show_cameras=False,         # 不显示相机网格和视锥
        plane_reflection=True,      # 开启平面反射
        ambient_light=(0.3, 0.3, 0.3),  # 环境光
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.81),
    ),
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_joint_limit=True,
    ),
    renderer=gs.renderers.Rasterizer(),  # 使用光栅化渲染器
)

# ========================== 添加地面 ==========================
plane = scene.add_entity(gs.morphs.Plane())

# ========================== 添加桌子 ==========================
# 桌子尺寸: 长1.16m, 宽0.77m, 高0.70m
# 位置: 机器人身前约48.5cm处（桌子中心）
table = scene.add_entity(
    gs.morphs.Box(
        size=(0.77, 1.16, 0.70),
        pos=(0.485, 0.0, 0.35),
        fixed=True,
    ),
    surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2, 1.0)),  # 木色
)

# ========================== 添加凳面 ==========================
# 放在桌子上方
stool_top = scene.add_entity(
    gs.morphs.Mesh(
        file='deploy_real/assets_obj/stool/stool-scale.obj',
        pos=(0.485, 0.0, 0.75),  # 放在桌面上
        euler=(90, 0, 0),
        scale=1.0,
        fixed=True,  # 凳面固定
    ),
)

# ========================== 添加凳腿 (4个) ==========================
stool_leg_positions = [
    (0.35, 0.25, 0.72),   # 左前
    (0.35, -0.25, 0.72),  # 右前
    (0.60, 0.25, 0.72),   # 左后
    (0.60, -0.25, 0.72),  # 右后
]

stool_legs = []
for i, pos in enumerate(stool_leg_positions):
    leg = scene.add_entity(
        gs.morphs.Mesh(
            file='deploy_real/assets_obj/stool-leg/stool-leg-scaled.obj',
            pos=pos,
            euler=(0, 0, 0),
            scale=1.0,
            fixed=False,  # 凳腿可以被抓取
        ),
    )
    stool_legs.append(leg)

# ========================== 添加G1机器人 ==========================
# fixed=True: 固定基座(pelvis)，机器人不会因重力倒下
g1 = scene.add_entity(
    gs.morphs.URDF(
        file='deploy_real/assets/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf',
        pos=(0, 0, 0.75),  # 机器人站立高度
        euler=(0, 0, 0),
        scale=1.0,
        fixed=True,  # 固定基座防止倒下
    ),
)

# ========================== 构建场景 ==========================
scene.build()

# ========================== 打印机器人信息 ==========================
print("\n========== G1 Robot Info ==========")
print(f"Number of joints: {g1.n_joints}")
print(f"Number of dofs: {g1.n_dofs}")
print(f"Number of links: {g1.n_links}")

print("\nJoint names:")
for i, joint in enumerate(g1.joints):
    print(f"  [{i:2d}] {joint.name}")

# ========================== 设置初始站立姿态 ==========================
# 手臂自然下垂的姿态
ARM_JOINTS = [
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

# 设置手臂初始姿态（略微弯曲）
arm_init_angles = [
    0.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0,   # 左臂
    0.0, -0.2, 0.0, 0.3, 0.0, 0.0, 0.0,  # 右臂
]

for joint_name, angle in zip(ARM_JOINTS, arm_init_angles):
    try:
        joint = g1.get_joint(joint_name)
        g1.set_dofs_position(np.array([angle]), [joint.dofs_idx_local])
    except:
        pass

print("\nInitial pose set")

# ========================== 主仿真循环 ==========================
print("\n========== Starting Simulation ==========")
print("Press Ctrl+C to exit")

try:
    for i in range(10000):
        scene.step()
        
        # 每1000步打印一次状态
        if i % 1000 == 0:
            print(f"Step: {i}")
            
except KeyboardInterrupt:
    print("\nSimulation stopped by user")