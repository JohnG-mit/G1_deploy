# G1机器人Genesis仿真与OMPL运动规划指南

## 概述

本指南介绍如何在Genesis仿真环境中搭建G1机器人凳子装配场景，并使用OMPL进行无碰撞运动规划。

## 目录结构

```
deploy_real/pose/genesis/
├── g1_stool_assembly_sim.py      # 主仿真脚本
├── g1_genesis_controller.py      # G1控制器封装
├── helloworld.py                 # Genesis基础示例
└── doc/                          # Genesis文档
```

## 关键概念

### 1. Genesis IK求解

Genesis内置了IK求解器，支持单link和多link IK：

```python
# 单link IK
qpos = robot.inverse_kinematics(
    link=end_effector,
    pos=np.array([x, y, z]),
    quat=np.array([w, x, y, z]),  # 注意：Genesis使用w-x-y-z格式
)

# 多link IK (双臂)
qpos = robot.inverse_kinematics_multilink(
    links=[left_ee, right_ee],
    poss=[left_pos, right_pos],
    quats=[left_quat, right_quat],
    rot_mask=[False, False, True],  # 可选：限制旋转方向
)
```

### 2. OMPL运动规划

Genesis集成了OMPL库进行碰撞检测和路径规划：

```python
# 安装OMPL
# pip install ompl

# 规划无碰撞路径
path = robot.plan_path(
    qpos_goal=target_qpos,
    num_waypoints=200,  # 路径点数量，200个点≈2秒轨迹（dt=0.01）
)

# 执行路径
for waypoint in path:
    robot.control_dofs_position(waypoint)
    scene.step()
```

### 3. 控制增益设置

对于G1机器人，需要为各关节设置合适的PD增益：

```python
# 设置位置增益
robot.set_dofs_kp(
    kp=np.array([...]),
    dofs_idx_local=[...],
)

# 设置速度增益
robot.set_dofs_kv(
    kv=np.array([...]),
    dofs_idx_local=[...],
)

# 设置力矩限制
robot.set_dofs_force_range(
    lower=np.array([...]),
    upper=np.array([...]),
    dofs_idx_local=[...],
)
```

## G1机器人关节映射

根据URDF文件`g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`，G1 29DOF版本的关节如下：

### 腿部 (12个关节)
| 关节名 | 描述 |
|--------|------|
| left/right_hip_pitch_joint | 髋关节俯仰 |
| left/right_hip_roll_joint | 髋关节侧摆 |
| left/right_hip_yaw_joint | 髋关节旋转 |
| left/right_knee_joint | 膝关节 |
| left/right_ankle_pitch_joint | 踝关节俯仰 |
| left/right_ankle_roll_joint | 踝关节侧摆 |

### 腰部 (3个关节)
| 关节名 | 描述 |
|--------|------|
| waist_yaw_joint | 腰部旋转 |
| waist_roll_joint | 腰部侧摆 |
| waist_pitch_joint | 腰部俯仰 |

### 手臂 (14个关节，左右各7个)
| 关节名 | 描述 |
|--------|------|
| left/right_shoulder_pitch_joint | 肩关节俯仰 |
| left/right_shoulder_roll_joint | 肩关节侧摆 |
| left/right_shoulder_yaw_joint | 肩关节旋转 |
| left/right_elbow_joint | 肘关节 |
| left/right_wrist_roll_joint | 腕关节旋转 |
| left/right_wrist_pitch_joint | 腕关节俯仰 |
| left/right_wrist_yaw_joint | 腕关节偏航 |

### 灵巧手 (Inspire Hand FTP)
URDF中还包含了Inspire Hand的手指关节，包括：
- 拇指 (thumb): 4个关节
- 食指 (index): 2个关节
- 中指 (middle): 2个关节
- 无名指 (ring): 2个关节
- 小指 (little): 2个关节

## 与Pinocchio IK的集成

项目中已有的`g1_arm_IK.py`使用Pinocchio+CasADi进行IK求解。关键点：

1. **锁定关节**: 下肢、腰部、wrist_pitch和所有手指关节被锁定
2. **活动自由度**: 保留12个上肢自由度（左右各6个）
3. **末端执行器**: 定义为wrist_yaw关节前推15cm处（掌心位置）

```python
# 从g1_arm_IK.py导入
from deploy_real.g1_arm_IK import G1_29_ArmIK

ik_solver = G1_29_ArmIK(Unit_Test=False, Visualization=False)

# 求解双臂IK
# left_wrist, right_wrist: 4x4齐次变换矩阵
sol_q, sol_tau = ik_solver.solve_ik(left_wrist, right_wrist)
```

## 凳子装配任务

### 场景设置

1. **桌子**: 长1.16m, 宽0.77m, 高0.70m, 位于机器人身前5cm
2. **凳面**: 放置在桌子上（固定）
3. **凳腿**: 4个，放置在桌子上待抓取（可移动）

### 模型文件

```
deploy_real/assets_obj/
├── stool/
│   ├── stool-scale.obj    # 凳面模型
│   └── stool.mtl          # 材质
└── stool-leg/
    ├── stool-leg-scaled.obj  # 凳腿模型
    └── stool-leg.mtl         # 材质
```

### 装配流程

1. **移动到初始位置** - 安全姿态
2. **抓取凳腿**:
   - IK求解预抓取位置（凳腿上方）
   - OMPL规划无碰撞路径
   - 执行路径
   - 下降抓取
   - 闭合灵巧手
3. **插入凳腿**:
   - IK求解孔上方位置
   - OMPL规划无碰撞路径
   - 执行路径
   - 插入
   - 松开灵巧手
4. **重复3次**完成其余凳腿

## 运行仿真

```bash
# 进入项目目录
cd /home/johng/repo/G1_deploy

# 运行主仿真
python deploy_real/pose/genesis/g1_stool_assembly_sim.py

# 或运行控制器测试
python deploy_real/pose/genesis/g1_genesis_controller.py
```

## 注意事项

1. **OMPL安装**: Genesis的运动规划需要OMPL库
   ```bash
   pip install ompl
   ```

2. **碰撞检测**: 确保在`RigidOptions`中启用碰撞检测
   ```python
   rigid_options=gs.options.RigidOptions(
       enable_collision=True,
   )
   ```

3. **四元数格式**: Genesis使用`w-x-y-z`格式，Pinocchio/SciPy使用`x-y-z-w`格式

4. **控制方式**:
   - `set_dofs_position()`: 直接设置位置（瞬移，不遵守物理）
   - `control_dofs_position()`: PD控制到目标位置（物理仿真）

5. **关节限位**: 确保IK解在关节限位范围内
   ```python
   rigid_options=gs.options.RigidOptions(
       enable_joint_limit=True,
   )
   ```

## 参考文档

- Genesis官方文档: `deploy_real/pose/genesis/doc/`
- [Inverse Kinematics & Motion Planning](doc/source/user_guide/getting_started/inverse_kinematics_motion_planning.md)
- [Advanced IK](doc/source/user_guide/getting_started/advanced_ik.md)
- [Control Your Robot](doc/source/user_guide/getting_started/control_your_robot.md)
