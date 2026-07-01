import numpy as np
from scipy.spatial.transform import Rotation as R

def get_target_pose_in_base_frame(T_base_camera, T_camera_object, T_grasp_offset=None):
    """
    计算机器人基坐标系下的末端目标位姿
    
    Args:
        T_base_camera (np.array): 4x4 矩阵, 相机在基坐标系下的位姿
        T_camera_object (np.array): 4x4 矩阵, 物体在相机坐标系下的位姿 (FoundationPose 输出)
        T_grasp_offset (np.array): 4x4 矩阵, 期望的抓取点相对于物体的位姿 (默认为重合)
        
    Returns:
        T_base_target (np.array): 4x4 矩阵, 目标物体在机器人基坐标系下的位姿
    """
    # 链式变换: Base -> Camera -> Object -> GraspTarget
    T_base_object = T_base_camera @ T_camera_object
    if T_grasp_offset is None:
        return T_base_object
    else:
        T_base_target = T_base_object @ T_grasp_offset
    
    return T_base_target


def get_camera_pose_in_base_frame(waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0):
    """
    构建 T_base_camera (Optical Frame) 矩阵。
    基于 G1 URDF 参数和相机安装角度。
    
    Args:
        waist_yaw (float): 腰部偏航角 (rad), 默认为 0
        waist_roll (float): 腰部横滚角 (rad), 默认为 0
        waist_pitch (float): 腰部俯仰角 (rad), 默认为 0
        
    Returns:
        T_base_optical (np.array): 4x4 变换矩阵，从 Optical Frame 到 Base Frame
    """
    # 1. 定义 URDF 中的固定参数 (单位: 米, 弧度)
    # ---------------------------------------------------------
    # 链路偏移 (从 URDF 提取)
    # Pelvis -> Waist Yaw: 0
    # Waist Yaw -> Waist Roll: [-0.0039635, 0, 0.044]
    # Waist Roll -> Torso (Waist Pitch): [0, 0, 0]
    # Torso -> D435 Link: [0.0576235, 0.01753, 0.42987]
    
    offset_pelvis_to_torso = np.array([-0.0039635, 0.0, 0.044])
    offset_torso_to_cam    = np.array([0.0576235, 0.01753, 0.42987])
    
    # 相机安装角度 (URDF d435_joint rpy)
    # pitch = 0.83077... rad (约 47.6度, 低头)
    cam_mount_pitch = 0.8307767239493009
    
    # 2. 计算 Base (Pelvis) -> Camera Link 的变换
    
    # T_pelvis_yaw
    R_yaw = R.from_euler('z', waist_yaw).as_matrix()
    T_pelvis_yaw = np.eye(4)
    T_pelvis_yaw[:3, :3] = R_yaw
    
    # T_yaw_roll
    R_roll = R.from_euler('x', waist_roll).as_matrix()
    T_yaw_roll = np.eye(4)
    T_yaw_roll[:3, :3] = R_roll
    T_yaw_roll[:3, 3] = offset_pelvis_to_torso # 近似放在这一级
    
    # T_roll_pitch (Torso)
    R_pitch = R.from_euler('y', waist_pitch).as_matrix()
    T_roll_torso = np.eye(4)
    T_roll_torso[:3, :3] = R_pitch
    
    # T_torso_camlink
    R_mount = R.from_euler('y', cam_mount_pitch).as_matrix()
    T_torso_camlink = np.eye(4)
    T_torso_camlink[:3, :3] = R_mount
    T_torso_camlink[:3, 3] = offset_torso_to_cam
    
    # 级联: Base -> Yaw -> Roll -> Torso -> CamLink
    T_base_link = T_pelvis_yaw @ T_yaw_roll @ T_roll_torso @ T_torso_camlink
    
    # 3. 计算 Camera Link -> Optical Frame 的变换
    # ---------------------------------------------------------
    # Link Frame: X前, Y左, Z上
    # Optical Frame: Z前, X右, Y下
    # 变换矩阵 R_link_optical (列向量为 Optical 轴在 Link 系中的表示)
    # X_opt (Right) = -Y_link = [0, -1, 0]
    # Y_opt (Down)  = -Z_link = [0, 0, -1]
    # Z_opt (Fwd)   =  X_link = [1, 0, 0]
    
    R_link_optical = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    
    T_link_optical = np.eye(4)
    T_link_optical[:3, :3] = R_link_optical
    
    # 4. 最终变换
    # ---------------------------------------------------------
    T_base_optical = T_base_link @ T_link_optical
    
    return T_base_optical


def plot_coordinate_frame(ax, T, label, scale=0.1):
    """辅助函数：绘制坐标系"""
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]
    
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, normalize=True)
    ax.text(origin[0], origin[1], origin[2], label)


if __name__ == "__main__":
    # 打印静态变换矩阵 (假设腰部角度为0)
    np.set_printoptions(precision=4, suppress=True)
    T_base_optical = get_camera_pose_in_base_frame()
    print("T_base_camera (Optical) with 0 joint angles:\n", T_base_optical)
    
    # 验证: 相机光轴(Optical Z)在Base系下的方向
    # 应该是 X轴向下倾斜 47.6度
    z_axis_optical = T_base_optical[:3, 2] # 旋转矩阵第三列
    print("\nOptical Z axis in Base Frame:", z_axis_optical)
    
    pitch_angle = np.arctan2(-z_axis_optical[2], z_axis_optical[0]) * 180 / np.pi
    print(f"Pitch angle (deg): {pitch_angle:.2f}") # 应该接近 47.6

    # --- 可视化 ---
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. 绘制 Base 坐标系
        plot_coordinate_frame(ax, np.eye(4), "Base", scale=0.2)
        
        # 2. 绘制 Camera Optical 坐标系
        plot_coordinate_frame(ax, T_base_optical, "Cam_Opt", scale=0.15)
        
        # 3. 模拟一个物体在相机前方 0.5m 处
        T_cam_obj = np.eye(4)
        T_cam_obj[:3, 3] = [0, 0, 0.5] # Optical Z轴前方
        
        T_base_obj = get_target_pose_in_base_frame(T_base_optical, T_cam_obj)
        plot_coordinate_frame(ax, T_base_obj, "Object", scale=0.1)
        
        # 绘制视线 (Base -> Cam -> Object)
        cam_pos = T_base_optical[:3, 3]
        obj_pos = T_base_obj[:3, 3]
        ax.plot([0, cam_pos[0], obj_pos[0]], 
                [0, cam_pos[1], obj_pos[1]], 
                [0, cam_pos[2], obj_pos[2]], 'k--', alpha=0.5)

        # 设置轴范围和标签
        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Left)')
        ax.set_zlabel('Z (Up)')
        
        # 调整视角以匹配机器人常规视角
        ax.view_init(elev=10, azim=-10)
        
        # 简单的等比例缩放 trick
        all_points = np.vstack([np.zeros(3), cam_pos, obj_pos])
        max_range = (all_points.max() - all_points.min()) / 2.0
        mid_x = (all_points[:,0].max() + all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max() + all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max() + all_points[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
        plt.title("G1 Coordinate Frames Visualization")
        plt.show()
        print("\nVisualization window opened.")
        
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")
    except Exception as e:
        print(f"\nVisualization error: {e}")
