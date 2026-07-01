import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# 配置中文字体支持
import matplotlib.font_manager as fm

# 清理matplotlib字体缓存并重建
try:
    fm._load_fontmanager(try_read_cache=False)
except:
    pass

# 使用系统中可用的CJK字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'AR PL UMing CN', 'AR PL UKai CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

print(f"当前使用字体: {plt.rcParams['font.sans-serif'][0]}")

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

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
    return poses

def transform_poses_to_base(poses_camera, waist_yaw=0, waist_pitch=0, waist_roll=0):
    """Transform pose sequence from camera frame to base frame"""
    T_base_camera = get_camera_pose_in_base_frame(waist_yaw, waist_pitch, waist_roll)
    poses_base = []
    for T_cam_obj in poses_camera:
        T_base_obj = get_target_pose_in_base_frame(T_base_camera, T_cam_obj)
        poses_base.append(T_base_obj)
    return poses_base

def plot_trajectory_analysis(poses_base, image_folder, save_path=None, use_chinese=True):
    """
    可视化物体轨迹和坐标轴方向
    
    Args:
        poses_base: list of (4,4) transformation matrices
        image_folder: path to image folder
        save_path: path to save the figure
        use_chinese: whether to use Chinese labels (set to False if font issues)
    """
    # 提取位置和旋转
    positions = np.array([T[:3, 3] for T in poses_base])
    
    # 提取物体坐标系的三个轴（在基座系中的表示）
    x_axes = np.array([T[:3, 0] for T in poses_base])  # 物体X轴
    y_axes = np.array([T[:3, 1] for T in poses_base])  # 物体Y轴
    z_axes = np.array([T[:3, 2] for T in poses_base])  # 物体Z轴
    
    # 标签字典
    if use_chinese:
        labels = {
            'trajectory': '轨迹',
            'start': '起点',
            'end': '终点',
            'forward': '前',
            'left': '左',
            'up': '上',
            'title_3d': '物体3D轨迹和坐标系\n红=X轴, 绿=Y轴, 蓝=Z轴',
            'title_xy': 'XY平面视图\n蓝色箭头=物体Z轴方向',
            'title_xz': 'XZ平面视图（侧视）',
            'title_pos': '位置随时间变化',
            'title_z': '物体Z轴方向变化\n（用于抓取）',
            'frame': '帧数',
            'position': '位置 (m)',
            'direction': '方向分量',
            'x_pos': 'X位置',
            'y_pos': 'Y位置',
            'z_pos': 'Z位置',
            'z_x': 'Z轴X分量',
            'z_y': 'Z轴Y分量',
            'z_z': 'Z轴Z分量',
            'image_title': '第0帧图像\n共{}帧'
        }
    else:
        labels = {
            'trajectory': 'Trajectory',
            'start': 'Start',
            'end': 'End',
            'forward': 'Forward',
            'left': 'Left',
            'up': 'Up',
            'title_3d': 'Object 3D Trajectory & Frames\nRed=X, Green=Y, Blue=Z',
            'title_xy': 'XY Plane View\nBlue arrows=Object Z-axis',
            'title_xz': 'XZ Plane View (Side)',
            'title_pos': 'Position vs Time',
            'title_z': 'Object Z-axis Direction\n(for grasping)',
            'frame': 'Frame',
            'position': 'Position (m)',
            'direction': 'Direction Component',
            'x_pos': 'X Position',
            'y_pos': 'Y Position',
            'z_pos': 'Z Position',
            'z_x': 'Z-axis X component',
            'z_y': 'Z-axis Y component',
            'z_z': 'Z-axis Z component',
            'image_title': 'Frame 0\nTotal {} frames'
        }
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # ========== 1. 3D轨迹图 ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label=labels['trajectory'])
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label=labels['start'])
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='x', label=labels['end'])
    
    # 绘制每隔10帧的物体坐标系
    step = max(1, len(poses_base) // 10)
    for i in range(0, len(poses_base), step):
        pos = positions[i]
        # X轴（红色）
        ax1.quiver(pos[0], pos[1], pos[2], 
                  x_axes[i, 0], x_axes[i, 1], x_axes[i, 2],
                  color='r', length=0.05, arrow_length_ratio=0.3, alpha=0.6)
        # Y轴（绿色）
        ax1.quiver(pos[0], pos[1], pos[2],
                  y_axes[i, 0], y_axes[i, 1], y_axes[i, 2],
                  color='g', length=0.05, arrow_length_ratio=0.3, alpha=0.6)
        # Z轴（蓝色）
        ax1.quiver(pos[0], pos[1], pos[2],
                  z_axes[i, 0], z_axes[i, 1], z_axes[i, 2],
                  color='b', length=0.05, arrow_length_ratio=0.3, alpha=0.6)
    
    ax1.set_xlabel(f'X ({labels["forward"]})')
    ax1.set_ylabel(f'Y ({labels["left"]})')
    ax1.set_zlabel(f'Z ({labels["up"]})')
    ax1.set_title(labels['title_3d'])
    ax1.legend()
    ax1.grid(True)
    
    # ========== 2. XY平面投影 ==========
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, marker='o', label=labels['start'])
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, marker='x', label=labels['end'])
    
    # 绘制物体Z轴在XY平面的投影（抓取方向）
    for i in range(0, len(poses_base), step):
        pos = positions[i]
        z_axis = z_axes[i]
        ax2.arrow(pos[0], pos[1], z_axis[0]*0.05, z_axis[1]*0.05,
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.6)
    
    ax2.set_xlabel(f'X ({labels["forward"]})')
    ax2.set_ylabel(f'Y ({labels["left"]})')
    ax2.set_title(labels['title_xy'])
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    
    # ========== 3. XZ平面投影 ==========
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
    ax3.scatter(positions[0, 0], positions[0, 2], c='g', s=100, marker='o', label=labels['start'])
    ax3.scatter(positions[-1, 0], positions[-1, 2], c='r', s=100, marker='x', label=labels['end'])
    ax3.set_xlabel(f'X ({labels["forward"]})')
    ax3.set_ylabel(f'Z ({labels["up"]})')
    ax3.set_title(labels['title_xz'])
    ax3.grid(True)
    ax3.axis('equal')
    ax3.legend()
    
    # ========== 4. 位置变化曲线 ==========
    ax4 = fig.add_subplot(2, 3, 4)
    frames = np.arange(len(positions))
    ax4.plot(frames, positions[:, 0], 'r-', label=labels['x_pos'])
    ax4.plot(frames, positions[:, 1], 'g-', label=labels['y_pos'])
    ax4.plot(frames, positions[:, 2], 'b-', label=labels['z_pos'])
    ax4.set_xlabel(labels['frame'])
    ax4.set_ylabel(labels['position'])
    ax4.set_title(labels['title_pos'])
    ax4.legend()
    ax4.grid(True)
    
    # ========== 5. 物体Z轴方向分析 ==========
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(frames, z_axes[:, 0], 'r-', label=labels['z_x'])
    ax5.plot(frames, z_axes[:, 1], 'g-', label=labels['z_y'])
    ax5.plot(frames, z_axes[:, 2], 'b-', label=labels['z_z'])
    ax5.set_xlabel(labels['frame'])
    ax5.set_ylabel(labels['direction'])
    ax5.set_title(labels['title_z'])
    ax5.legend()
    ax5.grid(True)
    
    # ========== 6. 显示关键帧图像 ==========
    ax6 = fig.add_subplot(2, 3, 6)
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if len(image_files) > 0:
        # 显示第一帧图像
        img = cv2.imread(image_files[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax6.imshow(img)
        ax6.set_title(labels['image_title'].format(len(image_files)))
        ax6.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存分析图到: {save_path}")
    
    plt.close()  # 关闭图形而不是显示
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("轨迹统计信息:")
    print("=" * 60)
    print(f"总帧数: {len(positions)}")
    print(f"\n位置范围:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")
    print(f"\n起点位置: {positions[0]}")
    print(f"终点位置: {positions[-1]}")
    print(f"总移动距离: {np.linalg.norm(positions[-1] - positions[0]):.3f} m")
    
    print(f"\n物体Z轴方向（用于抓取）:")
    print(f"  起点: {z_axes[0]}")
    print(f"  终点: {z_axes[-1]}")
    
    # 分析物体主要朝向
    z_mean = z_axes.mean(axis=0)
    z_mean = z_mean / np.linalg.norm(z_mean)
    print(f"\n平均Z轴方向: {z_mean}")
    
    if abs(z_mean[2]) > 0.7:
        print("  -> 物体主要朝向：竖直方向（向上/向下）")
    elif abs(z_mean[0]) > 0.7:
        print("  -> 物体主要朝向：前后方向")
    elif abs(z_mean[1]) > 0.7:
        print("  -> 物体主要朝向：左右方向")
    else:
        print("  -> 物体方向变化较大")

def create_image_montage(image_folder, sample_every=5, save_path=None):
    """创建图像蒙太奇，显示关键帧"""
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if len(image_files) == 0:
        print("未找到图像文件")
        return
    
    # 采样图像
    sampled_files = image_files[::sample_every]
    n_images = len(sampled_files)
    
    # 计算网格布局
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, img_file in enumerate(sampled_files):
        row = idx // n_cols
        col = idx % n_cols
        
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(img)
        frame_num = int(os.path.basename(img_file).split('.')[0])
        axes[row, col].set_title(f'Frame {frame_num}')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"已保存图像蒙太奇到: {save_path}")
    
    plt.close()  # 关闭图形而不是显示

def main():
    # 加载位姿序列
    pose_folder = os.path.join(current_dir, "ob_in_cam")
    image_folder = os.path.join(current_dir, "track_vis")
    
    print("=" * 60)
    print("加载物体位姿序列...")
    print("=" * 60)
    poses_camera = load_pose_sequence(pose_folder)
    print(f"加载了 {len(poses_camera)} 个位姿")
    
    # 转换到基座坐标系
    print("\n转换到基座坐标系...")
    poses_base = transform_poses_to_base(poses_camera)
    
    # 分析轨迹
    print("\n生成轨迹分析图...")
    # 如果字体有问题，设置 use_chinese=False 使用英文标签
    plot_trajectory_analysis(poses_base, image_folder, 
                            save_path=os.path.join(current_dir, "trajectory_analysis.png"),
                            use_chinese=True)  # 改为False使用英文
    
    # 创建图像蒙太奇
    print("\n生成图像蒙太奇...")
    create_image_montage(image_folder, sample_every=5,
                        save_path=os.path.join(current_dir, "image_montage.png"))

if __name__ == "__main__":
    main()
