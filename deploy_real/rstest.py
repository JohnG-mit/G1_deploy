import pyrealsense2 as rs
import cv2
import numpy as np
import torch
import time

# --- G1机器人的IP地址，请替换成实际的IP ---
G1_ROBOT_IP = "192.168.123.164"  # 示例IP，请务必修改为G1的真实IP地址
G1_ROBOT_PORT = 8554

if not isinstance(torch.__version__, str):
    torch.__version__ = str(torch.__version__)

# 1. 创建一个上下文（Context）
ctx = rs.context()

# 2. 添加网络设备
print(f"正在尝试连接到网络摄像头: {G1_ROBOT_IP}:{G1_ROBOT_PORT} ...")
net_device = ctx.add_net_device(f"{G1_ROBOT_IP}:{G1_ROBOT_PORT}")

# 3. 等待网络设备被发现
print("等待设备被发现...")
# 这是一个简单的等待机制，你可能需要根据网络情况调整等待时间
# 或者实现一个更鲁棒的循环来查询设备
time.sleep(5) 

# 4. 像往常一样配置和启动pipeline
pipeline = rs.pipeline(ctx)  # 使用带有网络设备的上下文
cfg = rs.config()

# 从网络设备获取序列号并启用它
# 注意：对于网络设备，最好明确指定序列号来配置
# 如果只有一个网络摄像头，通常可以省略这一步，但这是更稳妥的做法
# devices = ctx.query_devices()
# net_serial = ""
# for dev in devices:
#     if dev.get_info(rs.camera_info.name) == "Intel RealSense D435I": # 根据你的相机型号修改
#         net_serial = dev.get_info(rs.camera_info.serial_number)
#         break
# if net_serial:
#     print(f"找到网络摄像头，序列号: {net_serial}")
#     cfg.enable_device(net_serial)
# else:
#     print("错误：未找到网络摄像头。请检查IP地址和服务器状态。")
#     exit()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)

print("启动pipeline...")
pipeline.start(cfg)
try:
    while True:
        frames     = pipeline.wait_for_frames()
        aligned    = align.process(frames)
        color_frame= aligned.get_color_frame()
        depth_frame= aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 应用颜色映射到深度图像（用于可视化）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET)
        
        # 显示图像
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_colormap)
        
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
