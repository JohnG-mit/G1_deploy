import pyrealsense2 as rs
import numpy as np
import cv2

ctx = rs.context()
print(f"检测到 {len(ctx.devices)} 个设备")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
try:
    while True:
        # 获取帧数据
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        

        # 将帧数据转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 显示图像
        cv2.imshow('RGB Image', color_image)
        cv2.imshow('Depth Image', depth_image)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 停止数据流
    pipeline.stop()
    cv2.destroyAllWindows()