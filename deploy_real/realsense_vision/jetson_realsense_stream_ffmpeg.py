import argparse
import subprocess
import sys
import numpy as np
import pyrealsense2 as rs
import cv2

def start_ffmpeg_process(w, h, fps, client_ip):
    """Starts an FFmpeg process to stream video."""
    command_rgb = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '4000k',
        '-f', 'rtp',
        f'rtp://{client_ip}:5600'
    ]
    
    command_depth = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', # Depth is colorized to BGR before sending
        '-s', f'{w}x{h}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '2000k',
        '-f', 'rtp',
        f'rtp://{client_ip}:5602'
    ]

    # Start FFmpeg processes
    proc_rgb = subprocess.Popen(command_rgb, stdin=subprocess.PIPE)
    proc_depth = subprocess.Popen(command_depth, stdin=subprocess.PIPE)
    
    return proc_rgb, proc_depth

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--client-ip", required=True, help="IP address of the client machine.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    # Start FFmpeg subprocesses
    proc_rgb, proc_depth = start_ffmpeg_process(args.width, args.height, args.fps, args.client_ip)

    # Initialise RealSense
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipe = rs.pipeline()
    pipe.start(cfg)

    temp_filter = rs.temporal_filter()

    try:
        while True:
            frames = pipe.wait_for_frames()
            
            # Get RGB frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get and process depth frame
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_frame = temp_filter.process(depth_frame)
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Colorize depth for streaming
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Write to FFmpeg stdin
            if proc_rgb.stdin:
                proc_rgb.stdin.write(color_image.tobytes())
            if proc_depth.stdin:
                proc_depth.stdin.write(depth_colormap.tobytes())

    except KeyboardInterrupt:
        print("Interrupted, shutting down...")
    finally:
        proc_rgb.stdin.close()
        proc_rgb.wait()
        proc_depth.stdin.close()
        proc_depth.wait()
        pipe.stop()
        print("FFmpeg and RealSense pipeline stopped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)