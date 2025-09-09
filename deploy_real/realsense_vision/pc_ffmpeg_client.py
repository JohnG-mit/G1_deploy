import cv2
import numpy as np
import os
from pathlib import Path

# 授权OpenCV的FFmpeg后端使用rtp和udp协议
# 这必须在第一次调用cv2.VideoCapture之前设置
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,udp,rtp"

script_path = Path(__file__).resolve()
script_dir = script_path.parent
sdp_dir = os.path.join(script_dir, "sdp")

def create_sdp_file(filename, port, payload_type, bitrate, unitree_ip="192.168.123.164"):
    """Creates an SDP file to describe an RTP stream for FFmpeg."""
    # The IP address 127.0.0.1 is a placeholder; FFmpeg/OpenCV will listen on all interfaces
    # for the specified port.
    sdp_content = f"""v=0
o=- 0 0 IN IP4 127.0.0.1
s=No Name
c=IN IP4 {unitree_ip}
t=0 0
a=tool:libavformat 58.29.100
m=video {port} RTP/AVP {payload_type}
b=AS:{bitrate}
a=rtpmap:{payload_type} H264/90000
a=fmtp:{payload_type} packetization-mode=1
"""
    os.makedirs(sdp_dir, exist_ok=True)
    if os.path.exists(os.path.join(sdp_dir, filename)):
        print(f"SDP file {filename} already exists.")
    else:
        with open(os.path.join(sdp_dir, filename), 'w') as f:
            f.write(sdp_content.strip())
            print(f"Created SDP file: {filename}")
    return os.path.abspath(os.path.join(sdp_dir, filename))

def main():
    # To use OpenCV's FFmpeg backend for receiving RTP streams, we provide it with
    # an SDP file that describes the stream's parameters (port, codec, etc.).
    
    # Create SDP files for the streams.
    rgb_sdp_path = None
    depth_sdp_path = None
    try:
        rgb_sdp_path = create_sdp_file("rgb.sdp", port=5600, payload_type=96, bitrate=4000)
        depth_sdp_path = create_sdp_file("depth.sdp", port=5602, payload_type=96,bitrate=2000)

        # Pass the SDP file paths to VideoCapture.
        # This tells OpenCV to use its FFmpeg backend to listen for the RTP stream.
        # We explicitly pass cv2.CAP_FFMPEG to ensure the correct backend is used.
        cap_rgb = cv2.VideoCapture(rgb_sdp_path, cv2.CAP_FFMPEG)
        cap_depth = cv2.VideoCapture(depth_sdp_path, cv2.CAP_FFMPEG)

        if not cap_rgb.isOpened():
            print("Error: Cannot open RGB VideoCapture using FFmpeg. Check SDP file and network stream.")
            return
        if not cap_depth.isOpened():
            print("Error: Cannot open Depth VideoCapture using FFmpeg.")
            return

        print("Video streams are configured using FFmpeg. Press 'q' to quit.")

        while True:
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_depth, frame_depth = cap_depth.read()

            if ret_rgb:
                cv2.imshow('RGB Stream from G1', frame_rgb)
            
            if ret_depth:
                cv2.imshow('Depth Stream from G1', frame_depth)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("Exiting...")
        # Safely release captures
        if 'cap_rgb' in locals() and cap_rgb.isOpened():
            cap_rgb.release()
        if 'cap_depth' in locals() and cap_depth.isOpened():
            cap_depth.release()
        cv2.destroyAllWindows()
        
        # Clean up the generated SDP files
        # if rgb_sdp_path and os.path.exists(rgb_sdp_path):
        #     os.remove(rgb_sdp_path)
        #     print(f"Removed {rgb_sdp_path}")
        # if depth_sdp_path and os.path.exists(depth_sdp_path):
        #     os.remove(depth_sdp_path)
        #     print(f"Removed {depth_sdp_path}")
        # print("Cleanup complete.")

if __name__ == '__main__':
    main()
