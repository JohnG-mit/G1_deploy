import cv2
import numpy as np

def main():
    # GStreamer pipelines for receiving H.264 encoded streams via UDP
    # Note: These pipelines assume GStreamer is installed with necessary plugins (e.g., h264parse, avdec_h264 or another decoder)
    
    # Pipeline for the RGB stream on port 5600
    pipeline_rgb = (
        "udpsrc port=5600 "
        "! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 "
        "! rtph264depay "
        "! h264parse "
        "! avdec_h264 "
        "! videoconvert "
        "! appsink drop=true max-buffers=1 emit-signals=true"
        # "! autovideosink"
    )

    # Pipeline for the Depth stream on port 5602
    pipeline_depth = (
        "udpsrc port=5602 "
        "! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)97 "
        "! rtph264depay "
        "! h264parse "
        "! avdec_h264 "
        "! videoconvert "
        "! appsink drop=true max-buffers=1 emit-signals=true"
        # "! autovideosink"
    )

    # Create VideoCapture objects
    cap_rgb = cv2.VideoCapture(pipeline_rgb, cv2.CAP_GSTREAMER)
    cap_depth = cv2.VideoCapture(pipeline_depth, cv2.CAP_GSTREAMER)

    if not cap_rgb.isOpened():
        print("Error: Cannot open RGB VideoCapture. Check GStreamer installation and pipeline string.")
        return
    if not cap_depth.isOpened():
        print("Error: Cannot open Depth VideoCapture. Check GStreamer installation and pipeline string.")
        return

    print("Video streams are configured. Press 'q' to quit.")

    while True:
        # Read frames from the pipelines
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_depth, frame_depth = cap_depth.read()

        # Display the frames if successfully captured
        if ret_rgb:
            cv2.imshow('RGB Stream from G1', frame_rgb)
        
        if ret_depth:
            cv2.imshow('Depth Stream from G1', frame_depth)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    print("Exiting...")
    cap_rgb.release()
    cap_depth.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()