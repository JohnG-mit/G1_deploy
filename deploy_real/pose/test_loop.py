import sys
import os
import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from g1_arm_IK import G1_29_ArmIK
from utils.coordinate_trans import get_camera_pose_in_base_frame, get_target_pose_in_base_frame

def load_pose_from_txt(path):
    arr = np.loadtxt(path).reshape(4, 4)
    return arr

def mat_to_se3(mat):
    return pin.SE3(mat[:3, :3], mat[:3, 3])

def interpolate_pose(T_start, T_end, alpha):
    # Linear interpolation of translation
    p_start = T_start.translation
    p_end = T_end.translation
    p_interp = (1 - alpha) * p_start + alpha * p_end
    
    # Slerp for rotation
    R_start = T_start.rotation
    R_end = T_end.rotation
    q_start = pin.Quaternion(R_start)
    q_end = pin.Quaternion(R_end)
    q_interp = q_start.slerp(alpha, q_end)
    
    return pin.SE3(q_interp, p_interp)

def main():
    # 1. Initialize IK
    print("Initializing IK...")
    ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)
    
    # 2. Load Target Pose
    txt_path = os.path.join(current_dir, "test.txt")
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
        return

    print(f"Loading target pose from {txt_path}")
    T_camera_object_np = load_pose_from_txt(txt_path)
    
    # 3. Transform to Base Frame
    # Assuming default waist angles (0,0,0)
    print("Transforming pose to Base Frame...")
    T_base_camera = get_camera_pose_in_base_frame(0, 0, 0)
    T_base_target_np = get_target_pose_in_base_frame(T_base_camera, T_camera_object_np)
    
    T_target_R = mat_to_se3(T_base_target_np) # Assuming target is for Right Hand
    
    # 4. Define Home Pose
    # "Forearm 90 deg lifted, palms relative"
    # Position: In front of chest.
    # Left Hand: x=0.3, y=0.25, z=0.1
    # Right Hand: x=0.3, y=-0.25, z=0.1
    # Rotation:
    # Left Hand: Palm facing Right (-Y). 
    # Right Hand: Palm facing Left (+Y).
    
    # Construct rotations
    # Identity: X forward, Y left, Z up.
    # Right Hand (Palm Left +Y): Identity is close (Palm is Y).
    # Left Hand (Palm Right -Y): Rotate 180 around X? Y becomes -Y. Z becomes -Z.
    
    # Let's try these orientations.
    # Right Hand: Identity
    R_home_R = np.eye(3)
    # Left Hand: Rotate 180 around X
    R_home_L = np.eye(3)
    
    T_home_L = pin.SE3(R_home_L, np.array([0.3, 0.25, 0.1]))
    T_home_R = pin.SE3(R_home_R, np.array([0.3, -0.25, 0.1]))
    
    # 5. Loop
    q_current = ik.init_data.copy()
    
    # Move to Home first (quickly or directly)
    print("Moving to Initial Pose...")
    steps_init = 50
    # We don't know where we start, so let's just interpolate from current q's FK?
    # Or just jump to Home? IK might jump.
    # Let's interpolate from "current" (which is zero/init) to Home.
    
    # Get current FK
    L_start, R_start = ik.forward_kinematics(q_current)
    # Note: forward_kinematics returns pin.SE3 objects
    
    for i in range(steps_init):
        alpha = i / steps_init
        T_curr_L = interpolate_pose(L_start, T_home_L, alpha)
        T_curr_R = interpolate_pose(R_start, T_home_R, alpha)
        
        q_sol, _ = ik.solve_ik(
            left_wrist=T_curr_L.homogeneous,
            right_wrist=T_curr_R.homogeneous,
            current_lr_arm_motor_q=q_current
        )
        q_current = q_sol
        time.sleep(0.01)
        
    print("Reached Initial Pose. Starting Loop. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Move to Target (Right Hand only, Left stays at Home)
            steps = 100
            print("Moving to Target...")
            for i in range(steps):
                alpha = i / steps
                # Interpolate Right Hand
                T_curr_R = interpolate_pose(T_home_R, T_target_R, alpha)
                # Keep Left Hand at Home
                T_curr_L = T_home_L
                
                q_sol, _ = ik.solve_ik(
                    left_wrist=T_curr_L.homogeneous,
                    right_wrist=T_curr_R.homogeneous,
                    current_lr_arm_motor_q=q_current
                )
                q_current = q_sol
                time.sleep(0.02)
            
            time.sleep(1.0)
            
            # Move back to Home
            print("Moving to Home...")
            for i in range(steps):
                alpha = i / steps
                # Interpolate Right Hand back
                T_curr_R = interpolate_pose(T_target_R, T_home_R, alpha)
                T_curr_L = T_home_L
                
                q_sol, _ = ik.solve_ik(
                    left_wrist=T_curr_L.homogeneous,
                    right_wrist=T_curr_R.homogeneous,
                    current_lr_arm_motor_q=q_current
                )
                q_current = q_sol
                time.sleep(0.02)
                
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()
