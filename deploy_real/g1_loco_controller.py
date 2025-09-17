import time, sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.comm.motion_switcher import MotionSwitcherClient

# Assuming the remote_controller is in a shared/common location
# Adjust the import path if necessary
from common.remote_controller import RemoteController, KeyMap

def queryServiceName(form: str, name: str) -> str:
    if form == "0":
        if name == "normal":
            return "sport_mode"
        if name == "ai":
            return "ai_sport"
        if name == "advanced":
            return "advanced_sport"
    else:
        if name == "ai-w":
            return "wheeled_sport(go2W)"
        if name == "normal-w":
            return "wheeled_sport(b2W)"
    return ""

class G1LocoController:
    def __init__(self):
        # Initialize DDS
        ChannelFactoryInitialize(0, "enp2s0")
        
        self.loco_client = LocoClient()
        self.loco_client.Init()
        print("[DDS] Loco Client Initialized.")

        self.msc = MotionSwitcherClient()
        self.msc.Init()
        print("[DDS] Motion Switcher Client Initialized.")

        while not self.queryMotionStatus():
            print("Waiting for motion service to be active...")
            self.msc.SelectMode("ai")
            time.sleep(5)

        self.remote = RemoteController()
        # We need to get state to update remote, but LocoClient doesn't expose state subscriber directly.
        # For now, we assume another process might be publishing state, or we handle it differently.
        # This is a key difference from the arm controller.
        # For simplicity, we'll poll the remote directly in a loop without DDS state callback.

        self.thread = None
        self.running = False
        
        # Robot state flags
        self.is_standing = False
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.speed_factor = 2.0 # To adjust speed

        # Rotation timer variables
        self.yaw_angle = 0.0
        self.initial_yaw = None
        self.total_rotation = 0.0
        self.timing_rotation = False
        self.rotation_start_time = None

    def start(self):
        self.running = True
        self.thread = RecurrentThread(interval=0.05, target=self.remote_poll)
        self.thread.Start()
        print("[Controller] Loco control thread started.")
        print("[Info] Press 'start' to stand up/walk, 'select' to exit.")

    def stop(self):
        print("[Controller] Stopping loco controller...")
        self.running = False
        if self.thread and self.thread.IsRunning():
            self.thread.Stop()
            self.thread.Join()
        
        # Send a final stop command
        self.loco_client.StopMove()
        self.loco_client.Damp()
        print("[Controller] Robot damped and stopped.")

    def remote_poll(self):
        """
        Polls the remote controller and executes commands.
        This runs in a separate thread.
        """
        if not self.running:
            return

        # Note: Without a LowState subscription, we can't get remote data this way.
        # This is a placeholder for how you would integrate it if you had the state feed.
        # For a real scenario, you'd need to subscribe to LowState like in the arm controller.
        # Let's assume for this script we can't and will handle input in the main loop.
        # The logic is moved to a public method `handle_input` to be called from main.
        pass

    def queryMotionStatus(self):
        code, data = self.msc.CheckMode()
        if code == 0:
            print("CheckMode succeeded.")
            print(data)
        else:
            print(f"CheckMode failed. Error code: {code}")

        if not data["name"]:
            print("The motion control-related service is deactivated.")
            motionStatus = 0
        else:
            serviceName = queryServiceName(data["form"], data["name"])
            print(f"Service: {serviceName} is activate")
            motionStatus = 1
        return motionStatus

    def handle_input(self, r):
        """
        Handles remote controller button presses.
        Call this method in your main loop.
        """
        # Toggle stand/walk mode
        # if r[KeyMap.start] == 1:
        #     if not self.is_standing:
        #         print("[Action] Standing up...")
        #         self.loco_client.Squat2StandUp()
        #         self.is_standing = True
        #     else:
        #         print("[Action] Switching to walking mode...")
        #         self.loco_client.Start() # FSM 200 for walking
        #     time.sleep(0.2) # Debounce

        # Safety stop
        if r[KeyMap.select] == 1:
            self.stop()
            raise SystemExit("[Exit] Select button pressed. Exiting.")

        # Damp mode
        if r[KeyMap.L1] == 1:
            print("[Action] Damping...")
            self.loco_client.Damp()
            self.is_standing = False
            time.sleep(0.2)

        # Zero torque mode
        if r[KeyMap.L2] == 1:
            print("[Action] Zero Torque...")
            self.loco_client.ZeroTorque()
            self.is_standing = False
            time.sleep(0.2)

        # Movement control
        # self.vx = self.remote.ly * 0.5 * self.speed_factor # Forward/Backward
        # self.vy = -self.remote.lx * 0.5 * self.speed_factor # Left/Right
        # self.vyaw = -self.remote.rx * 0.5 * self.speed_factor # Turn
        self.vx = -2 * 0.5 * self.speed_factor # Forward/Backward
        self.vy = 0 * 0.5 * self.speed_factor # Left/Right
        self.vyaw = 0 * 0.5 * self.speed_factor # Turn

        # Rotation timing logic
        # Check if rx is maxed out for turning
        if abs(self.remote.rx) > 0.95:
            if not self.timing_rotation:
                # Start timing
                self.timing_rotation = True
                self.rotation_start_time = time.time()
                self.initial_yaw = self.yaw_angle
                self.total_rotation = 0.0
                print(f"[Timer] Started timing rotation. Initial Yaw: {np.rad2deg(self.initial_yaw):.2f} deg.")
        else:
            if self.timing_rotation:
                # Stop timing if rx is no longer maxed out
                self.timing_rotation = False
                duration = time.time() - self.rotation_start_time
                print(f"[Timer] Stopped timing. Duration: {duration:.2f}s, Total Rotation: {self.total_rotation:.2f} deg.")
                self.initial_yaw = None

        if self.timing_rotation:
            # This block will be executed in the next call after initial_yaw is set
            if self.initial_yaw is not None:
                # Calculate rotation delta and handle wrap-around
                current_yaw = self.yaw_angle
                delta_yaw = current_yaw - self.last_yaw_for_rotation
                
                if delta_yaw > np.pi:
                    delta_yaw -= 2 * np.pi
                elif delta_yaw < -np.pi:
                    delta_yaw += 2 * np.pi
                
                self.total_rotation += np.rad2deg(delta_yaw)

                if abs(self.total_rotation) >= 360:
                    duration = time.time() - self.rotation_start_time
                    print("="*50)
                    print(f"[Timer] Full circle detected!")
                    print(f"    Duration: {duration:.2f} seconds")
                    print(f"    Total Rotation: {self.total_rotation:.2f} degrees")
                    print("="*50)
                    # Reset for next measurement
                    self.timing_rotation = False
                    self.initial_yaw = None

        # Update last yaw for next calculation
        self.last_yaw_for_rotation = self.yaw_angle
        # print(self.vx)

        if abs(self.vx) > 0.05 or abs(self.vy) > 0.05 or abs(self.vyaw) > 0.05:
                # pass
            # if self.is_standing:
                # print("[Action] Walking...")
                self.loco_client.Move(self.vx, self.vy, self.vyaw, True)
        else:
            pass
            # Send stop command if there's no movement input
        #     self.loco_client.StopMove()
        # self.loco_client.Move(self.vx, self.vy, self.vyaw, True)

        # Stand height
        if r[KeyMap.Y] == 1:
            print("[Action] High Stand")
            self.loco_client.HighStand()
            time.sleep(0.2)
        if r[KeyMap.A] == 1:
            print("[Action] Low Stand")
            self.loco_client.LowStand()
            time.sleep(0.2)
            
        # Sit down
        if r[KeyMap.B] == 1:
            print("[Action] Sit")
            self.loco_client.Sit()
            self.is_standing = False
            time.sleep(0.2)

def main():
    print("Ensure no obstacle. Press ENTER to start.")
    input()

    ctrl = G1LocoController()
    
    # The remote controller state needs to be updated from a DDS subscription.
    # The LocoClient doesn't expose this, so we'll create a minimal subscriber here
    # just to get the remote data, similar to the arm controller.
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    def low_state_callback(msg: LowState_):
        ctrl.remote.set(msg.wireless_remote)
        # Update yaw angle from IMU data (quaternion)
        q = msg.imu_state.quaternion
        # Simple conversion from quaternion to yaw.
        # This is a common formula: yaw = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2^2 + q3^2))
        ctrl.yaw_angle = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
        
        # Initialize last_yaw_for_rotation on the first callback
        if not hasattr(ctrl, 'last_yaw_for_rotation'):
            ctrl.last_yaw_for_rotation = ctrl.yaw_angle

    # This is a simplified way to get remote data.
    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(low_state_callback, 10)
    
    print("[DDS] LowState Subscriber for remote control ready.")
    
    # ctrl.start() # We don't need a separate thread if we poll in main

    try:
        print("Controller is running. Use remote to control the robot.")
        print("Press Ctrl+C to exit.")
        ctrl.loco_client.Damp()
        time.sleep(1)
        ctrl.loco_client.StandUp()
        cur_fsmid = None
        while not cur_fsmid == (0, 4):
            cur_fsmid = ctrl.loco_client.GetFsmId()
            time.sleep(0.1)
        while not cur_fsmid == (0, 500):
            cur_fsmid = ctrl.loco_client.GetFsmId()
            ctrl.loco_client.Start()
            time.sleep(0.1)
        print(ctrl.loco_client.GetFsmId())
        print(ctrl.loco_client.GetFsmMode())
        print("Robot should be standing and ready to walk.")
        
        while True:
            # Get the latest remote state
            remote_buttons = ctrl.remote.button
            # Handle the input
            ctrl.handle_input(remote_buttons)
            # Loop at ~20Hz
            time.sleep(0.05)
            
    except (KeyboardInterrupt, SystemExit) as e:
        print(f"Caught exit signal: {e}")
    finally:
        print("Cleaning up...")
        ctrl.stop()
        low_state_sub.Close() # Clean up subscriber
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()
