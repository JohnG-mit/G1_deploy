import sys
import numpy as np
import time
import termios
from pathlib import Path

from threading import Lock, Thread
from pynput import keyboard

from common.utils import *
from common.command_helper import init_cmd_hg

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from inspire_sdkpy.inspire_sdk import ModbusDataHandler
from inspire_sdkpy.inspire_hand_defaut import get_inspire_hand_touch, get_inspire_hand_state, get_inspire_hand_ctrl
from inspire_sdkpy.inspire_dds import inspire_hand_touch, inspire_hand_state, inspire_hand_ctrl


script_dir = Path(__file__).parent
(script_dir / 'logs').mkdir(exist_ok=True)
logger = setup_logging(script_dir / 'logs' / 'g1_highlevel_arm_record.log')


class G1HighLevelArmRecorder:
    def __init__(self):
        self.mode_machine_ = 6  # 29dof(LowCmd_.mode_machine == 2)
        self.mode_pr_ = 0
        self.data_lock = Lock()
        self.state_lock = Lock()

        cfg = load_cfg(script_dir / "configs/config_record.yaml")

        self.num_joints = cfg.num_joints
        self.control_dt = cfg.control_dt
        self.kps = cfg.kps
        self.kds = cfg.kds
        self.action_joints = cfg.action_joints
        self.fixed_joints = cfg.fixed_joints
        self.stiffness_factor = cfg.stiffness_factor
        self.default_angles = cfg.default_angles

        self.current_q = np.zeros_like(self.action_joints, dtype=np.float32)
        self.tgt_q = self.current_q.copy()

        self.low_state = None
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self.lowstate_cb, 10)
        self.wait_for_low_state()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()

        # Hand control states
        self.lh_angle = [0, 0, 0, 0, 0, 1000]  # Left hand finger angles
        self.rh_angle = [0, 0, 0, 0, 0, 1000]  # Right hand finger angles
        self.hand_control_lock = Lock()
        self.lh_cmd = get_inspire_hand_ctrl()
        self.rh_cmd = get_inspire_hand_ctrl()
        self.lh_cmd.mode = 0b0101
        self.rh_cmd.mode = 0b0101
        self.lh_cmd.force_set = [300, 300, 300, 300, 300, 300]
        self.rh_cmd.force_set = [300, 300, 300, 300, 300, 300]
        self.lh_pub = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_hand_ctrl)
        self.lh_pub.Init()
        self.rh_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_hand_ctrl)
        self.rh_pub.Init()

        self.hand_state = {}
        self.rh_handler = Thread(target=self.hand_state_worker, args=('192.168.123.211','r', self.hand_state, self.control_dt))
        self.lh_handler = Thread(target=self.hand_state_worker, args=('192.168.123.210','l', self.hand_state, self.control_dt))
        self.rh_handler.daemon = True
        self.lh_handler.daemon = True
        self.rh_handler.start()
        self.lh_handler.start()
        self.wait_for_hand_state()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        while not queryMotionStatus(self.msc):
            logger.info("Waiting for motion service to be active...")
            self.msc.SelectMode("ai")
            time.sleep(2)

        self.loco_client = LocoClient()
        self.loco_client.Init()

        self.records_dir = script_dir / 'records'
        self.records_dir.mkdir(exist_ok=True)
        single_files = list(self.records_dir.glob("single_*.npz"))
        traj_files = list(self.records_dir.glob("traj_*.npz"))
        self.single_index = max(
            [int(f.stem.split('_')[1]) for f in single_files],
            default=-1
        ) + 1
        self.traj_index = max(
            [int(f.stem.split('_')[1]) for f in traj_files],
            default=-1
        ) + 1
        self.recording = False
        self.frame_recording = False
        self.t_record_start = 0
        self.record_buffer_t = []
        self.record_buffer_q = []
        self.records_dir = script_dir / 'records'
        self.records_dir.mkdir(exist_ok=True)
        logger.info(f"Record data saved at: {self.records_dir}")
    
    def wait_for_low_state(self):
        while not self.low_state or self.low_state.tick == 0:
            time.sleep(self.control_dt)
        logger.info("Successfully connected to the robot.")

    def wait_for_hand_state(self):
        while not ("r" in self.hand_state and "l" in self.hand_state):
            time.sleep(self.control_dt)
        logger.info("Successfully connected to the hand.")

    def send_arm_cmd(self, cmd: LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        self.arm_pub.Write(cmd)
    
    def send_hand_cmd(self, cmd: inspire_hand_ctrl, LR: str):
        if LR == 'l':
            self.lh_pub.Write(cmd)
        elif LR == 'r':
            self.rh_pub.Write(cmd)
        else:
            logger.error(f"Invalid LR value: {LR}. Must be 'l' or 'r'.")
    
    def lowstate_cb(self, msg: LowState_):
        with self.state_lock:
            self.low_state = msg

    def hand_state_worker(self, ip, LR, hand_state_dict, control_dt):
        handler = ModbusDataHandler(ip=ip, LR=LR, initDDS=False)
        try:
            while True:
                hand_state_dict[LR] = handler.read()
                time.sleep(control_dt)
        except KeyboardInterrupt:
            logger.info(f"'Ctrl+C' detected in hand_state_worker ({LR}), exiting...")
        except Exception as e:
            logger.error(f"Exception in hand_state_worker ({LR}): {e}")

    def reset_initial(self):
        with self.state_lock and self.data_lock:
            for i, joint_idx in enumerate(self.action_joints):
                self.current_q[i] = self.low_state.motor_state[joint_idx].q
            self.tgt_q = self.current_q.copy()

    def start(self, duration=4.0):
        self.enter_fix_stand()

        init_cmd_hg(self.low_cmd, mode_machine=self.mode_machine_, mode_pr=self.mode_pr_)
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.reset_initial()

        self.control_loop = RecurrentThread(interval=self.control_dt, target=self.Loop, name="ControlLoop")
        self.control_loop.Start()

        logger.info(f"Move to default position in {duration} seconds...")
        self.move_to_default(duration=duration)

    def Loop(self):
        try:
            kps_arr = self.kps.copy()
            kds_arr = self.kds.copy()
            if self.recording or self.frame_recording:
                kps_arr = self.kps * self.stiffness_factor
                kds_arr = self.kds * self.stiffness_factor
            
            with self.data_lock:
                tgt_q = self.tgt_q.copy()
            for i, joint_idx in enumerate(self.action_joints):
                self.low_cmd.motor_cmd[joint_idx].q = tgt_q[i]
                self.low_cmd.motor_cmd[joint_idx].tau = 0
                self.low_cmd.motor_cmd[joint_idx].qd = 0
                self.low_cmd.motor_cmd[joint_idx].kd = kds_arr[i]
                self.low_cmd.motor_cmd[joint_idx].kp = kps_arr[i]
            for i, joint_idx in enumerate(self.fixed_joints):
                # self.low_cmd.motor_cmd[joint_idx].q = self.default_angles[i]
                self.low_cmd.motor_cmd[joint_idx].tau = 0
                self.low_cmd.motor_cmd[joint_idx].qd = 0
                self.low_cmd.motor_cmd[joint_idx].kd = self.kds[i]
                self.low_cmd.motor_cmd[joint_idx].kp = self.kps[i]

            # Send hand commands with current angles
            with self.hand_control_lock:
                self.lh_cmd.angle_set = self.lh_angle.copy()
                self.rh_cmd.angle_set = self.rh_angle.copy()

            self.send_arm_cmd(self.low_cmd)
            self.lh_pub.Write(self.lh_cmd)
            self.rh_pub.Write(self.rh_cmd)
            # self.send_hand_cmd(self.lh_cmd, 'l')
            # self.send_hand_cmd(self.rh_cmd, 'r')

            if self.recording:
                t = time.time() - self.t_record_start
                q = np.array([self.low_state.motor_state[m].q for m in self.action_joints])
                self.record_buffer_t.append(t)
                self.record_buffer_q.append(q)
            if self.frame_recording:
                self.reset_initial()

        except ValueError as e:
            print(f"ValueError occurred: {e}")
            pass

    def move_to_default(self, duration=4.0):
        default_q = np.zeros(len(self.action_joints))
        for i, joint_idx in enumerate(self.action_joints):
            default_q[i] = self.default_angles[joint_idx]
        self.reset_initial()
        steps = int(duration / self.control_dt)
        for step in range(steps):
            alpha = (step + 1) / steps
            with self.data_lock:
                self.tgt_q = (1 - alpha) * self.current_q + alpha * default_q
            time.sleep(self.control_dt)
    
    def enter_fix_stand(self):
        self.loco_client.Damp()
        time.sleep(1)
        self.loco_client.StandUp()
        cur_fsmid = None
        while not cur_fsmid == (0, 4):
            cur_fsmid = self.loco_client.GetFsmId()
            time.sleep(0.1)
        time.sleep(2)
        logger.info("Robot is standing.")

    def Run(self):
        self.reset_initial()
        logger.info("Ready for recording. Press 'r' to start/stop recording trajectory, 's' to start/save current pos, 'p' to play, 'q' to quit.")
        logger.info("Hand control: Press 'o' to open left hand, 'u' to close left hand, 'k' to open right hand, 'm' to close right hand.")
        logger.info("Hand control: Press 'b' to open both hands, 'n' to close both hands.")

        def on_press(key):
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            try:
                if key.char == 'r':
                    if not self.recording:
                        self.recording = True
                        self.t_record_start = time.time()
                        self.record_buffer_t = []
                        self.record_buffer_q = []
                        logger.info("[*] Started recording...")
                    else:
                        self.reset_initial()
                        self.recording = False
                        if len(self.record_buffer_t) > 1:
                            filename = f"traj_{self.traj_index}.npz"
                            filepath = self.records_dir / filename
                            np.savez_compressed(
                                filepath, 
                                t=np.array(self.record_buffer_t), 
                                q=np.array(self.record_buffer_q)
                            )
                            logger.info(f"[+] Recording saved at {filepath}")
                            self.traj_index += 1
                        else:
                            logger.warning("[!] Recording too short, not saved.")
                elif key.char == 'p':
                    logger.info("Entering playback mode. Press 'q' to exit playback.")
                    while True:
                        if not self.recording:
                            logger.info("[*] Available recordings:")
                            single_files = list(self.records_dir.glob("single_*.npz"))
                            traj_files = list(self.records_dir.glob("traj_*.npz"))
                            
                            file_map = {}
                            index = 0
                            
                            logger.info("Single frames:")
                            for file in single_files:
                                logger.info(f"[{index}] {file.name}")
                                file_map[index] = ("single", int(file.stem.split('_')[1]))
                                index += 1
                            
                            logger.info("Trajectories:")
                            for file in traj_files:
                                logger.info(f"[{index}] {file.name}")
                                file_map[index] = ("traj", int(file.stem.split('_')[1]))
                                index += 1
                            
                            termios.tcflush(sys.stdin, termios.TCIFLUSH)
                            time.sleep(0.5)
                            user_input = input("Enter the index of the file to play (or 'l' to leave): ")
                            if user_input.isdigit():
                                user_input = int(user_input)
                                if user_input in file_map:
                                    file_type, file_idx = file_map[user_input]
                                    if file_type == "single":
                                        logger.info(f"[*] Playing single frame with index {file_idx}...")
                                        try:
                                            self.play_frame(file_idx=file_idx)
                                        except Exception as e:
                                            logger.error(f"[!] Error playing frame: {e}")
                                    elif file_type == "traj":
                                        logger.info(f"[*] Playing trajectory with index {file_idx}...")
                                        try:
                                            self.play_trajectory(file_idx=file_idx)
                                        except Exception as e:
                                            logger.error(f"[!] Error playing trajectory: {e}")
                            elif user_input == 'l':
                                logger.info("Exiting playback mode.")
                                logger.info("Ready for recording. Press 'r' to start/stop recording trajectory, 's' to start/save current pos, 'p' to play, 'q' to quit.")
                                logger.info("Hand control: Press 'o' to open left hand, 'u' to close left hand, 'k' to open right hand, 'm' to close right hand.")
                                logger.info("Hand control: Press 'b' to open both hands, 'n' to close both hands.")
                                break
                            else:
                                logger.error("[!] Invalid index entered.")
                        else:
                            logger.warning("[!] Cannot play while recording.")
                elif key.char == 's':
                    if not self.frame_recording:
                        self.frame_recording = True
                        logger.info("[*] Started recording single frame...")
                    else:
                        self.save_frame()
                        self.frame_recording = False
                elif key.char == 'q':
                    logger.info("[*] Quitting...")
                    return False
                # Hand control keys
                elif key.char == 'o':  # Open left hand
                    with self.hand_control_lock:
                        self.lh_angle = [1000, 1000, 1000, 1000, 1000, 1000]
                    logger.info("[*] Left hand opened")
                elif key.char == 'u':  # Close left hand
                    with self.hand_control_lock:
                        self.lh_angle = [0, 0, 0, 0, 500, 1000]
                    logger.info("[*] Left hand closed")
                elif key.char == 'k':  # Open right hand
                    with self.hand_control_lock:
                        self.rh_angle = [1000, 1000, 1000, 1000, 1000, 1000]
                    logger.info("[*] Right hand opened")
                elif key.char == 'm':  # Close right hand
                    with self.hand_control_lock:
                        self.rh_angle = [0, 0, 0, 0, 500, 1000]
                    logger.info("[*] Right hand closed")
                elif key.char == 'b':  # Open both hands
                    with self.hand_control_lock:
                        self.lh_angle = [800, 800, 800, 800, 800, 1000]
                        self.rh_angle = [800, 800, 800, 800, 800, 1000]
                    logger.info("[*] Both hands opened")
                elif key.char == 'n':  # Close both hands
                    with self.hand_control_lock:
                        self.lh_angle = [0, 0, 0, 0, 500, 1000]
                        self.rh_angle = [0, 0, 0, 0, 500, 1000]
                    logger.info("[*] Both hands closed")
            except AttributeError:
                pass
            except KeyboardInterrupt:
                logger.info("'Ctrl+C' detected, exiting...")
                return False

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        try:
            listener.join()
        except KeyboardInterrupt:
            logger.info("'Ctrl+C' detected, exiting...")
        except Exception as e:
            logger.error(f"Exception in listener: {e}")
        finally:
            self.reset_initial()
            self.move_to_default(duration=4.0)
            self.loco_client.Damp()
            logger.info("Exited cleanly.")

    def save_frame(self):
        positions_to_save = self.current_q.copy()

        filename = f"single_{self.single_index}.npz"
        filepath = self.records_dir / filename
        
        try:
            np.savez(filepath, q=positions_to_save)
            logger.info(f"\n[+] joint position saved at {filepath}")
            self.single_index += 1
        except Exception as e:
            logger.error(f"\n[!] Failed to save file: {e}")

    def play_frame(self, duration=2.0, file_idx=0):
        self.reset_initial()
        filepath = self.records_dir / f"single_{file_idx}.npz"
        if filepath.exists():
            data = np.load(filepath)
            record_q = data['q']
        else:
            logger.error(f"\n[!] No file found for index {file_idx}")
            return

        steps = int(duration / self.control_dt)
        for step in range(steps):
            alpha = (step + 1) / steps
            with self.data_lock:
                self.tgt_q = (1 - alpha) * self.current_q + alpha * record_q
            time.sleep(self.control_dt)

    def play_trajectory(self, file_idx=0):
        self.reset_initial()
        filepath = self.records_dir / f"traj_{file_idx}.npz"
        if filepath.exists():
            data = np.load(filepath)
            record_t = data['t']
            record_q = data['q']
        else:
            logger.error(f"\n[!] No file found for index {file_idx}")
            return

        start_time = time.time()
        steps = int(4.0 / self.control_dt)
        for step in range(steps):
            alpha = (step + 1) / steps
            with self.data_lock:
                self.tgt_q = (1 - alpha) * self.current_q + alpha * record_q[0]
            time.sleep(self.control_dt)
        for i in range(1, len(record_t)):
            while time.time() - start_time < record_t[i]:
                time.sleep(0.001)
            with self.data_lock:
                self.tgt_q = record_q[i]
        logger.info("[*] Finished playing trajectory.")


if __name__ == "__main__":
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)
    duration = 4.0

    rc = G1HighLevelArmRecorder()
    rc.start(duration=duration)
    rc.Run()
