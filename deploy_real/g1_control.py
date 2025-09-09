import time
import sys
import numpy as np
import yaml
import os
from multiprocessing import Process
from pynput import keyboard

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

file_dir = os.path.dirname(os.path.abspath(__file__))

class G1JointIndex:
    LeftLegHipPitch = 0
    LeftLegHipRoll = 1
    LeftLegHipYaw = 2
    LeftLegKnee = 3
    LeftLegAnklePitch = 4
    LeftLegAnkleRoll = 5
    RightLegHipPitch = 6
    RightLegHipRoll = 7
    RightLegHipYaw = 8
    RightLegKnee = 9
    RightLegAnklePitch = 10
    RightLegAnkleRoll = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29

def init_cmd_hg(cmd: LowCmd_, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

def create_damping_cmd(cmd: LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 8
        cmd.motor_cmd[i].tau = 0

def create_zero_cmd(cmd: LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

class Config: pass

def load_cfg(path=os.path.join(file_dir, "configs/config_high_level.yaml")) -> Config:
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)
    # cfg.kps_record = cfg.kps_play * 1
    # cfg.kds_record = cfg.kds_play * 1
    return cfg

class ControlHandler:
    def __init__(self):
        config = load_cfg(path = os.path.join(file_dir, "configs/config_high_level.yaml"))
        # config = load_cfg(path = os.path.join(file_dir, "configs/FixedPose.yaml"))

        self.control_dt = config.control_dt
        self.num_joints = config.num_joints
        self.action_joints = config.action_joints

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.low_state = None
        self.mode_machine_ = 0
        self.mode_pr_ = 0
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        self.wait_for_low_state()

        init_cmd_hg(self.low_cmd, mode_machine=self.mode_machine_, mode_pr=self.mode_pr_)  # activate G1 motor

        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.tgt_q = np.zeros(self.num_joints)

        self.initial_q = None
        self.kps_ctrl = config.kps
        self.kds_ctrl = config.kds
        self.default_angles = np.array(config.default_angles, dtype=np.float32)

    def LowStateHgHandler(self, msg: LowState_):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine

    def wait_for_low_state(self):
        while not self.low_state or self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def send_cmd(self, cmd: LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def start(self):
        print("Entering damping mode for a short period...")
        # for _ in range(50):  # 持续发送50次阻尼指令 (大约1秒)
        #     self.enter_damping_mode()
        #     time.sleep(self.control_dt)
        
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1

        self.cur_step = 0
        if self.initial_q is None:
            self.initial_q = self.qj.copy()
            for i in range(self.num_joints):
                self.initial_q[i] = self.low_state.motor_state[i].q
        self.tgt_q = self.initial_q.copy()
        
        self.thread = RecurrentThread(interval=self.control_dt, target=self.Loop, name="ControlLoop")
        self.thread.Start()

    def Loop(self):
        try:
            # if self.cur_step == 0:
            #     print("Control loop started.")
            # for i in range(self.num_joints):  # Update joint states
            #     self.qj[i] = self.low_state.motor_state[i].q

            kps = self.kps_ctrl.copy()
            kds = self.kds_ctrl.copy()

            tgt_q = self.tgt_q.copy()
            for i in range(self.num_joints):
                self.low_cmd.motor_cmd[i].q = tgt_q[i]
                self.low_cmd.motor_cmd[i].tau = 0
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].kp = kps[i]
            
            self.send_cmd(self.low_cmd)

        except ValueError as e:
            print(f"ValueError occurred: {e}")
            pass

    def enter_damping_mode(self):
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)

    def enter_zero_mode(self):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
    
    def reset_initial(self):
        self.cur_step = 0
        for i in range(self.num_joints):
            self.initial_q[i] = self.low_state.motor_state[i].q
        self.tgt_q = self.initial_q.copy()

    def move_to_default(self, duration=4.0):
        if self.cur_step < 10000:
            self.cur_step += 1

        total_steps = int(duration / self.control_dt)
        r = min(self.cur_step / total_steps, 1.0)
        if self.initial_q is not None:
            self.tgt_q = (1 - r) * self.initial_q + r * self.default_angles
    
    def read_record_single(self, file_path):
        with open(file_path, 'rb') as f:
            data = np.load(f)
            self.record_q = data['q']
        print(f"{file_path} loaded successfully.")

    def play_record_single(self, duration=4.0):
        tgt_q = self.qj.copy()
        for i in self.action_joints:
            tgt_q[i] = self.record_q[i]

        if self.cur_step < 10000:
            self.cur_step += 1
        
        total_steps = int(duration / self.control_dt)
        r = min(self.cur_step / total_steps, 1.0)
        if self.initial_q is not None:
            self.tgt_q = (1 - r) * self.initial_q + r * tgt_q

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp2s0")
    record_path = os.path.join(file_dir, "records", "single_3.npz")

    handler = ControlHandler()
    handler.start()

    try:
        duration = 4.0
        total_steps = int(duration / handler.control_dt)
        print(f"Move to default position in {duration} seconds...")
        for _ in range(total_steps):
            handler.move_to_default(duration=duration)
            time.sleep(handler.control_dt)
        handler.reset_initial()
        # while True:
        #     time.sleep(10)
        with keyboard.Events() as events:
            print("Press 'p' to play the recorded motion, 'Esc' to exit.")
            for event in events:
                if event.key == keyboard.Key.esc:
                    break
                elif event.key == keyboard.KeyCode.from_char('p'):
                    handler.reset_initial()
                    handler.read_record_single(file_path=record_path)
                    for _ in range(total_steps):
                        handler.play_record_single(duration)
                        time.sleep(handler.control_dt)
                else:
                    pass
    except KeyboardInterrupt:
        print("'Ctrl+C' detected, exiting...")
    finally:
        print("Entering default pose...")
        handler.reset_initial()
        for _ in range(total_steps):
            handler.move_to_default(duration)
            time.sleep(handler.control_dt)
    print("Exit")
