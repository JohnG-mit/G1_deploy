# deploy_real/joint_visualizer.py

import sys
import os
import yaml
import numpy as np
import time
from threading import Lock, Thread
import termios
import tty
import select

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


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
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29

    # 创建一个反向映射，从ID到名称
    _id_to_name_map = {v: k for k, v in locals().items() if isinstance(v, int)}

    @classmethod
    def get_name(cls, joint_id):
        return cls._id_to_name_map.get(joint_id, f"UnknownJoint_{joint_id}")
    

# -----------------------------------------------------------------------------
# 主可视化类
# -----------------------------------------------------------------------------
class JointVisualizer:
    def __init__(self):
        self.low_state = None
        self.first_state = False
        self.data_lock = Lock()
        self.running = True
        self.num_joints = 29
        self.record_index = 1

        # 创建 records 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_dir = os.path.join(script_dir, 'records')
        os.makedirs(self.records_dir, exist_ok=True)
        print(f"数据将保存在: {self.records_dir}")

        # 为每个关节ID获取名称
        self.joint_names = [G1JointIndex.get_name(j) for j in range(self.num_joints)]


        # 初始化关节位置
        self.joint_positions = np.zeros(self.num_joints)

    def Init(self):
        """初始化DDS通信"""
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self.state_cb, 10)
        print("等待机器人状态数据...")
        while not self.first_state:
            time.sleep(0.5)
        print("已接收到数据，开始在终端显示。按 Ctrl+C 退出。")

    def state_cb(self, msg: LowState_):
        """DDS回调函数，用于更新关节位置"""
        with self.data_lock:
            for i, joint_id in enumerate(range(self.num_joints)):
                self.joint_positions[i] = msg.motor_state[joint_id].q
        if not self.first_state:
            self.first_state = True

    def save_positions(self):
        """保存当前关节位置到 .npz 文件"""
        with self.data_lock:
            # 复制0-28号关节的位置数据
            positions_to_save = self.joint_positions[:29].copy()
        
        filename = f"single_{self.record_index}.npz"
        filepath = os.path.join(self.records_dir, filename)
        
        try:
            np.savez(filepath, q=positions_to_save)
            # 在显示循环的下一次刷新前，这个消息可能会被清除，但在终端历史记录中可见
            print(f"\n[+] 关节位置已保存到 {filepath}")
            self.record_index += 1
        except Exception as e:
            print(f"\n[!] 保存文件失败: {e}")

    def display_loop(self):
        """在终端中循环打印关节数据"""
        while self.running:
            with self.data_lock:
                joint_positions = self.joint_positions.copy()

            # 清除屏幕
            os.system('cls' if os.name == 'nt' else 'clear')
            
            for i, joint_id in enumerate(range(self.num_joints)):
                name = self.joint_names[i]
                pos = joint_positions[i]
                print(f"{joint_id:<3}: {name:<20} {pos:8.4f}")

            time.sleep(0.5)

    def run(self):
        """启动可视化并监听键盘输入"""
        display_thread = Thread(target=self.display_loop, daemon=True)
        display_thread.start()

        # 保存终端的原始设置
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            # 设置终端为原始模式
            tty.setcbreak(sys.stdin.fileno())
            print("按 'r' 记录当前姿态, 按 Ctrl+C 退出。")

            while self.running:
                # 使用 select 检查是否有键盘输入
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char == 'r' or char == 'R':
                        self.save_positions()
                    elif char == '\x03': # Ctrl+C
                        raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在退出...")
        finally:
            self.running = False
            # 恢复终端的原始设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            display_thread.join(timeout=1)

if __name__ == "__main__":
    print("启动关节位置实时监视器...")
    
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)

    visualizer = JointVisualizer()
    visualizer.Init()
    visualizer.run()

    print("程序已关闭。")