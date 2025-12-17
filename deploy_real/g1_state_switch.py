import sys
import numpy as np
import time
from pynput import keyboard
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher import MotionSwitcherClient

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
    
def queryMotionStatus(msc):
    code, data = msc.CheckMode()
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
    
# msc = MotionSwitcherClient()
# msc.Init()
# print("[DDS] Motion Switcher Client Initialized.")
# # 下面是一个进入调试模式的示例循环
# while queryMotionStatus(msc):
#     print("Try to deactivate the motion control-related service.")
#     ret, _ = msc.ReleaseMode()
#     if ret == 0:
#         print("ReleaseMode succeeded.")
#     else:
#         print(f"ReleaseMode failed. Error code: {ret}")
#     time.sleep(5)

# # 下面是一个进入运控模式的示例循环
# while not queryMotionStatus(msc):
#     print("Waiting for motion service to be active...")
#     msc.SelectMode("ai")
#     time.sleep(5)

def switch_to_debug_mode(msc):
    """切换到调试模式(释放运控服务)"""
    print("\n[切换到调试模式] 尝试释放运控服务...")
    while queryMotionStatus(msc):
        ret, _ = msc.ReleaseMode()
        if ret == 0:
            print("✓ 成功切换到调试模式")
        else:
            print(f"✗ 切换失败,错误代码: {ret}")
        time.sleep(1)

def switch_to_motion_mode(msc):
    """切换到运控模式(激活AI模式)"""
    print("\n[切换到运控模式] 尝试激活AI运动模式...")
    while not queryMotionStatus(msc):
        msc.SelectMode("ai")
        time.sleep(1)

if __name__ == "__main__":
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)
    
    # 初始化Motion Switcher Client
    msc = MotionSwitcherClient()
    msc.Init()
    print("[DDS] Motion Switcher Client Initialized.")
    print("\n=== 模式切换控制 ===")
    print("按键说明:")
    print("  z - 切换到调试模式(释放运控服务)")
    print("  c - 切换到运控模式(激活AI模式)")
    print("  q - 退出程序")
    print("==================\n")
    
    # 查询当前状态
    queryMotionStatus(msc)
    
    print("\n请输入命令 (z/c/q):")
    # 使用标准输入循环接收命令
    while True:
        try:
            # 读取用户输入
            cmd = input("> ").strip().lower()
            
            if cmd == 'z':
                switch_to_debug_mode(msc)
            elif cmd == 'c':
                switch_to_motion_mode(msc)
            elif cmd == 'q':
                print("\n退出程序...")
                break
            else:
                print("无效命令,请输入 z/c/q")
                
        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C,退出程序...")
            break
        except EOFError:
            print("\n\n输入结束,退出程序...")
            break
