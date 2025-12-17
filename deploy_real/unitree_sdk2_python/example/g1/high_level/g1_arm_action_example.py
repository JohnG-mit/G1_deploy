import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import action_map
from dataclasses import dataclass
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
@dataclass
class TestOption:
    name: str
    id: int

option_list = [
    TestOption(name="release arm", id=0),     
    TestOption(name="shake hand", id=1),    
    TestOption(name="high five", id=2), 
    TestOption(name="hug", id=3), 
    TestOption(name="high wave", id=4),
    TestOption(name="clap", id=5), 
    TestOption(name="face wave", id=6),
    TestOption(name="left kiss", id=7),
    TestOption(name="heart", id=8),
    TestOption(name="right heart", id=9),
    TestOption(name="hands up", id=10),
    TestOption(name="x-ray", id=11),
    TestOption(name="right hand up", id=12),
    TestOption(name="reject", id=13),
    TestOption(name="right kiss", id=14), 
    TestOption(name="two-hand kiss", id=15),  
    
]

class UserInterface:
    def __init__(self):
        self.test_option_ = None

    def convert_to_int(self, input_str):
        try:
            return int(input_str)
        except ValueError:
            return None

    def terminal_handle(self):
        input_str = input("Enter id or name: \n")

        if input_str == "list":
            self.test_option_.name = None
            self.test_option_.id = None
            for option in option_list:
                print(f"{option.name}, id: {option.id}")
            return

        for option in option_list:
            if input_str == option.name or self.convert_to_int(input_str) == option.id:
                self.test_option_.name = option.name
                self.test_option_.id = option.id
                print(f"Test: {self.test_option_.name}, test_id: {self.test_option_.id}")
                return

        print("No matching test option found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    test_option = TestOption(name=None, id=None) 
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    msc = MotionSwitcherClient()
    msc.Init()
    print("[DDS] Motion Switcher Client Initialized.")
    while not queryMotionStatus(msc):
        print("Waiting for motion service to be active...")
    msc.SelectMode("ai")
    time.sleep(5)

    armAction_client = G1ArmActionClient()  
    armAction_client.SetTimeout(3.0)
    armAction_client.Init()

    # actionList = armAction_client.GetActionList()
    # print("actionList\n",actionList)

    print("Input \"list\" to list all test option ...")
    while True:
        user_interface.terminal_handle()

        print(f"Updated Test Option: Name = {test_option.name}, ID = {test_option.id}")

        if test_option.id == 0:
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 1:
            armAction_client.ExecuteAction(action_map.get("shake hand"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 2:
            armAction_client.ExecuteAction(action_map.get("high five"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 3:
            armAction_client.ExecuteAction(action_map.get("hug"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 4:
            armAction_client.ExecuteAction(action_map.get("high wave"))
        elif test_option.id == 5:
            armAction_client.ExecuteAction(action_map.get("clap"))
        elif test_option.id == 6:
            armAction_client.ExecuteAction(action_map.get("face wave"))
        elif test_option.id == 7:
            armAction_client.ExecuteAction(action_map.get("left kiss"))
        elif test_option.id == 8:
            armAction_client.ExecuteAction(action_map.get("heart"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 9:
            armAction_client.ExecuteAction(action_map.get("right heart"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 10:
            armAction_client.ExecuteAction(action_map.get("hands up"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 11:
            armAction_client.ExecuteAction(action_map.get("x-ray"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 12:
            armAction_client.ExecuteAction(action_map.get("right hand up"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 13:
            armAction_client.ExecuteAction(action_map.get("reject"))
            time.sleep(2)
            armAction_client.ExecuteAction(action_map.get("release arm"))
        elif test_option.id == 14:
            armAction_client.ExecuteAction(action_map.get("right kiss"))
        elif test_option.id == 15:
            armAction_client.ExecuteAction(action_map.get("two-hand kiss"))

        time.sleep(1)