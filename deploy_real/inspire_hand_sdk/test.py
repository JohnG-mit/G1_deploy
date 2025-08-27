import time
import sys
import numpy as np
import os

# Add parent directory to sys.path to find unitree_sdk2py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.utils.thread import Thread

from inspire_sdkpy import inspire_sdk, inspire_hand_defaut,inspire_dds

handler=inspire_sdk.ModbusDataHandler(ip='192.168.123.211',LR='r',device_id=1)
states_structure = [
    ('pos_act', 1534, 6, 'short'),
    ('angle_act', 1546, 6, 'short'),
    ('force_act', 1582, 6, 'short'),
    ('current', 1594, 6, 'short'),
    ('err', 1606, 3, 'byte'),
    ('status', 1612, 3, 'byte'),
    ('temperature', 1618, 3, 'byte')
]
gesture_no_set = (1008, 1, 'byte')  # 8握拳；9二指捏；10完全张开；11完全张开；12完全张开；13握拳；14握拳；
user_def_angle = [
    ('14', 1066, 6, 'short'),
    ('15', 1078, 6, 'short'),
    ('16', 1090, 6, 'short'),
    ('17', 1102, 6, 'short'),
    ('18', 1114, 6, 'short'),
    ('19', 1126, 6, 'short'),
    ('20', 1138, 6, 'short'),
    ('21', 1150, 6, 'short'),
    ('22', 1162, 6, 'short'),
    ('23', 1174, 6, 'short'),
    ('24', 1186, 6, 'short'),
    ('25', 1198, 6, 'short'),
    ('26', 1210, 6, 'short'),
    ('27', 1222, 6, 'short'),
    ('28', 1234, 6, 'short'),
    ('29', 1246, 6, 'short'),
    ('30', 1258, 6, 'short'),
    ('31', 1270, 6, 'short'),
    ('32', 1282, 6, 'short'),
    ('33', 1294, 6, 'short'),
    ('34', 1306, 6, 'short'),
    ('35', 1318, 6, 'short')
]

# pubr = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
# pubr.Init()
# cmd = inspire_hand_defaut.get_inspire_hand_ctrl()

handler.client.write_register(gesture_no_set[0], 8, handler.device_id)
print(handler.read_and_parse_registers(1056, user_def_angle[0][2], user_def_angle[0][3]))
print(handler.read_and_parse_registers(user_def_angle[0][1], user_def_angle[0][2], user_def_angle[0][3]))
# print(handler.read_and_parse_registers(user_def_angle[-1][1], 120, user_def_angle[-1][3]))
# for i in range(0, 180, 6):
#     print(f"current addr: {1318+i}")
#     print(handler.read_and_parse_registers(1318+i, 6, 'short'))
print(handler.read_and_parse_registers(1498, 6, 'short'))