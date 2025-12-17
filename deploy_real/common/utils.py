import os
import yaml
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from enum import Enum, unique


class Config: pass

def load_cfg(path) -> Config:
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)
    # cfg.kps_record = cfg.kps_play * 1
    # cfg.kds_record = cfg.kds_play * 1
    return cfg

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

def setup_logging(log_file, level=logging.INFO, max_bytes=10 * 1024 * 1024, backup_count=5):
    """
    Sets up logging with both console and file handlers.
    
    :param log_file: Path to the log file.
    :param level: Logging level (default: logging.INFO).
    :param max_bytes: Maximum size of the log file in bytes before rotation (default: 10MB).
    :param backup_count: Number of backup files to keep (default: 5).
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def progress_bar(current, total, length=50):
    percent = current / total
    filled = int(length * percent)
    bar = "█" * filled + "-" * (length - filled)
    return f"\r|{bar}| {percent:.1%} [{current:.3f}s/{total:.3f}s]"

def scale_values(values, target_ranges):
    scaled = []
    for val, (new_min, new_max) in zip(values, target_ranges):
        scaled_val = (val + 1) * (new_max - new_min) / 2 + new_min
        scaled.append(scaled_val)
    return np.array(scaled)

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
