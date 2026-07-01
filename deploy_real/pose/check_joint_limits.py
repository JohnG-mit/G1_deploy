#!/usr/bin/env python3
"""
检查轨迹文件中的关节角度是否超过URDF限位
"""
import numpy as np
import xml.etree.ElementTree as ET
import sys

def parse_urdf_joint_limits(urdf_path):
    """
    从URDF文件解析关节限位
    
    Returns:
        dict: {joint_name: {'lower': float, 'upper': float}}
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    joint_limits = {}
    
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        # 只处理revolute关节
        if joint_type == 'revolute':
            limit = joint.find('limit')
            if limit is not None:
                lower = float(limit.get('lower'))
                upper = float(limit.get('upper'))
                joint_limits[joint_name] = {'lower': lower, 'upper': upper}
    
    return joint_limits

def get_arm_joint_names():
    """获取12个手臂关节的名称（按顺序）"""
    return [
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_joint',
        'left_wrist_roll_joint',
        'left_wrist_pitch_joint',
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint',
        'right_elbow_joint',
        'right_wrist_roll_joint',
        'right_wrist_pitch_joint'
    ]

def check_trajectory_limits(traj_file, urdf_path):
    """
    检查轨迹文件中的关节角度是否超限
    
    Args:
        traj_file: 轨迹文件路径
        urdf_path: URDF文件路径
    """
    # 加载轨迹数据
    data = np.load(traj_file)
    traj = data['traj']
    
    print(f"\n{'='*70}")
    print(f"分析轨迹文件: {traj_file}")
    print(f"{'='*70}")
    print(f"总帧数: {traj.shape[0]}")
    print(f"每帧维度: {traj.shape[1]}")
    
    # 提取关节角度 (前12维)
    q_array = traj[:, :12]
    
    # 解析URDF限位
    joint_limits = parse_urdf_joint_limits(urdf_path)
    arm_joint_names = get_arm_joint_names()
    
    # 检查每个关节
    violations = {}
    
    print(f"\n{'='*70}")
    print(f"关节限位检查")
    print(f"{'='*70}")
    
    for idx, joint_name in enumerate(arm_joint_names):
        if joint_name not in joint_limits:
            print(f"警告: 关节 {joint_name} 在URDF中未找到限位")
            continue
        
        limits = joint_limits[joint_name]
        q_values = q_array[:, idx]
        
        # 统计
        q_min = np.min(q_values)
        q_max = np.max(q_values)
        q_mean = np.mean(q_values)
        q_std = np.std(q_values)
        
        # 检查超限
        lower_violations = np.sum(q_values < limits['lower'])
        upper_violations = np.sum(q_values > limits['upper'])
        total_violations = lower_violations + upper_violations
        
        # 超限幅度
        lower_exceed = min(0, q_min - limits['lower'])
        upper_exceed = max(0, q_max - limits['upper'])
        
        print(f"\n关节 {idx}: {joint_name}")
        print(f"  限位范围: [{limits['lower']:.4f}, {limits['upper']:.4f}] rad")
        print(f"  实际范围: [{q_min:.4f}, {q_max:.4f}] rad")
        print(f"  均值±标准差: {q_mean:.4f} ± {q_std:.4f}")
        
        if total_violations > 0:
            print(f"  ❌ 超限帧数: {total_violations}/{len(q_values)} ({100*total_violations/len(q_values):.2f}%)")
            if lower_violations > 0:
                print(f"     下限超限: {lower_violations} 帧, 最大超出 {abs(lower_exceed):.4f} rad")
            if upper_violations > 0:
                print(f"     上限超限: {upper_violations} 帧, 最大超出 {upper_exceed:.4f} rad")
            
            violations[joint_name] = {
                'index': idx,
                'lower_violations': lower_violations,
                'upper_violations': upper_violations,
                'lower_exceed': lower_exceed,
                'upper_exceed': upper_exceed,
                'limits': limits,
                'actual_range': (q_min, q_max)
            }
        else:
            print(f"  ✅ 所有帧均在限位内")
    
    # 总结
    print(f"\n{'='*70}")
    print(f"总结")
    print(f"{'='*70}")
    
    if violations:
        print(f"\n发现 {len(violations)} 个关节存在超限问题:")
        for joint_name, info in violations.items():
            total = info['lower_violations'] + info['upper_violations']
            print(f"  - {joint_name} (关节{info['index']}): {total} 帧超限")
        
        print(f"\n建议:")
        print(f"  1. 在IK求解时添加关节限位约束")
        print(f"  2. 调整轨迹规划参数（如偏移量）")
        print(f"  3. 使用梯度裁剪或软约束")
    else:
        print(f"\n✅ 所有关节角度均在限位范围内！")
    
    return violations

def main():
    # 路径配置
    traj_file = 'records/traj_1766643215.npz'
    urdf_path = 'deploy_real/assets/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf'
    
    try:
        violations = check_trajectory_limits(traj_file, urdf_path)
        
        if violations:
            print(f"\n{'='*70}")
            print(f"退出代码: 1 (发现超限)")
            print(f"{'='*70}")
            sys.exit(1)
        else:
            print(f"\n{'='*70}")
            print(f"退出代码: 0 (无超限)")
            print(f"{'='*70}")
            sys.exit(0)
            
    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 - {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()
