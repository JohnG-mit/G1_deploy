#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation

# 分析轨迹文件格式
files = ['records/traj_1.npz', 'records/traj_1766477199.npz']

for fp in files:
    try:
        data = np.load(fp, allow_pickle=True)
        print(f'\n{"="*60}')
        print(f'文件: {fp}')
        print(f'{"="*60}')
        print(f'Keys: {list(data.keys())}')
        
        for key in data.keys():
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f'\n  {key}:')
                print(f'    shape: {val.shape}')
                print(f'    dtype: {val.dtype}')
                if len(val.shape) == 2:
                    print(f'    -> 共{val.shape[0]}帧, 每帧{val.shape[1]}维')
                elif len(val.shape) == 1 and len(val) <= 10:
                    print(f'    -> 值: {val}')
            else:
                print(f'\n  {key}:')
                print(f'    类型: {type(val).__name__}')
                print(f'    值: {val}')
                
        # 如果有traj字段，分析结构
        if 'traj' in data:
            traj = data['traj']
            print(f'\n  {"="*50}')
            print(f'  轨迹数据结构分析:')
            print(f'  {"="*50}')
            print(f'    总帧数: {traj.shape[0]}')
            print(f'    每帧维度: {traj.shape[1]}')
            print(f'    预期格式: q(12) + pL(3) + qL(4) + pR(3) + qR(4) = 26维')
            
            if traj.shape[1] == 26:
                print(f'\n  第一帧数据示例:')
                first = traj[0]
                print(f'    q(12):')
                print(f'      {first[:12]}')
                print(f'    pL(3) - 左手位置:')
                print(f'      {first[12:15]}')
                print(f'    qL(4) - 左手四元数(x,y,z,w):')
                print(f'      {first[15:19]}')
                print(f'    pR(3) - 右手位置:')
                print(f'      {first[19:22]}')
                print(f'    qR(4) - 右手四元数(x,y,z,w):')
                print(f'      {first[22:26]}')
                
                # 验证四元数
                qL = first[15:19]
                qR = first[22:26]
                print(f'\n  四元数验证:')
                print(f'    左手四元数模: {np.linalg.norm(qL):.6f} (应该≈1.0)')
                print(f'    右手四元数模: {np.linalg.norm(qR):.6f} (应该≈1.0)')
                
    except Exception as e:
        print(f'\nError loading {fp}: {e}')
        import traceback
        traceback.print_exc()

print(f'\n{"="*60}')
print('分析完成')
print(f'{"="*60}')
