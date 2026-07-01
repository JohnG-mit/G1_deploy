import numpy as np
import pinocchio as pin

def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero vector")
    return v / n

def make_ee_pose_from_object(
    T_base_obj: np.ndarray,
    *,
    v_long_obj: np.ndarray,
    approach: str = "side",
    standoff: float = 0.10,
    offset_base: np.ndarray | None = None,
    world_up_base: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ee_z_points_to_object: bool = True,
) -> pin.SE3:
    """从物体位姿构造末端（L_ee/R_ee）目标位姿。

    约定：机器人 base 坐标系为 X前、Y左、Z上。

    目标：
    - 末端 x 轴：指向物体长边（由 v_long_obj 指定，定义在物体坐标系中）
    - 末端 z 轴：为“接近方向”(approach)
      - "side": 水平面内从机器人指向物体（把物体位置投影到 XY 平面得到方向）
      - "top":  从上往下（-Z）
      - "custom": 使用 offset_base 以外再传入你自己的 z 方向（可按需扩展）

    注意：
    - 只用“长边方向”无法唯一确定旋转（还缺少绕 x 的自由度），所以必须指定接近方向。
    - v_long_obj 的符号（+/-）不重要；但会影响最后的 y/z 朝向。

    Args:
        T_base_obj: (4,4) 物体在 base 下的位姿（Base <- Obj）
        v_long_obj: (3,) 物体长边方向在物体坐标系中的单位向量（例如你的凳子腿 OBJ 长边≈+Y，则 [0,1,0]）
        approach: "side" 或 "top"。
        standoff: 预抓取距离（沿接近方向反方向退开 standoff 米）
        offset_base: (3,) 额外平移偏置，默认 0。
        world_up_base: (3,) base 的“上方向”，默认 [0,0,1]。
        ee_z_points_to_object: True 表示末端 +Z 指向物体（接近方向）；False 表示 +Z 背离物体。

    Returns:
        pin.SE3: 末端目标位姿（世界系 o/base 下的 L_ee/R_ee）
    """
    if offset_base is None:
        offset_base = np.zeros(3)

    R_bo = T_base_obj[:3, :3]
    t_bo = T_base_obj[:3, 3]

    # 1) ee-x：对齐物体长边（把物体坐标系的长边方向旋到 base）
    x = _normalize(R_bo @ _normalize(np.asarray(v_long_obj, dtype=float).reshape(3)))

    # 2) ee-z：接近方向
    if approach == "top":
        z = -_normalize(world_up_base)
    elif approach == "side":
        # 从机器人（base 原点）指向物体中心的水平向量：把 z 分量置 0
        v = np.array([t_bo[0], t_bo[1], 0.0], dtype=float)
        if np.linalg.norm(v) < 1e-9:
            # 物体正好在 base 正上/正下时，随便选一个水平向量兜底
            v = np.array([1.0, 0.0, 0.0], dtype=float)
        z = _normalize(v)
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'side' or 'top'.")

    if not ee_z_points_to_object:
        z = -z

    # 防止 z 与 x 近似平行导致叉积退化（比如长轴水平且你又选了同方向接近）
    if abs(float(np.dot(x, z))) > 0.95:
        # 换一个与 x 不平行的兜底方向：优先用 world_up
        z = _normalize(world_up_base)
        if abs(float(np.dot(x, z))) > 0.95:
            z = np.array([0.0, 1.0, 0.0], dtype=float)

    # 3) ee-y：用叉乘补齐右手系
    #    这里用 y = z × x，保证 (x, y, z) 构成右手系
    y = _normalize(np.cross(z, x))
    z = np.cross(x, y)  # 再正交化一次，确保数值稳定

    # 4) 平移：从物体中心退开 standoff（得到预抓取位姿），再加额外偏置
    #    如果 z 指向物体，那么退开方向是 -z
    t_be = t_bo - z * float(standoff) + np.asarray(offset_base, dtype=float).reshape(3)

    R_be = np.stack([x, y, z], axis=1)  # 列向量分别是 ee 的 x/y/z 轴在 base 下的表达
    return pin.SE3(R_be, t_be)

if __name__ == "__main__":
    # 这里仅演示“怎么调用函数”，不直接依赖你的机器人控制对象。
    # 你需要在自己的控制脚本里：
    # 1) 得到 T_cam_obj（FoundationPose 输出 ob_in_cam）
    # 2) 得到 T_base_cam（相机 optical 在 base 下的外参）
    # 3) 计算 T_base_obj = T_base_cam @ T_cam_obj
    # 4) 构造 poseL/poseR 喂给 ctrl.move_to(...)

    # 例：你的凳子腿 OBJ 长边≈OBJ 的 +Y（见 mesh_axis_probe 输出），所以 v_long_obj = [0,1,0]
    demo_T_base_obj = np.eye(4)
    demo_T_base_obj[:3, 3] = [0.5, 0.0, 0.2]  # 物体在机器人前方 0.5m

    pose_pregrasp = make_ee_pose_from_object(
        demo_T_base_obj,
        v_long_obj=np.array([0.0, 1.0, 0.0]),
        approach="side",
        standoff=0.10,
        offset_base=np.zeros(3),
        world_up_base=np.array([0.0, 0.0, 1.0]),
        ee_z_points_to_object=True,
    )

    print("pregrasp SE3:\n", pose_pregrasp.homogeneous)