import os
import sys

from polars import col
from sympy import collect_const

# from torch import Tensor

current_dir = os.path.dirname(os.path.abspath(__file__))  # pin
parent_dir = os.path.dirname(current_dir)  # pose
project_root = os.path.dirname(parent_dir)  # deploy_real
# sys.path.insert(0, "/home/johng/repo/pyroboplan/src")
sys.path.insert(0, project_root)

from g1_arm_IK import G1_29_ArmIK
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg

import time
import numpy as np
import pinocchio as pin
import coal
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import copy
import shutil

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

def _rpy_to_rot(rpy):
    return R.from_euler('XYZ', rpy).as_matrix()

def _align_with_robot_base(pos):
    return np.array([pos[0], pos[1], pos[2]-0.793])

def interpolate_pose(T_start, T_end, alpha):
    # Linear interpolation of translation
    p_start = T_start.translation
    p_end = T_end.translation
    p_interp = (1 - alpha) * p_start + alpha * p_end
    
    # Slerp for rotation
    R_start = T_start.rotation
    R_end = T_end.rotation
    q_start = pin.Quaternion(R_start)
    q_end = pin.Quaternion(R_end)
    q_interp = q_start.slerp(alpha, q_end)
    
    return pin.SE3(q_interp, p_interp)

def set_robot_vis(ik):
    # Initialize the Meshcat visualizer for visualization
    vis = MeshcatVisualizer(ik.reduced_robot.model, ik.reduced_robot.collision_model, ik.reduced_robot.visual_model)
    vis.initViewer(open=True, loadModel=True)
    vis.displayVisuals(True)
    vis.displayCollisions(True)
    vis.displayFrames(True, frame_ids=[ik.L_hand_id, ik.R_hand_id], axis_length = 0.15, axis_width = 5)
    vis.display(pin.neutral(ik.reduced_robot.model))

    # Enable the display of end effector target frames with short axis lengths and greater width.
    frame_viz_names = ['L_ee_target', 'R_ee_target']
    FRAME_AXIS_POSITIONS = (
        np.array([[0, 0, 0], [1, 0, 0],
                    [0, 0, 0], [0, 1, 0],
                    [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
    )
    FRAME_AXIS_COLORS = (
        np.array([[1, 0, 0], [1, 0.6, 0],
                    [0, 1, 0], [0.6, 1, 0],
                    [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
    )
    axis_length = 0.1
    axis_width = 20
    for frame_viz_name in frame_viz_names:
        vis.viewer[frame_viz_name].set_object(
            mg.LineSegments(
                mg.PointsGeometry(
                    position=axis_length * FRAME_AXIS_POSITIONS,
                    color=FRAME_AXIS_COLORS,
                ),
                mg.LineBasicMaterial(
                    linewidth=axis_width,
                    vertexColors=True,
                ),
            )
        )
    return vis

def save_meshcat_frame(vis, save_dir, frame_idx):
    image = vis.viewer.get_image()
    if image is None:
        print(f"[Recorder] Warning: empty image at frame {frame_idx}")
        return False

    save_path = os.path.join(save_dir, f"frame_{frame_idx:05d}.png")
    if hasattr(image, "save"):
        image.save(save_path)
        return True

    print(f"[Recorder] Warning: unsupported image type {type(image)}")
    return False

def update_collision_display(vis, contacts, *, point_size=0.03, color=0xFF0000):
    """
    在 Meshcat 里更明显地显示碰撞接触点：
    - 使用 Points（点大小在浏览器里可见）
    - 线段 linewidth 在多数 WebGL 实现里不生效，因此只作为辅助
    """
    # 清空旧显示（避免残留）
    try:
        vis.viewer["collision_display"].delete()
    except Exception:
        pass

    if contacts is None or len(contacts) == 0:
        return

    pts = np.asarray(contacts, dtype=np.float32).reshape(-1, 3)  # (N,3)

    # 1) 大号点（强烈推荐，最明显）
    vis.viewer["collision_display/points"].set_object(
        mg.Points(
            mg.PointsGeometry(position=pts.T),
            mg.PointsMaterial(size=float(point_size), color=int(color)),
        )
    )

    # 2) 可选：线段（可能在浏览器里仍然很细）
    if pts.shape[0] >= 2:
        vis.viewer["collision_display/segments"].set_object(
            mg.LineSegments(
                mg.PointsGeometry(position=pts.T),
                mg.LineBasicMaterial(color=int(color)),
            )
        )

def collision_check(robot, q_sol):
    pin.computeCollisions(
            robot.model, robot.data, robot.collision_model, robot.collision_data, q_sol, False
        )
    contacts = []
    for k in range(len(robot.collision_model.collisionPairs)):
        cr = robot.collision_data.collisionResults[k]
        cp = robot.collision_model.collisionPairs[k]
        if cr.isCollision():
            print(
                "collision between:",
                robot.collision_model.geometryObjects[cp.first].name,
                " and ",
                robot.collision_model.geometryObjects[cp.second].name,
            )
            for contact in cr.getContacts():
                contacts.extend(
                    [contact.getNearestPoint1(), contact.getNearestPoint2()]
                )
    if len(contacts) == 0:
        pass
    return contacts

def joint_limit_violations(robot, q, tol=1e-9):
    q_arr = np.asarray(q, dtype=float)
    lower = robot.model.lowerPositionLimit
    upper = robot.model.upperPositionLimit
    below = np.where(q_arr < (lower - tol))[0]
    above = np.where(q_arr > (upper + tol))[0]
    return below, above

def print_joint_limit_error(robot, q, state_name):
    below, above = joint_limit_violations(robot, q)
    if below.size == 0 and above.size == 0:
        return False

    q_arr = np.asarray(q, dtype=float)
    print(f"[Planner] Error: {state_name} state violates joint limits!")
    for j in below:
        print(
            f"  joint[{j}] = {q_arr[j]:.6f} < lower = {robot.model.lowerPositionLimit[j]:.6f}"
        )
    for j in above:
        print(
            f"  joint[{j}] = {q_arr[j]:.6f} > upper = {robot.model.upperPositionLimit[j]:.6f}"
        )
    return True

def build_scene(robot, scene_cfg:dict):
    collision_model = robot.collision_model

    def _add_env_object(name: str, geometry, placement: pin.SE3, *, color=None, robot_geom_count: int):
        env_obj = pin.GeometryObject(name, 0, geometry, placement)
        if color is not None:
            env_obj.meshColor = np.asarray(color, dtype=float)
        env_id = collision_model.addGeometryObject(env_obj)

        # Add robot-env collision pairs explicitly (Pinocchio does not auto-generate them).
        for robot_gid in range(robot_geom_count):
            collision_model.addCollisionPair(pin.CollisionPair(robot_gid, env_id))
        return env_id

    robot_geom_count = len(collision_model.geometryObjects)

    table_size = scene_cfg["table"]["size"]
    table_center = scene_cfg["table"]["center"]
    table_geom = coal.Box(*table_size)
    table_pose = pin.SE3(np.eye(3), _align_with_robot_base(table_center))
    _add_env_object(
        name="table",
        geometry=table_geom,
        placement=table_pose,
        color=(0.8, 0.6, 0.4, 1.0),
        robot_geom_count=robot_geom_count,
    )

    stool_off = scene_cfg["stool"].get("mesh_offset", (0.0, 0.0, 0.0))
    stool_mesh = coal.MeshLoader().load(scene_cfg["stool"]["mesh_path"])
    stool_pose_ref = pin.SE3(
        _rpy_to_rot(scene_cfg["stool"]["rpy"]), 
        _align_with_robot_base(scene_cfg["stool"]["xyz"]),
    )
    stool_pose = stool_pose_ref * pin.SE3(np.eye(3), np.array(stool_off, dtype=float))
    _add_env_object(
        "stool_top",
        stool_mesh,
        stool_pose,
        color=[0.9, 0.9, 0.9, 1.0],
        robot_geom_count=robot_geom_count,
    )

    # --- Leg (mesh): static or attached depending on phase ---
    leg_off = scene_cfg["leg"].get("mesh_offset", (0.0, 0.0, 0.0))
    leg_mesh = coal.MeshLoader().load(scene_cfg["leg"]["mesh_path"])
    leg_pose_ref = pin.SE3(
        _rpy_to_rot(scene_cfg["leg"]["pickup_rpy"]),
        _align_with_robot_base(scene_cfg["leg"]["pickup_xyz"]),
    )
    leg_pose = leg_pose_ref * pin.SE3(np.eye(3), np.array(leg_off, dtype=float))
    _add_env_object(
        "stool_leg",
        leg_mesh,
        leg_pose,
        color=[0.7, 0.4, 0.3, 1.0],
        robot_geom_count=robot_geom_count,
    )

def plan(robot, q_start, q_goal, planner="RRTConnect", planning_time=5.0, smooth_path=True, num_waypoints=None):
    ''' Plan a collision-free path from q_start to q_goal using OMPL
    Parameters
    ----------
    robot : pin.RobotWrapper
        The Pinocchio robot model with collision model and data.
    q_start : np.ndarray
        The starting joint configuration.
    q_goal : np.ndarray
        The goal joint configuration.
    planner : str, optional
            The name of the motion planning algorithm to use. Supported planners: 'PRM', 'RRT', 'RRTConnect', 'RRTstar', 'EST', 'FMT', 'BITstar', 'ABITstar'. Defaults to 'RRTConnect'.
    planning_time : float, optional
        The maximum time allowed for planning in seconds. Defaults to 5.0.
    smooth_path : bool, optional
        Whether to apply path smoothing after planning. Defaults to True.
    num_waypoints : int, optional
        The number of waypoints to interpolate the path to. If None, uses the original path waypoints. Defaults to None.
    Returns
    -------
    waypoints : list of np.ndarray
        A list of joint configurations representing the planned path.
    '''
    ou.setLogLevel(ou.LOG_ERROR)
    space = ob.RealVectorStateSpace(robot.model.nq)
    bounds = ob.RealVectorBounds(robot.model.nq)
    for i in range(robot.model.nq):
        bounds.setLow(i, float(robot.model.lowerPositionLimit[i]))
        bounds.setHigh(i, float(robot.model.upperPositionLimit[i]))
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    def is_valid(state):
        q = np.array([state[i] for i in range(robot.model.nq)], dtype=float)
        below, above = joint_limit_violations(robot, q)
        if below.size > 0 or above.size > 0:
            return False
        return not pin.computeCollisions(
            robot.model, robot.data, robot.collision_model, robot.collision_data, q, True
        )

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))

    try:
        planner_cls = getattr(og, planner)
        if not issubclass(planner_cls, ob.Planner):
            raise ValueError
        ss.setPlanner(planner_cls(ss.getSpaceInformation()))
    except (AttributeError, ValueError) as e:
        raise(f"'{planner}' is not a valid planner. See OMPL documentation for details.", e)
    
    start = ob.State(space)
    goal = ob.State(space)
    for i in range(robot.model.nq):
        start[i] = q_start[i]
    for i in range(robot.model.nq):
        goal[i] = q_goal[i]

    if not is_valid(start):
        if print_joint_limit_error(robot, q_start, "Start"):
            return None
        print("[Planner] Error: Start state is in collision!")
        contacts = collision_check(robot, q_start)
        return None
    if not is_valid(goal):
        if print_joint_limit_error(robot, q_goal, "Goal"):
            return None
        print("[Planner] Error: Goal state is in collision!")
        contacts = collision_check(robot, q_goal)
        return None

    ss.setStartAndGoalStates(start, goal, 0.02)

    print(f"[Planner] Starting planning (timeout: {planning_time}s)...")
    solved = ss.solve(float(planning_time))
    waypoints = []
    if solved:
        print("[Planner] Path solution found successfully.")
        path = ss.getSolutionPath()
        if smooth_path:
            ps = og.PathSimplifier(ss.getSpaceInformation())
            # simplify the path
            try:
                ps.partialShortcutPath(path)
                ps.ropeShortcutPath(path)
            except:
                ps.shortcutPath(path)
            ps.smoothBSpline(path)

        if num_waypoints is not None:
            path.interpolate(num_waypoints)
        waypoints = [
            np.array([state[i] for i in range(robot.model.nq)], dtype=float)
            for state in path.getStates()
        ]
    else:
        print("[Planner] Path planning failed. Returning empty path.")
        waypoints = []

    return waypoints

if __name__ == "__main__":
    arm_ik = G1_29_ArmIK(ForceReload=True)
    g1 = arm_ik.reduced_robot

    table_size = (0.77, 1.5, 0.80)
    table_center = (0.60, 0.0, 0.40)

    # stool top
    stool_mesh_path = os.path.join(project_root, 'assets_obj/stool/stool.obj')
    stool_xyz = (0.37, 0.0, 0.82)
    stool_euler = (0.0, 0.0, 0.0)

    # stool leg
    stool_leg_mesh_path = os.path.join(project_root, 'assets_obj/stool-leg/stool-leg.obj')
    stool_leg_xyz = (0.27, -0.15, 0.93)  # right-front
    stool_leg_euler = (0.0, 0.0, 0.0)
    hole_offset = (0.15, -0.12, -0.02)    # right-front hole (relative to stool center)
    hole_pos = (stool_xyz[0] + hole_offset[0], stool_xyz[1] + hole_offset[1], stool_xyz[2] + hole_offset[2])

    scene_cfg = {
        "table": {
            "size": table_size,
            "center": table_center,
            "mesh_offset": (0.0, 0.0, 0.0),
        },
        "stool": {
            "mesh_path": stool_mesh_path,
            "xyz": stool_xyz,
            "rpy": stool_euler,
            "mesh_offset": (0.0, 0.0, 0.0),
        },
        "leg": {
            "mesh_path": stool_leg_mesh_path,
            "pickup_xyz": stool_leg_xyz,
            "pickup_rpy": stool_leg_euler,
        },
    }
    build_scene(g1, scene_cfg=scene_cfg)
    g1.collision_data = g1.collision_model.createData()
    vis = set_robot_vis(arm_ik)

    img_dir = os.path.join(current_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    print(f"[Recorder] Frame images will be saved to: {img_dir}")

    save_once_done = False
    saved_frame_count = 0

    # q_home = [-0.06300107389688492, 1.6363739967346191, 0.06309694796800613, 1.433660864830017, -0.06913699209690094, -0.01609145849943161, -0.023728765547275543, -1.5769442319869995, -0.06358829885721207, 1.4105912446975708, 0.04851214215159416, 0.04398690164089203]
    q_home = [0.053, 0.486, 0.173, 1.249, 0.167, -0.032, 0.054, -0.434, -0.173, 1.249, -0.167, 0.032]
    T_left_base, T_right_base = arm_ik.forward_kinematics(np.array(q_home))

    q_current = np.zeros_like(q_home)
    input("Press Enter to move to initial pose...")
    steps_init = 50
    L_start, R_start = arm_ik.forward_kinematics(q_current)
    for i in range(steps_init):
        
        alpha = i / steps_init
        T_curr_L = interpolate_pose(L_start, T_left_base, alpha)
        T_curr_R = interpolate_pose(R_start, T_right_base, alpha)
        
        q_sol, _ = arm_ik.solve_ik(
            left_wrist=T_curr_L.homogeneous,
            right_wrist=T_curr_R.homogeneous,
            current_lr_arm_motor_q=q_current
        )

        # contacts = collision_check(g1, q_sol)
        # update_collision_display(vis, contacts, point_size=0.24)
        
        q_current = q_sol
        vis.display(q_sol)
        time.sleep(0.01)
        
    T_right_target = pin.SE3(
        np.eye(3),
        # np.array([0.14007, -0.19996, 0.14972]),
        np.array([0.19007, -0.19996, 0.14972]),
    )
    q_start = copy.deepcopy(q_current)

    steps = 50
    q_inter = copy.deepcopy(q_current)
    for i in range(steps):
        alpha = i / steps
        # Interpolate Right Hand
        T_curr_R = interpolate_pose(T_right_base, T_right_target, alpha)
        # Keep Left Hand at Home
        T_curr_L = T_left_base
        q_sol, _ = arm_ik.solve_ik(
            left_wrist=T_curr_L.homogeneous,
            right_wrist=T_curr_R.homogeneous,
            current_lr_arm_motor_q=q_inter,
            use_filter=False
        )
        q_inter = q_sol
    L_fk, R_fk = arm_ik.forward_kinematics(np.array(q_sol))
    print("q_sol pos err R:", np.linalg.norm(R_fk.translation - T_right_target.translation))
    print("Sol joint angles:\n", q_sol)
    q_goal = copy.deepcopy(q_sol)

    path = plan(
        g1, 
        q_start, 
        q_goal,
        planner="RRTConnect", 
        planning_time=5.0, 
        smooth_path=True, 
        num_waypoints=100,
    )
    if path is None:
        print("Planning failed.")
    else:
        print(f"[Success] Total path waypoints: {len(path)}")

    step_count = 0
    try:
        if path:
            while True:
                save_this_cycle = not save_once_done
                for q_target in path:
                    contacts = collision_check(g1, q_target)
                    update_collision_display(vis, contacts, point_size=0.04)
                    vis.display(q_target)
                    q_current = q_target
                    step_count += 1
                    if save_this_cycle and step_count % 10 == 0:
                        if save_meshcat_frame(vis, img_dir, saved_frame_count):
                            saved_frame_count += 1
                    time.sleep(0.02)
                
                for q_target in reversed(path):
                    contacts = collision_check(g1, q_target)
                    update_collision_display(vis, contacts, point_size=0.04)
                    vis.display(q_target)
                    q_current = q_target
                    time.sleep(0.02)

                if save_this_cycle:
                    save_once_done = True
                    print(f"[Recorder] Saved {saved_frame_count} frames (first full loop only).")
                
                time.sleep(1.0)

    # try:
        else:
            print("Reached Initial Pose. Starting Loop. Press Ctrl+C to stop.")
            input("Press Enter to move to initial pose...")
            while True:
                save_this_cycle = not save_once_done
                steps = 100
                for i in range(steps):
                    alpha = i / steps
                    # Interpolate Right Hand
                    T_curr_R = interpolate_pose(T_right_base, T_right_target, alpha)
                    # Keep Left Hand at Home
                    T_curr_L = T_left_base
                    
                    q_sol, _ = arm_ik.solve_ik(
                        left_wrist=T_curr_L.homogeneous,
                        right_wrist=T_curr_R.homogeneous,
                        current_lr_arm_motor_q=q_current
                    )

                    contacts = collision_check(g1, q_sol)
                    update_collision_display(vis, contacts, point_size=0.04)
                    
                    q_current = q_sol
                    vis.display(q_sol)
                    if save_this_cycle:
                        if save_meshcat_frame(vis, img_dir, saved_frame_count):
                            saved_frame_count += 1
                    time.sleep(0.02)
                
                for i in range(steps):
                    alpha = i / steps
                    # Interpolate Right Hand
                    T_curr_R = interpolate_pose(T_right_target, T_right_base, alpha)
                    # Keep Left Hand at Home
                    T_curr_L = T_left_base
                    q_sol, _ = arm_ik.solve_ik(
                        left_wrist=T_curr_L.homogeneous,
                        right_wrist=T_curr_R.homogeneous,
                        current_lr_arm_motor_q=q_current
                    )
                    contacts = collision_check(g1, q_sol)
                    update_collision_display(vis, contacts, point_size=0.04)

                    q_current = q_sol
                    vis.display(q_sol)
                    if save_this_cycle:
                        if save_meshcat_frame(vis, img_dir, saved_frame_count):
                            saved_frame_count += 1
                    time.sleep(0.02)

                if save_this_cycle:
                    save_once_done = True
                    print(f"[Recorder] Saved {saved_frame_count} frames (first full loop only).")
                
                time.sleep(1.0)

    except KeyboardInterrupt:
        pass