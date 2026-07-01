"""
OMPL + Pinocchio G1 Robot Manipulation Tutorial
Scenario: Inserting stool legs into an IKEA stool on a table.

This script demonstrates:
1. Setting up a Pinocchio robot model with collision geometries.
2. Adding environmental obstacles (Table, Stool) to the collision model.
3. Using OMPL to plan a collision-free path in configuration space.
4. Visualizing the result in MuJoCo.
"""

import sys
import os
import time
import numpy as np
import pinocchio as pin
import coal
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import copy
import shutil

OMPL_AVAILABLE = False
ob = None
og = None

# OMPL imports (optional)
try:
    from ompl import base as ob  # type: ignore
    from ompl import geometric as og  # type: ignore
    OMPL_AVAILABLE = True
    print("[✓] OMPL imported successfully")
except Exception as e:
    print("[✗] Could not import OMPL (planning disabled)")
    print(f"    Reason: {type(e).__name__}: {e}")
    print("    Suggested fix: ensure OMPL is installed and NumPy ABI is compatible")
    print("    e.g. conda/mamba: mamba install -c conda-forge 'numpy<2' ompl")
    raise e

MUJOCO_AVAILABLE = False

# Mujoco imports (optional)
try:
    import mujoco  # type: ignore
    import mujoco.viewer  # type: ignore
    MUJOCO_AVAILABLE = True
    print("[✓] Mujoco imported successfully")
except Exception as e:
    print("[✗] Could not import mujoco (visualization disabled)")
    print(f"    Reason: {type(e).__name__}: {e}")
    print("    Installation: pip install mujoco")
    raise e

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))  # pose
parent_dir = os.path.dirname(current_dir)  # deploy_real
project_root = os.path.dirname(parent_dir)  # project root
sys.path.insert(0, parent_dir)

from g1_arm_IK import G1_29_ArmIK

class GraspController:
    """Control hand grasping and object attachment in Pinocchio.
    
    Since finger joints are locked in the reduced model, this implements:
    1. Visual hand state (open/closed) for MuJoCo visualization
    2. Kinematic attachment: re-parenting object geometry to hand frame
    3. Collision pair management: disable object-robot collisions when grasped
    """
    
    def __init__(self, robot, hand_frame_name: str = "R_ee"):
        self.robot = robot
        self.model = robot.model
        self.collision_model = robot.collision_model
        self.hand_frame_name = hand_frame_name
        self.hand_frame_id = self.model.getFrameId(hand_frame_name)
        
        # Grasp state
        self.is_grasping = False
        self.grasped_object_id = None
        self.grasped_object_name = None
        self.original_parent_joint = None
        self.original_placement = None
        self.robot_geom_count = None  # Set during attach
        
        # Hand finger joints (from URDF - right hand as example)
        # These are locked in reduced model, but we track them for future use
        self.finger_joint_names = [
            "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint",
            "right_index_1_joint", "right_index_2_joint",
            "right_middle_1_joint", "right_middle_2_joint",
            "right_ring_1_joint", "right_ring_2_joint",
            "right_little_1_joint", "right_little_2_joint",
        ]
        
    def get_hand_state(self) -> dict:
        """Return current hand state for visualization/debugging."""
        return {
            "is_grasping": self.is_grasping,
            "grasped_object": self.grasped_object_name,
            "hand_frame": self.hand_frame_name,
        }
    
    def set_finger_positions(self, positions: dict, mj_model=None, mj_data=None):
        """Set finger joint positions (for MuJoCo visualization).
        
        Args:
            positions: dict mapping joint_name -> angle (rad)
            mj_model: MuJoCo model (optional, for visualization)
            mj_data: MuJoCo data (optional, for visualization)
        """
        if mj_model is None or mj_data is None:
            # Fingers are locked in Pinocchio reduced model, skip
            return
        
        for joint_name, angle in positions.items():
            try:
                jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jnt_id >= 0:
                    qpos_addr = mj_model.jnt_qposadr[jnt_id]
                    mj_data.qpos[qpos_addr] = angle
            except:
                pass  # Joint not found or not controllable
    
    def close_hand(self, mj_model=None, mj_data=None, closure_amount: float = 0.8):
        """Close hand fingers (visual only in MuJoCo, since fingers are locked in Pinocchio).
        
        Args:
            closure_amount: 0.0 (fully open) to 1.0 (fully closed)
        """
        # Typical finger closure angles (rad) for Inspire Hand
        # Thumb: 3 DOF rotation + flex
        # Others: 2 DOF flex
        max_angles = {
            "right_thumb_1_joint": 0.5 * closure_amount,   # rotation
            "right_thumb_2_joint": 1.2 * closure_amount,   # flex
            "right_thumb_3_joint": 0.8 * closure_amount,
            "right_thumb_4_joint": 0.6 * closure_amount,
            "right_index_1_joint": 1.0 * closure_amount,
            "right_index_2_joint": 0.8 * closure_amount,
            "right_middle_1_joint": 1.0 * closure_amount,
            "right_middle_2_joint": 0.8 * closure_amount,
            "right_ring_1_joint": 1.0 * closure_amount,
            "right_ring_2_joint": 0.8 * closure_amount,
            "right_little_1_joint": 1.0 * closure_amount,
            "right_little_2_joint": 0.8 * closure_amount,
        }
        self.set_finger_positions(max_angles, mj_model, mj_data)
        print(f"[Grasp] Hand closed (visual, amount={closure_amount:.2f})")
    
    def open_hand(self, mj_model=None, mj_data=None):
        """Open hand fingers (visual only)."""
        zero_angles = {name: 0.0 for name in self.finger_joint_names}
        self.set_finger_positions(zero_angles, mj_model, mj_data)
        print("[Grasp] Hand opened (visual)")
    
    def attach_object(self, q_current: np.ndarray, object_name: str, robot_geom_count: int):
        """Attach object to hand frame (kinematic attachment in Pinocchio).
        
        Args:
            q_current: Current robot configuration
            object_name: Name of the geometry object to attach
            robot_geom_count: Number of robot geometries (before env objects)
        """
        if self.is_grasping:
            print(f"[Grasp] Warning: Already grasping {self.grasped_object_name}")
            return
        
        try:
            geom_id = self.collision_model.getGeometryId(object_name)
        except:
            print(f"[Grasp] Error: Object '{object_name}' not found in collision model")
            return
        
        # Update geometry placements to current configuration
        pin.forwardKinematics(self.model, self.robot.data, q_current)
        pin.updateFramePlacements(self.model, self.robot.data)
        pin.updateGeometryPlacements(
            self.model, self.robot.data,
            self.collision_model, self.robot.collision_data
        )
        
        # Get current poses
        oMf = self.robot.data.oMf[self.hand_frame_id]  # world -> hand_frame
        parent_joint = self.model.frames[self.hand_frame_id].parentJoint
        oMjoint = self.robot.data.oMi[parent_joint]    # world -> parent_joint
        oMg = self.robot.collision_data.oMg[geom_id]   # world -> object
        
        # Compute object placement relative to hand frame
        fMg = oMf.actInv(oMg)  # hand_frame -> object
        
        # Re-parent object to hand's parent joint
        geom_obj = self.collision_model.geometryObjects[geom_id]
        self.original_parent_joint = geom_obj.parentJoint
        self.original_placement = geom_obj.placement.copy()
        
        # Update to attach to hand
        jointMf = oMjoint.actInv(oMf)  # joint -> hand_frame
        jointMg = jointMf * fMg        # joint -> object
        
        geom_obj.parentJoint = parent_joint
        if hasattr(geom_obj, "parentFrame"):
            geom_obj.parentFrame = self.hand_frame_id
        geom_obj.placement = jointMg
        self.collision_model.geometryObjects[geom_id] = geom_obj
        
        # Remove object-robot collision pairs (avoid self-collision)
        new_pairs = []
        for cp in self.collision_model.collisionPairs:
            a, b = cp.first, cp.second
            involves_obj = (a == geom_id) or (b == geom_id)
            involves_robot = (a < robot_geom_count) or (b < robot_geom_count)
            if involves_obj and involves_robot:
                continue  # Skip this pair
            new_pairs.append(cp)
        
        self.collision_model.collisionPairs = new_pairs
        self.robot.collision_data = self.collision_model.createData()
        
        # Update state
        self.is_grasping = True
        self.grasped_object_id = geom_id
        self.grasped_object_name = object_name
        self.robot_geom_count = robot_geom_count
        
        print(f"[Grasp] Attached '{object_name}' to {self.hand_frame_name}")
        print(f"[Grasp] Removed {len(self.collision_model.collisionPairs)} collision pairs")
    
    def detach_object(self):
        """Detach object from hand (release grasp)."""
        if not self.is_grasping:
            print("[Grasp] Warning: Not grasping any object")
            return
        
        if self.grasped_object_id is None:
            print("[Grasp] Error: No grasped object ID stored")
            return
        
        # Restore original parent and placement
        geom_obj = self.collision_model.geometryObjects[self.grasped_object_id]
        geom_obj.parentJoint = self.original_parent_joint
        geom_obj.placement = self.original_placement
        if hasattr(geom_obj, "parentFrame"):
            geom_obj.parentFrame = 0  # Reset to world frame
        self.collision_model.geometryObjects[self.grasped_object_id] = geom_obj
        
        # Restore robot-object collision pairs
        if self.robot_geom_count is not None:
            for robot_gid in range(self.robot_geom_count):
                self.collision_model.addCollisionPair(
                    pin.CollisionPair(robot_gid, self.grasped_object_id)
                )
        
        self.robot.collision_data = self.collision_model.createData()
        
        print(f"[Grasp] Detached '{self.grasped_object_name}' from {self.hand_frame_name}")
        
        # Reset state
        self.is_grasping = False
        self.grasped_object_id = None
        self.grasped_object_name = None
        self.original_parent_joint = None
        self.original_placement = None

class OMPLPlannerWithCollision:
    def __init__(self, arm_ik: G1_29_ArmIK, *, debug_collisions: bool = False, debug_every: int = 2000, debug_max_pairs: int = 30):
        self.arm_ik = arm_ik
        self.robot = self.arm_ik.reduced_robot
        self.model = self.robot.model
        self.data = self.robot.data
        self.collision_model = self.robot.collision_model
        self.collision_data = self.robot.collision_data
        
        self.joint_lower = self.model.lowerPositionLimit
        self.joint_upper = self.model.upperPositionLimit
        self.n_joints = self.model.nq

        # Debug options (OMPL 会高频调用 is_valid，必须限频/限量)
        self.debug_collisions = bool(debug_collisions)
        self.debug_every = int(debug_every)
        self.debug_max_pairs = int(debug_max_pairs)
        self._validity_calls = 0
        self.last_collision_pairs = []  # 保存最近一次检测到的碰撞对（便于外部查看）
        
        # Ensure collision model exists (G1_29_ArmIK cache might skip it)
        self.ensure_collision_model()
        
        # Initialize grasp controller
        self.grasp_controller = GraspController(self.robot, hand_frame_name="R_ee")
        
        # Scene obstacles are built via build_scene(phase, scene_cfg)
        
        print(f"\n[Planner] Configuration:")
        print(f"  - DOF: {self.n_joints}")
        print(f"  - Joint limits: [{self.joint_lower.min():.2f}, {self.joint_upper.max():.2f}]")

    def _collect_colliding_pairs(self):
        """Collect colliding geometry-object name pairs from collision_data."""
        pairs = []
        # collisionResults 与 collisionPairs 一一对应
        for k, res in enumerate(self.collision_data.collisionResults):
            try:
                collided = bool(res.isCollision())
            except Exception:
                # 兼容性兜底：如果绑定差异导致无 isCollision()
                collided = False

            if not collided:
                continue

            cp = self.collision_model.collisionPairs[k]
            go1 = self.collision_model.geometryObjects[cp.first].name
            go2 = self.collision_model.geometryObjects[cp.second].name
            pairs.append((go1, go2))
        return pairs

    def ensure_collision_model(self):
        """
        If the robot was loaded from cache, it might lack collision geometry.
        We need to reload it from URDF to get the collision model.
        """
        if self.collision_model is None:
            raise RuntimeError(
                "[Planner] collision_model is missing. "
                "Please construct G1_29_ArmIK with ForceReload=True so the reduced robot is built from URDF with collision geometries."
            )

        print("[Planner] Collision model already present.")

    @staticmethod
    def _rpy_to_rot(rpy):
        # MuJoCo's `euler` with seq="xyz" matches SciPy intrinsic "XYZ".
        return R.from_euler('XYZ', rpy).as_matrix()

    def _add_env_object(self, name: str, geometry, placement: pin.SE3, *, color=None, robot_geom_count: int):
        env_obj = pin.GeometryObject(name, 0, geometry, placement)
        if color is not None:
            env_obj.meshColor = np.asarray(color, dtype=float)
        env_id = self.collision_model.addGeometryObject(env_obj)

        # Add robot-env collision pairs explicitly (Pinocchio does not auto-generate them).
        for robot_gid in range(robot_geom_count):
            self.collision_model.addCollisionPair(pin.CollisionPair(robot_gid, env_id))

        return env_id

    def build_scene(self, phase: str, scene_cfg: dict):
        """Build collision world for a given phase.

        phase:
          - 'pickup'    : leg is a static obstacle at pickup pose
          - 'carry'     : leg is attached to the R_ee frame
          - 'installed' : leg is a static obstacle at insertion pose
        """
        if phase not in {"pickup", "carry", "installed"}:
            raise ValueError(f"Unknown phase: {phase}")

        # Clone robot collision model (each phase has its own env objects + pairs)
        self.collision_model = copy.deepcopy(self.robot.collision_model)
        self.collision_data = self.collision_model.createData()
        self.robot.collision_model = self.collision_model
        self.robot.collision_data = self.collision_data

        robot_geom_count = len(self.collision_model.geometryObjects)

        # --- Table ---
        table_size = scene_cfg["table"]["size"]
        table_center = scene_cfg["table"]["center"]
        table_geom = coal.Box(*table_size)
        table_pose = pin.SE3(np.eye(3), np.array(table_center, dtype=float))
        self._add_env_object(
            "table",
            table_geom,
            table_pose,
            color=[0.6, 0.4, 0.2, 1.0],
            robot_geom_count=robot_geom_count,
        )

        # --- Stool (static mesh) ---
        stool_off = scene_cfg["stool"].get("mesh_offset", (0.0, 0.0, 0.0))
        stool_mesh = coal.MeshLoader().load(scene_cfg["stool"]["mesh_path"])
        stool_pose_ref = pin.SE3(
            self._rpy_to_rot(scene_cfg["stool"]["rpy"]),
            np.array(scene_cfg["stool"]["xyz"], dtype=float),
        )
        stool_pose = stool_pose_ref * pin.SE3(np.eye(3), np.array(stool_off, dtype=float))
        self._add_env_object(
            "stool_top",
            stool_mesh,
            stool_pose,
            color=[0.9, 0.9, 0.9, 1.0],
            robot_geom_count=robot_geom_count,
        )

        # --- Leg (mesh): static or attached depending on phase ---
        leg_off = scene_cfg["leg"].get("mesh_offset", (0.0, 0.0, 0.0))
        leg_mesh = coal.MeshLoader().load(scene_cfg["leg"]["mesh_path"])

        if phase == "pickup":
            leg_pose_ref = pin.SE3(
                self._rpy_to_rot(scene_cfg["leg"]["pickup_rpy"]),
                np.array(scene_cfg["leg"]["pickup_xyz"], dtype=float),
            )
            leg_pose = leg_pose_ref * pin.SE3(np.eye(3), np.array(leg_off, dtype=float))
            self._add_env_object(
                "stool_leg_pickup",
                leg_mesh,
                leg_pose,
                color=[0.8, 0.2, 0.2, 1.0],
                robot_geom_count=robot_geom_count,
            )
        elif phase == "installed":
            leg_pose_ref = pin.SE3(
                self._rpy_to_rot(scene_cfg["leg"]["installed_rpy"]),
                np.array(scene_cfg["leg"]["installed_xyz"], dtype=float),
            )
            leg_pose = leg_pose_ref * pin.SE3(np.eye(3), np.array(leg_off, dtype=float))
            self._add_env_object(
                "stool_leg_installed",
                leg_mesh,
                leg_pose,
                color=[0.8, 0.2, 0.2, 1.0],
                robot_geom_count=robot_geom_count,
            )
        else:  # carry
            r_ee_fid = self.model.getFrameId("R_ee")
            parent_joint = self.model.frames[r_ee_fid].parentJoint

            T_ee_leg = np.asarray(scene_cfg["leg"]["T_ee_leg"], dtype=float)
            if T_ee_leg.shape != (4, 4):
                raise ValueError("scene_cfg['leg']['T_ee_leg'] must be a 4x4 transform")
            placement = pin.SE3(T_ee_leg[:3, :3], T_ee_leg[:3, 3])

            attached = pin.GeometryObject(
                "stool_leg_attached",
                parent_joint,
                r_ee_fid,
                leg_mesh,
                placement,
            )
            attached.meshColor = np.array([0.8, 0.2, 0.2, 1.0])
            attached_id = self.collision_model.addGeometryObject(attached)

            # Only check attached-leg vs environment objects (avoid robot-vs-attached pairs)
            for env_gid in range(robot_geom_count, attached_id):
                self.collision_model.addCollisionPair(pin.CollisionPair(env_gid, attached_id))

        self.collision_data = pin.GeometryData(self.collision_model)
        print(f"[Collision] Scene built for phase='{phase}'.")

    def is_valid(self, state):
        """OMPL state validity callback: joint limits + collisions."""
        self._validity_calls += 1

        q = np.zeros(self.n_joints, dtype=float)
        for i in range(self.n_joints):
            q[i] = float(state[i])
            if q[i] < self.joint_lower[i] or q[i] > self.joint_upper[i]:
                return False

        stop_at_first = not self.debug_collisions
        in_collision = pin.computeCollisions(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q,
            stop_at_first,
        )
        if in_collision:
            pairs = self._collect_colliding_pairs()
            self.last_collision_pairs = pairs

            if self.debug_collisions and (self._validity_calls % self.debug_every == 0):
                print(f"[Collision][is_valid #{self._validity_calls}] q causes collision, pairs (showing up to {self.debug_max_pairs}):")
                for a, b in pairs[: self.debug_max_pairs]:
                    print(f"  - {a}  <->  {b}")
                if len(pairs) > self.debug_max_pairs:
                    print(f"  ... {len(pairs) - self.debug_max_pairs} more")
            return False

        self.last_collision_pairs = []
        return True

    def plan(self, q_start, q_goal, planning_time=5.0):
        """Plan from start to goal configuration."""
        if not OMPL_AVAILABLE:
            raise RuntimeError(
                "[Planner] OMPL is not available (planning disabled). "
                "Fix OMPL import first (common fix: install a NumPy-compatible OMPL build or downgrade to numpy<2)."
            )

        space = ob.RealVectorStateSpace(self.n_joints)
        bounds = ob.RealVectorBounds(self.n_joints)
        for i in range(self.n_joints):
            bounds.setLow(i, float(self.joint_lower[i]))
            bounds.setHigh(i, float(self.joint_upper[i]))
        space.setBounds(bounds)

        ss = og.SimpleSetup(space)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        start = ob.State(space)
        goal = ob.State(space)
        for i in range(self.n_joints):
            start[i] = float(q_start[i])
            goal[i] = float(q_goal[i])

        if not self.is_valid(start):
            print("[Planner] Error: Start state is invalid (collision or limits)!")
            return None
        if not self.is_valid(goal):
            print("[Planner] Error: Goal state is invalid (collision or limits)!")
            return None

        ss.setStartAndGoalStates(start, goal, 0.02)

        planner = og.RRTConnect(ss.getSpaceInformation())
        planner.setRange(0.1)
        ss.setPlanner(planner)

        print(f"[Planner] Starting planning (timeout: {planning_time}s)...")
        solved = ss.solve(float(planning_time))
        if not solved:
            print("[Planner] No solution found.")
            return None

        ss.simplifySolution()
        path = ss.getSolutionPath()
        path.interpolate(100)

        waypoints = []
        for i in range(path.getStateCount()):
            st = path.getState(i)
            waypoints.append(np.array([st[j] for j in range(self.n_joints)], dtype=float))
        return waypoints

def _normalize(v, eps=1e-9):
    """Normalize vector with safety check"""
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize zero vector")
    return v / n

def compute_initial_ee_pose_in_base(T_base_obj_first, grasp_style='side', offset_base=np.array([0.0, -0.05, 0.0])):
    """
    根据首帧物体位姿和抓取方式，计算末端执行器在基座坐标系下的初始位姿
    
    这一步定义：在基座坐标系下，末端执行器应该处于什么位姿来抓取物体
    
    Args:
        T_base_obj_first: (4,4) 首帧物体在基座系下的位姿
        grasp_style: 抓取方式
        offset_base: (3,) 在基座坐标系下的位置偏移
    
    Returns:
        pin.SE3 - T_base_ee_first (末端执行器在基座系下的初始位姿)
    """
    obj_position = T_base_obj_first[:3, 3]
    obj_rotation = T_base_obj_first[:3, :3]
    
    if grasp_style == 'side':
        R_base_ee = np.eye(3)
        t_base_ee = obj_position + offset_base
        
    elif grasp_style == 'top':
        ee_y = np.array([0.0, 0.0, -1.0])
        ee_x = np.array([0.0, 1.0, 0.0])
        ee_z = np.array([-1.0, 0.0, 0.0])
        R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        t_base_ee = obj_position + offset_base
        
    elif grasp_style == 'front':
        ee_x = np.array([1.0, 0.0, 0.0])
        
        obj_z = obj_rotation[:, 2]
        obj_z_horizontal = np.array([obj_z[0], obj_z[1], 0.0])
        if np.linalg.norm(obj_z_horizontal) > 0.1:
            ee_z = _normalize(obj_z_horizontal)
        else:
            ee_z = np.array([0.0, 0.0, 1.0])
        
        ee_y = _normalize(np.cross(ee_z, ee_x))
        ee_x = np.cross(ee_y, ee_z)
        
        R_base_ee = np.stack([ee_x, ee_y, ee_z], axis=1)
        t_base_ee = obj_position + np.array([-0.08, 0.0, 0.0])
        
    else:
        raise ValueError(f"Unknown grasp_style: {grasp_style}")
    
    return pin.SE3(R_base_ee, t_base_ee)

def debug_geom_pose(model, data, geom_name: str):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"geom not found: {geom_name}")
    mujoco.mj_forward(model, data)
    print("geom_name:", geom_name, "gid:", gid)
    print("model.geom_pos (local):", model.geom_pos[gid])
    print("data.geom_xpos (world origin):", data.geom_xpos[gid])
    bid = int(model.geom_bodyid[gid])
    print("parent body:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid))

def create_scene_xml(scene_cfg: dict = None) -> str:
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("[Scene] MuJoCo is required to generate the Inspire-hand MJCF from URDF.")

    assets_dir = os.path.join(project_root, 'deploy_real/assets')
    output_path = os.path.join(assets_dir, 'g1_stool_scene.xml')

    # Ensure stool meshes are reachable from assets/meshes/ (template uses meshdir="meshes")
    meshes_dir = os.path.join(assets_dir, 'meshes')
    os.makedirs(meshes_dir, exist_ok=True)
    stool_src = scene_cfg["stool"]["mesh_path"]
    leg_src = scene_cfg["leg"]["mesh_path"]
    stool_dst = os.path.join(meshes_dir, 'stool.obj')
    leg_dst = os.path.join(meshes_dir, 'stool-leg.obj')
    if os.path.exists(stool_src):
        shutil.copyfile(stool_src, stool_dst)
    else:
        raise FileNotFoundError(f"Stool mesh not found: {stool_src}")
    if os.path.exists(leg_src):
        shutil.copyfile(leg_src, leg_dst)
    else:
        raise FileNotFoundError(f"Stool leg mesh not found: {leg_src}")

    xml_file = ET.parse(os.path.join(assets_dir, 'g1_manul_scene.xml'))
    root = xml_file.getroot()
    wb = root.find('worldbody')

    # Table (box size is HALF extents in MJCF)
    table_center = scene_cfg["table"]["center"]
    table_size = scene_cfg["table"]["size"]
    table_half = (table_size[0] / 2.0, table_size[1] / 2.0, table_size[2] / 2.0)
    
    table_body = ET.SubElement(
        wb,
        "body",
        {
            "name": "table_body",
            "pos": f"{table_center[0]} {table_center[1]} {table_center[2]}",
        },
    )
    ET.SubElement(
        table_body,
        "geom",
        {
            "name": "table",
            "type": "box",
            "size": f"{table_half[0]} {table_half[1]} {table_half[2]}",
            "pos": "0 0 0",
            "rgba": "0.6 0.4 0.2 1",
            "contype": "1",
            "conaffinity": "1",
        },
    )

    # Stool placement: compute Z so the bottom rests on the table top.
    stool_xyz = scene_cfg["stool"]["xyz"]
    stool_euler = scene_cfg["stool"]["rpy"]
    stool_off = scene_cfg["stool"].get("mesh_offset", (0.0, 0.0, 0.0))
    
    stool_body = ET.SubElement(
        wb,
        "body",
        {
            "name": "stool_top_body",
            "pos": f"{stool_xyz[0]} {stool_xyz[1]} {stool_xyz[2]}",
            "euler": f"{stool_euler[0]} {stool_euler[1]} {stool_euler[2]}",
        },
    )
    ET.SubElement(
        stool_body,
        "geom",
        {
            "name": "stool_top",
            "type": "mesh",
            "mesh": "stool_top_mesh",
            "pos": f"{stool_off[0]} {stool_off[1]} {stool_off[2]}",
            "rgba": "0.9 0.9 0.9 1",
            "contype": "1",
            "conaffinity": "1",
        },
    )

    # Stool leg mesh (pickup pose)
    leg_pickup_xyz = scene_cfg["leg"]["pickup_xyz"]
    leg_pickup_rpy = scene_cfg["leg"]["pickup_rpy"]
    leg_off = scene_cfg["leg"].get("mesh_offset", (0.0, 0.0, 0.0))
    
    leg_body = ET.SubElement(
        wb,
        "body",
        {
            "name": "stool_leg_body",
            "pos": f"{leg_pickup_xyz[0]} {leg_pickup_xyz[1]} {leg_pickup_xyz[2]}",
            "euler": f"{leg_pickup_rpy[0]} {leg_pickup_rpy[1]} {leg_pickup_rpy[2]}",
        },
    )
    ET.SubElement(
        leg_body,
        "geom",
        {
            "name": "stool_leg",
            "type": "mesh",
            "mesh": "stool_leg_mesh",
            "pos": f"{leg_off[0]} {leg_off[1]} {leg_off[2]}",
            "rgba": "0.8 0.2 0.2 1",
            "contype": "1",
            "conaffinity": "1",
        },
    )

    xml_file.write(output_path)
    print(f"[Scene] Created scene MJCF: {output_path}")
    return output_path

def get_arm_joint_indices(model):
    """Get joint indices for the 12 arm joints in MuJoCo."""
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("[Vis] MuJoCo is not available; cannot query joint indices.")
    # Note: These names must match the URDF/XML joint names
    joint_names = [
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
        'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
        'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_yaw_joint'
    ]
    
    qpos_addrs = []
    for name in joint_names:
        try:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = model.jnt_qposadr[jnt_id]
            qpos_addrs.append(qpos_addr)
        except:
            print(f"[Warning] Joint {name} not found in MuJoCo model")
            qpos_addrs.append(-1)
    return qpos_addrs

def get_T_from_mujoco(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str, debug: bool = False) -> np.ndarray:
    """Get the pose of a geom in the MuJoCo world frame."""
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("[Vis] MuJoCo is not available; cannot query geom pose.")
    
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id == -1:
        raise ValueError(f"Geom '{geom_name}' not found in MuJoCo model.")
    
    pos = data.geom_xpos[geom_id]
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    if debug:
        print(f"[Vis] Geom '{geom_name}' pose:")
        print(f"  Position: {pos}")
        print(f"  Rotation:\n{mat}")
        debug_geom_pose(model, data, geom_name)
    
    T = np.eye(4, dtype=float)
    T[:3, :3] = mat
    T[:3, 3] = pos
    return T

def _mujoco2robo(T_mujoco: np.ndarray) -> np.ndarray:
    """Convert MuJoCo world coordinates to robot base coordinates."""
    T_robo = T_mujoco.copy()
    T_robo[:3, 3] += np.array([0.0, 0.0, -0.793])  # adjust mujoco world coordinate to robot base coordinate
    return T_robo

def main():
    print("="*60)
    print("G1 Robot Stool Leg Insertion - OMPL Planning")
    print("="*60)
    
    # 1. Setup Scene
    table_center = (0.55, 0.0, 0.40)
    table_size = (0.77, 1.16, 0.80)  # full extents

    # stool top
    stool_mesh_path = os.path.join(project_root, 'deploy_real/assets_obj/stool/stool.obj')
    stool_xyz = (0.32, 0.0, 0.82)
    stool_euler = (0.0, 0.0, 0.0)

    # stool leg
    stool_leg_mesh_path = os.path.join(project_root, 'deploy_real/assets_obj/stool-leg/stool-leg.obj')
    stool_leg_xyz = (0.22, -0.15, 0.93)  # right-front
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
            "installed_xyz": hole_pos,
            "installed_rpy": (np.pi / 2, 0.0, 0.0),
            "T_ee_leg": np.eye(4),
            "mesh_offset": (0.0, 0.0, 0.0),
        },
    }
    scene_xml = create_scene_xml(scene_cfg)
    print(f"[Setup] Generated scene XML: {scene_xml}")

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # 2. Initialize IK
    print("[Setup] Initializing Robot Model...")
    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False, ForceReload=True)
    
    # 3. Define Task
    # A known reasonable arm home pose (copied from deploy_real/pose/test_loop.py)
    q_start = np.array([
        -0.06300107389688492,  1.6363739967346191,   0.06309694796800613,  1.433660864830017,
        -0.06913699209690094, -0.01609145849943161, -0.023728765547275543, -1.5769442319869995,
        -0.06358829885721207,  1.4105912446975708,   0.04851214215159416,  0.04398690164089203,
    ], dtype=float)
    qpos_addrs = get_arm_joint_indices(model)
    for i, addr in enumerate(qpos_addrs):
        if addr >= 0:
            data.qpos[addr] = q_start[i]
    mujoco.mj_forward(model, data)

    # Keep left hand where it is at start (avoid guessing world height)
    L0, _ = arm_ik.forward_kinematics(q_start)
    T_left_target = L0.homogeneous
    
    # --- Phase 1: pickup (leg is a static obstacle) ---

    # Right hand pre-grasp target: approach the pickup leg from -X
    T_leg = get_T_from_mujoco(model, data, "stool_leg", False)
    T_robo_leg = _mujoco2robo(T_leg)
    T_right_target = compute_initial_ee_pose_in_base(
                                                      T_robo_leg, 
                                                      grasp_style='side', 
                                                      offset_base=np.array([-0.10, -0.05, 0.0])
                                                    ).homogeneous
    print(f"[Task] Right hand target pose for pickup:\n{T_right_target}")
    
    print("[Task] Solving IK for goal configuration...")
    q_goal = arm_ik.solve_ik(T_left_target, T_right_target, q_start)
    
    if q_goal is None:
        print("[Error] IK failed to find a goal configuration.")
        return
        
    if isinstance(q_goal, tuple):
        q_goal = q_goal[0]
    q_goal = np.asarray(q_goal).flatten()
    
    print("[Task] Planning path to pre-grasp...")
    planner = OMPLPlannerWithCollision(arm_ik, debug_collisions=True, debug_every=1, debug_max_pairs=100)
    planner.build_scene("pickup", scene_cfg)
    path1 = planner.plan(q_start, q_goal, planning_time=10.0)
    if path1 is None:
        print("[Error] Planning to pre-grasp failed.")
        return
    print(f"[Success] Pre-grasp path found with {len(path1)} waypoints.")

    # --- Phase 2: carry/insert (leg attached to the hand) ---
    # planner2 = OMPLPlannerWithCollision(arm_ik)
    # planner2.build_scene("carry", scene_cfg)

    # installed_pos = np.array(scene_cfg["leg"]["installed_xyz"], dtype=float)
    # insert_target_pos_r = installed_pos + np.array([-0.10, 0.0, 0.0])
    # T_insert = np.eye(4)
    # T_insert[:3, 3] = insert_target_pos_r

    # print("[Task] Solving IK for insertion configuration...")
    # q_insert = arm_ik.solve_ik(T_left_target, T_insert, q_goal)
    # if isinstance(q_insert, tuple):
    #     q_insert = q_insert[0]
    # q_insert = np.asarray(q_insert).flatten()

    # print("[Task] Planning path to insertion...")
    # path2 = planner2.plan(q_goal, q_insert, planning_time=10.0)
    # if path2 is None:
    #     print("[Error] Planning to insertion failed.")
    #     return
    # print(f"[Success] Insertion path found with {len(path2)} waypoints.")

    # # --- Phase 3: retract (leg is fixed, must not hit it) ---
    # planner3 = OMPLPlannerWithCollision(arm_ik)
    # planner3.build_scene("installed", scene_cfg)

    # retract_pos_r = insert_target_pos_r + np.array([-0.10, 0.0, 0.0])
    # T_retract = np.eye(4)
    # T_retract[:3, 3] = retract_pos_r

    # print("[Task] Solving IK for retract configuration...")
    # q_retract = arm_ik.solve_ik(T_left_target, T_retract, q_insert)
    # if isinstance(q_retract, tuple):
    #     q_retract = q_retract[0]
    # q_retract = np.asarray(q_retract).flatten()

    # print("[Task] Planning path to retract...")
    # path3 = planner3.plan(q_insert, q_retract, planning_time=10.0)
    # if path3 is None:
    #     print("[Error] Planning to retract failed.")
    #     return
    # print(f"[Success] Retract path found with {len(path3)} waypoints.")

    path = path1
    print(f"[Success] Total path waypoints: {len(path)}")
    
    # 5. Visualize with grasp demonstration
    print("[Vis] Loading MuJoCo viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[Vis] Playing trajectory with grasp demo (Press ESC to exit)...")
        
        grasp_executed = False
        release_executed = False
        loop_count = 0
        
        while viewer.is_running():
            # Phase 1: Move to pre-grasp position
            for idx, q in enumerate(path):
                if not viewer.is_running(): break
                
                # Update joint positions
                for i, addr in enumerate(qpos_addrs):
                    if addr >= 0:
                        data.qpos[addr] = q[i]
                
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
                
                # Grasp at the end of path (near object)
                if idx == len(path) - 1 and not grasp_executed:
                    print("\n" + "="*60)
                    print("[Demo] Reached pre-grasp position")
                    print("="*60)
                    time.sleep(0.5)
                    
                    # Close hand (visual)
                    print("[Demo] Closing hand...")
                    for closure in np.linspace(0, 0.8, 10):
                        planner.grasp_controller.close_hand(model, data, closure)
                        mujoco.mj_forward(model, data)
                        viewer.sync()
                        time.sleep(0.05)
                    
                    # Attach object (kinematic)
                    print("[Demo] Attaching object to hand...")
                    # Note: We need to get robot_geom_count from planner
                    robot_geom_count = len([g for g in planner.collision_model.geometryObjects 
                                           if 'stool_leg' not in g.name and 'table' not in g.name and 'stool_top' not in g.name])
                    planner.grasp_controller.attach_object(q, "stool_leg_pickup", robot_geom_count)
                    
                    grasp_executed = True
                    print("[Demo] Grasp complete! Holding for 2 seconds...")
                    time.sleep(2.0)
            
            # Phase 2: Hold and show grasped state
            if grasp_executed and not release_executed:
                print("\n" + "="*60)
                print("[Demo] Object is grasped. State:")
                print(planner.grasp_controller.get_hand_state())
                print("="*60)
                time.sleep(2.0)
                
                # Release (for demo purposes - in real task this happens at installation)
                print("[Demo] Releasing object...")
                planner.grasp_controller.detach_object()
                
                # Open hand (visual)
                print("[Demo] Opening hand...")
                for closure in np.linspace(0.8, 0, 10):
                    planner.grasp_controller.close_hand(model, data, closure)
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(0.05)
                
                release_executed = True
                print("[Demo] Release complete!")
                time.sleep(2.0)
            
            # Reset for next loop
            if grasp_executed and release_executed:
                loop_count += 1
                print(f"\n[Demo] Loop {loop_count} complete. Restarting in 3 seconds...")
                time.sleep(3.0)
                grasp_executed = False
                release_executed = False

if __name__ == "__main__":
    main()
