import os
import json
import math
import shutil
from pathlib import Path

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

# ==========================================
# 다양한 언어 명령어 템플릿
# ==========================================
INSTRUCTION_TEMPLATES = {
    "grasp": [
        "grasp the {color} {desc}",
        "grab the {color} {desc}",
        "pick the {color} {desc}"
    ],
    "lift": [
        "lift the {color} {desc}",
        "pick up and hold the {color} {desc}",
        "raise the {color} {desc}"
    ],
    "push": [
        "push the {color} {desc}",
        "move the {color} {desc} forward",
        "slide the {color} {desc} away"
    ]
}


class DatasetLogger:
    def __init__(self, root_dir="dataset_raw", keep_failed=False):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed = keep_failed
        self.episode_dir = None
        self.meta = None

    def start_episode(
        self,
        episode_id,
        instruction,
        goal_xy,
        box_init_xy,
        box_init_yaw,
        task_type="pick",
        target_color=None,
        target_body_name=None,
        all_object_init_poses=None,
    ):
        episode_name = f"episode_{episode_id:06d}"
        self.episode_dir = self.root_dir / episode_name
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.meta = {
            "episode_id": int(episode_id),
            "instruction": str(instruction),
            "task_type": str(task_type),
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "box_init_xy": [float(box_init_xy[0]), float(box_init_xy[1])],
            "box_init_yaw": float(box_init_yaw),
            "success": False,
            "steps": []
        }

        if target_color is not None:
            self.meta["target_color"] = str(target_color)
        if target_body_name is not None:
            self.meta["target_body_name"] = str(target_body_name)
        if all_object_init_poses is not None:
            self.meta["all_object_init_poses"] = all_object_init_poses

    def log_step(
        self,
        step_idx,
        image_rgb,
        joint_angles,
        gripper_state,
        object_pose,
        ee_pose,
        action,
        is_first=False,
        is_last=False,
    ):
        image_file = f"frame_{step_idx:06d}.png"
        image_path = self.episode_dir / image_file
        Image.fromarray(image_rgb).save(image_path)

        step_data = {
            "t": int(step_idx),
            "image_file": image_file,
            "joint_angles": [float(x) for x in joint_angles],
            "gripper_state": float(gripper_state),
            "object_pose": [float(x) for x in object_pose],
            "ee_pose": [float(x) for x in ee_pose],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        }
        self.meta["steps"].append(step_data)

    def finalize_episode(self, success, exception_text=None):
        self.meta["success"] = bool(success)
        if exception_text is not None:
            self.meta["exception"] = str(exception_text)

        meta_path = self.episode_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

        if (not success) and (not self.keep_failed):
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def abort_episode(self):
        if self.episode_dir is not None and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class SyncSimRaccoonDataset:
    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    GRIPPER_SPEED = 15.0

    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0

    MODE_POSITION = 0
    MODE_VELOCITY = 1

    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85

    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1
    GRIP_MODE_VERT = 2

    OBJECT_MAP = {
        "red": {"body": "target_object", "type": "cylinder", "desc": "cylinder"},
        "blue": {"body": "target_object_blue", "type": "cylinder", "desc": "cylinder"},
        "green": {"body": "target_object_green", "type": "cylinder", "desc": "cylinder"},
        "yellow": {"body": "target_object_yellow", "type": "cylinder", "desc": "cylinder"},
        "red_cube": {"body": "target_object_cube_red", "type": "box", "desc": "cube"},
        "blue_sphere": {"body": "target_object_sphere_blue", "type": "sphere", "desc": "sphere"},
    }
    TARGET_COLORS = tuple(OBJECT_MAP.keys())

    DEFAULT_OBJECT_X_RANGE = (-0.12, 0.12)
    DEFAULT_OBJECT_Y_RANGE = (0.15, 0.19)
    DEFAULT_MIN_OBJECT_DISTANCE = 0.035

    def __init__(self, xml_path, image_size=(256, 256), camera_name=None, use_viewer=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer

        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.active_object_body_name = self.OBJECT_MAP["red"]["body"]

        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

        self.reset_episode(
            object_specs=self.make_default_object_specs(),
            target_color="red",
        )

    # ---------- kinematics / commands ----------
    def _calc_inv_kinematics(self, x, y, z):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)):
            if (-28.0 <= x <= 28.0) and (-15 <= y <= 28.0) and (0 <= z <= 36.25):
                x, y = y, -x
                th1 = math.atan2(y, x)
                c1 = math.cos(th1)
                s1 = math.sin(th1)
                x = x - self.L4 * c1
                y = y - self.L4 * s1
                zL1 = z - self.L1
                c3 = (x * x + y * y + zL1 * zL1 - self.L2 * self.L2 - self.L3 * self.L3) / (2 * self.L2 * self.L3)
                c32 = c3 * c3
                if c32 > 1:
                    c32 = 1
                s3 = -math.sqrt(1 - c32)
                th3 = math.atan2(s3, c3)
                M1 = c3 * self.L3 + self.L2
                M2 = z - self.L1
                M3 = s3 * self.L3
                M4 = c1 * x + s1 * y
                c2 = M1 * M2 - M3 * M4
                s2 = -M2 * M3 - M1 * M4
                th2 = math.atan2(s2, c2)
                th1 = math.degrees(th1)
                th2 = math.degrees(th2)
                th3 = math.degrees(th3)
                th4 = -(th2 + th3) - 90

                if th1 < -120 or th1 > 120:
                    return None
                if th2 < -90 or th2 > 30:
                    return None
                if th3 < -150 or th3 > 0:
                    return None

                return [th1, th2, th3, th4]
            return None
        return None

    def degree_to(self, joints, degrees, speed=70):
        j_list = joints if isinstance(joints, (list, tuple)) else [joints]
        d_list = degrees if isinstance(degrees, (list, tuple)) else [degrees]

        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)

        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                percent = np.clip(speed, 0.0, 100.0)
                self.joint_velocities[idx] = (percent / 100.0) * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm, y_cm, z_cm, speed=70):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표입니다: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self):
        self.gripper_target = self.GRIP_OPEN

    def close_gripper(self):
        self.gripper_target = self.GRIP_CLOSE

    def lockh(self):
        self.gripper_mode = self.GRIP_MODE_HORZ

    def lockv(self):
        self.gripper_mode = self.GRIP_MODE_VERT

    def unlock(self):
        if self.gripper_mode != self.GRIP_MODE_FREE:
            self.target_angles[3] = self.data.qpos[3]
            self.gripper_mode = self.GRIP_MODE_FREE

    def execute_action(self, action, speed=70):
        target_x, target_y, target_z, gripper = action
        self.move_to(target_x * 100.0, target_y * 100.0, target_z * 100.0, speed=speed)

        if gripper >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    # ---------- synchronous stepping ----------
    def _apply_controls_once(self):
        dt = self.model.opt.timestep

        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                if self.gripper_mode == self.GRIP_MODE_HORZ:
                    desired = base_angle - np.radians(90)
                else:
                    desired = base_angle - np.radians(180)

                error = desired - self.current_setpoints[i]
                speed_rad_s = self.MAX_SPEEDS[i]
                limit_step = speed_rad_s * dt
                step = np.clip(error, -limit_step, limit_step)
                self.current_setpoints[i] += step
            else:
                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt
                        step_val = np.clip(error, -max_step, max_step)
                        self.current_setpoints[i] += step_val

            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        try:
            touch_L = self.data.sensor("sensor_L").data[0]
            touch_R = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_L > 0.1) and (touch_R > 0.1)
        except Exception:
            is_touched = False

        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            g_move = np.clip(g_err, -g_step, g_step)
            self.current_setpoints[4] += g_move

        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps):
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def steps_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds=2.0):
        self.step_n(self.steps_for_seconds(seconds))

    # ---------- rendering / state ----------
    def get_robot_state(self):
        joint_angles = [float(self.data.qpos[i]) for i in range(4)]
        gripper_state = float(self.data.qpos[4])
        return {
            "joint_angles": joint_angles,
            "gripper_state": gripper_state
        }

    def get_object_pose(self, body_name="target_object"):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def render_rgb(self):
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        image = self.renderer.render()
        return image.copy()

    def get_observation(self, object_body_name=None):
        if object_body_name is None:
            object_body_name = self.active_object_body_name

        rs = self.get_robot_state()
        obj = self.get_object_pose(object_body_name)
        img = self.render_rgb()

        link4_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link4")
        if link4_id != -1:
            ee_pos = self.data.xpos[link4_id].copy()
            ee_pose_list = [float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])]
        else:
            ee_pose_list = [0.0, 0.0, 0.0]

        return {
            "image": img,
            "joint_angles": rs["joint_angles"],
            "gripper_state": rs["gripper_state"],
            "object_pose": obj,
            "ee_pose": ee_pose_list,
        }

    # ---------- reset / success ----------
    def reset_object_pose(self, body_name="target_object", x=0.15, y=0.15, z=0.02, yaw=0.0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num < 1:
            raise ValueError(f"{body_name} has no joint")

        joint_id = jnt_adr
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    @classmethod
    def make_default_object_specs(cls):
        x_values = np.linspace(
            cls.DEFAULT_OBJECT_X_RANGE[0] * 0.75,
            cls.DEFAULT_OBJECT_X_RANGE[1] * 0.75,
            len(cls.TARGET_COLORS),
        )
        y_center = float(sum(cls.DEFAULT_OBJECT_Y_RANGE) / 2.0)
        return {
            color: {
                "body_name": cls.OBJECT_MAP[color]["body"],
                "x": float(x_values[idx]),
                "y": y_center,
                "yaw": 0.0,
            }
            for idx, color in enumerate(cls.TARGET_COLORS)
        }

    @classmethod
    def sample_object_specs(
        cls,
        rng,
        colors=None,
        x_range=None,
        y_range=None,
        yaw_range=(-np.pi / 4, np.pi / 4),
        min_distance=None,
        max_tries=1000,
    ):
        colors = tuple(colors or cls.TARGET_COLORS)
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or cls.DEFAULT_OBJECT_Y_RANGE
        min_distance = cls.DEFAULT_MIN_OBJECT_DISTANCE if min_distance is None else min_distance

        if len(colors) == 0:
            raise ValueError("colors는 비어 있을 수 없습니다.")

        specs = {}
        placed_xy = []
        placement_order = list(colors)
        rng.shuffle(placement_order)

        for color in placement_order:
            if color not in cls.OBJECT_MAP:
                raise ValueError(f"지원하지 않는 색상/물체입니다: {color}")

            for _ in range(max_tries):
                x = float(rng.uniform(x_range[0], x_range[1]))
                y = float(rng.uniform(y_range[0], y_range[1]))
                xy = np.array([x, y], dtype=np.float64)

                if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                    specs[color] = {
                        "body_name": cls.OBJECT_MAP[color]["body"],
                        "x": x,
                        "y": y,
                        "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                    }
                    placed_xy.append(xy)
                    break
            else:
                raise RuntimeError(
                    f"물체 {len(colors)}개를 겹치지 않게 배치하지 못했습니다. "
                )

        return {color: specs[color] for color in colors}

    @staticmethod
    def specs_to_meta(object_specs):
        return {
            color: {
                "body_name": str(spec["body_name"]),
                "xy": [float(spec["x"]), float(spec["y"])],
                "yaw": float(spec["yaw"]),
            }
            for color, spec in object_specs.items()
        }

    def reset_colored_objects(self, object_specs, target_color):
        if target_color not in object_specs:
            raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

        self.active_object_body_name = object_specs[target_color]["body_name"]

        for color, spec in object_specs.items():
            body_name = spec["body_name"]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"body not found for color '{color}': {body_name}")

            self.reset_object_pose(
                body_name,
                x=spec["x"],
                y=spec["y"],
                z=0.02,
                yaw=spec["yaw"],
            )

    def reset_episode(self, object_specs, target_color="red"):
        home = np.radians([0.0, -10.0, -140.0, 60.0])

        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION

        self.data.qvel[:] = 0.0

        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE

        self.reset_colored_objects(object_specs=object_specs, target_color=target_color)
        mujoco.mj_forward(self.model, self.data)

        self.step_n(20)

    def get_gripper_touch_state(self):
        try:
            touch_l = float(self.data.sensor("sensor_L").data[0])
            touch_r = float(self.data.sensor("sensor_R").data[0])
        except Exception:
            touch_l = 0.0
            touch_r = 0.0
        return touch_l, touch_r

    def is_body_touching_robot(self, body_name, ignored_geom_names=("floor",)):
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target_body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        target_object_body_ids = set()
        for info in self.OBJECT_MAP.values():
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, info["body"])
            if b_id != -1:
                target_object_body_ids.add(b_id)

        ignored_geom_names = set(ignored_geom_names or [])

        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            if target_body_id not in (body1, body2):
                continue

            other_geom = geom2 if body1 == target_body_id else geom1
            other_body = body2 if body1 == target_body_id else body1

            other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_geom_name in ignored_geom_names:
                continue

            if other_body in target_object_body_ids:
                continue

            return True

        return False

    # ==========================================
    # 태스크(Task)별 행동 궤적 계획 (Z축 미세조정)
    # ==========================================
    def make_task_plan(self, box_x, box_y, obj_type="cylinder", task_type="grasp"):
        z_above = 0.10
        
        # 큐브와 구가 미끄러지지 않도록 Z 높이를 조금 더 내렸습니다. (기존 0.015 -> 0.012)
        if obj_type == "box":
            z_grasp = 0.012 
        elif obj_type == "sphere":
            z_grasp = 0.011 
        else:
            z_grasp = 0.020
            
        if task_type == "grasp":
            return [
                [box_x, box_y, z_above, 0],   
                [box_x, box_y, z_grasp, 0],   
                [box_x, box_y, z_grasp, 1],   
            ]
        elif task_type == "lift":
            return [
                [box_x, box_y, z_above, 0],   
                [box_x, box_y, z_grasp, 0],   
                [box_x, box_y, z_grasp, 1],   
                [box_x, box_y, z_above, 1],  
            ]
        elif task_type == "push":
            return [
                [box_x, box_y - 0.03, z_above, 0], 
                [box_x, box_y - 0.03, z_grasp, 1], 
                [box_x, box_y + 0.03, z_grasp, 1], 
            ]
        
        return []

    # ==========================================
    # 태스크별 성공 판별 로직 (터치 임계값 완화)
    # ==========================================
    def is_task_success(self, target_body_name, task_type, initial_obj_pose, touch_threshold=0.05):
        touch_l, touch_r = self.get_gripper_touch_state()
        # 큐브나 구는 센서에 살짝만 닿을 수 있으므로 threshold를 완화(0.1 -> 0.05)
        both_touched = (touch_l > touch_threshold) and (touch_r > touch_threshold)
        gripper_is_closed = float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        is_touching_robot = self.is_body_touching_robot(target_body_name)
        
        current_pose = self.get_object_pose(target_body_name)
        
        if task_type == "grasp":
            return bool(both_touched and gripper_is_closed and is_touching_robot)
        elif task_type == "lift":
            return bool(both_touched and gripper_is_closed and is_touching_robot and (current_pose[2] > initial_obj_pose[2] + 0.03))
        elif task_type == "push":
            return bool(current_pose[1] > initial_obj_pose[1] + 0.02) 
            
        return False

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if hasattr(self, 'renderer') and self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass


def run_episode_and_record(
    rc: SyncSimRaccoonDataset,
    logger: DatasetLogger,
    episode_id: int,
    object_specs: dict,
    target_color: str = "red",
    task_type: str = "grasp",
    rng=None,
    speed: int = 250, # 속도 최적화
    settle_seconds_per_action: float = 0.4, # 대기시간 최적화
    initial_settle_seconds: float = 0.1,
    hz: int = 10,
    touch_threshold: float = 0.05, # 임계값 완화
):
    target_spec = object_specs[target_color]
    target_body_name = target_spec["body_name"]
    target_x = float(target_spec["x"])
    target_y = float(target_spec["y"])
    target_yaw = float(target_spec["yaw"])

    obj_info = SyncSimRaccoonDataset.OBJECT_MAP[target_color]
    obj_type = obj_info["type"]
    obj_desc = obj_info["desc"]
    
    color_name = target_color.split('_')[0]
    instruction_template = rng.choice(INSTRUCTION_TEMPLATES[task_type])
    instruction = instruction_template.format(color=color_name, desc=obj_desc)

    rc.reset_episode(object_specs=object_specs, target_color=target_color)
    rc.lockh()

    if initial_settle_seconds > 0:
        rc.settle_steps(seconds=initial_settle_seconds)

    initial_obj_pose = rc.get_object_pose(target_body_name)

    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        task_type=task_type,
        goal_xy=[target_x, target_y],
        box_init_xy=[target_x, target_y],
        box_init_yaw=target_yaw,
        target_color=target_color,
        target_body_name=target_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
    )

    try:
        plan = rc.make_task_plan(target_x, target_y, obj_type=obj_type, task_type=task_type)

        obs = rc.get_observation()
        dt = 1.0 / hz
        step_counter = 0

        for action in plan:
            rc.execute_action(action, speed=speed)
            num_frames = int(settle_seconds_per_action * hz)

            for _ in range(num_frames):
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )
                rc.settle_steps(seconds=dt)
                obs = rc.get_observation()
                step_counter += 1

        logger.log_step(
            step_idx=step_counter,
            image_rgb=obs["image"],
            joint_angles=obs["joint_angles"],
            gripper_state=obs["gripper_state"],
            object_pose=obs["object_pose"],
            ee_pose=obs["ee_pose"],
            action=plan[-1],
            is_first=False,
            is_last=True,
        )

        success = rc.is_task_success(
            target_body_name=target_body_name,
            task_type=task_type,
            initial_obj_pose=initial_obj_pose,
            touch_threshold=touch_threshold,
        )
        logger.finalize_episode(success=success)
        return success

    except Exception as e:
        logger.abort_episode()
        raise e


# ==========================================
# [완벽 밸런싱 로직] 색상과 태스크의 '조합' 단위로 타겟 추적
# ==========================================
def _balanced_target_counts(num_episodes, combinations):
    base = num_episodes // len(combinations)
    remainder = num_episodes % len(combinations)
    return {
        combo: base + (1 if idx < remainder else 0)
        for idx, combo in enumerate(combinations)
    }

def _sample_remaining_combo(rng, target_counts, success_counts):
    remaining_combos = []
    remaining_weights = []

    for combo, target_count in target_counts.items():
        remaining = target_count - success_counts[combo]
        if remaining > 0:
            remaining_combos.append(combo)
            remaining_weights.append(remaining)

    if not remaining_combos:
        return None, None

    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_weights /= remaining_weights.sum()
    
    chosen_idx = rng.choice(len(remaining_combos), p=remaining_weights)
    return remaining_combos[chosen_idx]


def collect_dataset(
    xml_path="Raccoon_colored_cylinder.xml",
    dataset_root="raccoon_tasks_extended", 
    num_episodes=400,
    colors=("red", "blue", "green", "yellow", "red_cube", "blue_sphere"),
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    speed=250, # 속도 최적화
    settle_seconds_per_action=0.4, # 대기시간 최적화
    initial_settle_seconds=0.1,
    hz=10,
    touch_threshold=0.05, # 임계값 완화
    seed=None,
    max_attempts=None,
    object_x_range=(-0.12, 0.12), 
    object_y_range=(0.15, 0.19),  
    min_object_distance=0.035,
):
    colors = tuple(colors)
    tasks = ("grasp", "lift", "push")
    
    # 18가지 (색상 x 태스크) 조합 생성
    combinations = [(c, t) for c in colors for t in tasks]
    
    target_counts = _balanced_target_counts(num_episodes, combinations)
    rng = np.random.default_rng(seed)

    if max_attempts is None:
        max_attempts = max(num_episodes * 20, num_episodes + 100)

    rc = SyncSimRaccoonDataset(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
    )
    logger = DatasetLogger(root_dir=dataset_root, keep_failed=keep_failed)

    # 조합별 성공 횟수 추적
    success_counts = {combo: 0 for combo in combinations}
    attempt_count = 0

    try:
        while sum(success_counts.values()) < num_episodes and attempt_count < max_attempts:
            attempt_count += 1

            # 색상과 태스크를 조합 단위로 샘플링 (부족한 것 우선)
            target_color, task_type = _sample_remaining_combo(rng, target_counts, success_counts)
            if target_color is None:
                break

            try:
                object_specs = SyncSimRaccoonDataset.sample_object_specs(
                    rng=rng,
                    colors=colors,
                    x_range=object_x_range,
                    y_range=object_y_range,
                    min_distance=min_object_distance,
                )
            except RuntimeError as e:
                continue

            episode_id = attempt_count if keep_failed else (sum(success_counts.values()) + 1)

            try:
                success = run_episode_and_record(
                    rc=rc,
                    logger=logger,
                    episode_id=episode_id,
                    object_specs=object_specs,
                    target_color=target_color,
                    task_type=task_type,
                    rng=rng, 
                    speed=speed,
                    settle_seconds_per_action=settle_seconds_per_action,
                    initial_settle_seconds=initial_settle_seconds,
                    hz=hz,
                    touch_threshold=touch_threshold,
                )

                if success:
                    success_counts[(target_color, task_type)] += 1

                # 로깅 시 색상별, 태스크별 총합을 계산하여 예쁘게 출력
                color_totals = {c: sum(success_counts[(c, t)] for t in tasks) for c in colors}
                task_totals = {t: sum(success_counts[(c, t)] for c in colors) for t in tasks}

                print(
                    f"[Attempt {attempt_count:04d}] episode_id={episode_id:06d} | "
                    f"task_type='{task_type}' | color='{target_color}' | "
                    f"success={success} | Total Success={sum(success_counts.values())}/{num_episodes}\n"
                    f"   => Colors: {color_totals}\n"
                    f"   => Tasks : {task_totals}"
                )
            except Exception as e:
                print(f"[Attempt {attempt_count:04d}] task_type='{task_type}' | color='{target_color}' | exception: {e}")

    finally:
        rc.close()

    total_success = sum(success_counts.values())
    print(f"완료: success episodes = {total_success}/{num_episodes}, attempts = {attempt_count}")

if __name__ == "__main__":
    collect_dataset(
        xml_path="Raccoon_colored_cylinder.xml",
        dataset_root="raccoon_tasks_extended", 
        num_episodes=400,
        colors=("red", "blue", "green", "yellow", "red_cube", "blue_sphere"), 
        keep_failed=False,
        use_viewer=False,
        camera_name="front_view",
        speed=120,
        settle_seconds_per_action=1.0,
        initial_settle_seconds=0.1,
        touch_threshold=0.05,
    )