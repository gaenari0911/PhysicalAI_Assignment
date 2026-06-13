import argparse
import base64
import io
import json
import math
import os
import re
import time
from contextlib import nullcontext
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2  # [웹캠 연동] OpenCV 라이브러리
import mujoco
import numpy as np
import requests
from PIL import Image
from sshtunnel import SSHTunnelForwarder

from raccoon_env import SyncSimRaccoonEnv

try:
    from roboid import Raccoon
except ImportError:
    Raccoon = None

# ==========================================
# 6개의 물체 정보 매핑
# ==========================================
OBJECT_MAP = {
    "red": {"body": "target_object", "type": "cylinder", "desc": "cylinder"},
    "blue": {"body": "target_object_blue", "type": "cylinder", "desc": "cylinder"},
    "green": {"body": "target_object_green", "type": "cylinder", "desc": "cylinder"},
    "yellow": {"body": "target_object_yellow", "type": "cylinder", "desc": "cylinder"},
    "red_cube": {"body": "target_object_cube_red", "type": "box", "desc": "cube"},
    "blue_sphere": {"body": "target_object_sphere_blue", "type": "sphere", "desc": "sphere"},
}
TARGET_COLORS = tuple(OBJECT_MAP.keys())

DEFAULT_OBJECT_X_RANGE = (-0.13, 0.13)
DEFAULT_OBJECT_Y_RANGE = (0.16, 0.25)
DEFAULT_MIN_OBJECT_DISTANCE = 0.05
DEFAULT_YAW_RANGE = (-math.pi / 4, math.pi / 4)


def image_to_b64(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def request_action(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    timeout: float = 60.0,
) -> Dict[str, Any]:
    payload = {
        "instruction": instruction,
        "image_b64": image_to_b64(image_rgb),
        "unnorm_key": unnorm_key,
        "do_sample": False,
    }
    response = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    if not response.ok:
        print(f"[SERVER ERROR] {response.status_code} | {response.text}")
        response.raise_for_status()
    return response.json()


def resolve_ssh_password(args: argparse.Namespace) -> Optional[str]:
    if args.ssh_password:
        return args.ssh_password
    env_password = os.environ.get("OPENVLA_SSH_PASSWORD")
    if env_password:
        return env_password
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def open_ssh_tunnel(args: argparse.Namespace) -> SSHTunnelForwarder:
    ssh_password = resolve_ssh_password(args)
    tunnel = SSHTunnelForwarder(
        ssh_address_or_host=(args.ssh_host, args.ssh_port),
        ssh_username=args.ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=(args.remote_server_host, args.remote_server_port),
        local_bind_address=(args.local_server_host, args.local_server_port),
    )
    tunnel.start()
    return tunnel


def build_server_url(args: argparse.Namespace, tunnel: Optional[SSHTunnelForwarder]) -> str:
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required when --use_ssh_tunnel is not enabled.")
    return args.server_url


def maybe_tunnel_context(args: argparse.Namespace):
    if args.use_ssh_tunnel:
        return open_ssh_tunnel(args)
    return nullcontext(None)


class RealRaccoonController:
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0
    HOME_DEGREES = (0.0, -10.0, -140.0, 60.0)

    def __init__(
        self,
        require_ready: bool = True,
        home_wait_seconds: float = 5.0,
        beep_on_ready: bool = True,
    ) -> None:
        if Raccoon is None:
            raise ImportError(
                "roboid 패키지를 import할 수 없습니다. 실제 라쿤봇 제어 환경에서 실행하거나 "
                "--use_real_robot 옵션을 끄세요."
            )

        self.hw = Raccoon()
        ready = bool(getattr(getattr(self.hw, "_roboid", None), "_ready", False))
        if not ready:
            msg = "라쿤봇 하드웨어 연결에 실패했습니다. USB/Bluetooth 연결과 전원을 확인하세요."
            if require_ready:
                raise RuntimeError(msg)
            print(f"[REAL_ROBOT WARN] {msg} 시뮬레이션 명령만 계속합니다.")
            self.hw = None
            return

        self.go_home(wait_seconds=home_wait_seconds)
        self.lockh()
        self.open_gripper()

        if beep_on_ready:
            try:
                self.hw.beep()
            except Exception as exc:
                print(f"[REAL_ROBOT WARN] beep 실패: {exc}")

        print("[REAL_ROBOT] 하드웨어 연결 성공")

    @property
    def connected(self) -> bool:
        return self.hw is not None

    def _try_call(self, fn_name: str, *candidate_args: Sequence[Any]) -> bool:
        if not self.connected:
            return False
        fn = getattr(self.hw, fn_name, None)
        if fn is None:
            return False

        last_error: Optional[Exception] = None
        for args in candidate_args:
            try:
                fn(*args)
                return True
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return False

    def _send_joint_degrees(self, degrees: Sequence[float], speed: int = 70) -> None:
        degrees = [float(v) for v in degrees[:4]]
        speed = int(speed)

        if self._try_call("set_degree", (*degrees, speed), tuple(degrees)):
            return

        if self._try_call("degree_to", (*degrees, speed), tuple(degrees), ([1, 2, 3, 4], degrees, speed)):
            return

        per_joint_ok = True
        for joint_id, degree in enumerate(degrees, start=1):
            if not (
                self._try_call("set_degree", (joint_id, degree, speed), (joint_id, degree))
                or self._try_call("degree_to", (joint_id, degree, speed), (joint_id, degree))
            ):
                per_joint_ok = False
                break
        if per_joint_ok:
            return

        raise AttributeError("Raccoon 객체에서 set_degree/degree_to 관절 제어 API를 찾지 못했습니다.")

    def _calc_inv_kinematics(self, x_cm: float, y_cm: float, z_cm: float) -> Optional[list[float]]:
        if not (
            isinstance(x_cm, (int, float))
            and isinstance(y_cm, (int, float))
            and isinstance(z_cm, (int, float))
        ):
            return None

        if not ((-28.0 <= x_cm <= 28.0) and (-15.0 <= y_cm <= 28.0) and (0.0 <= z_cm <= 36.25)):
            return None

        x, y, z = y_cm, -x_cm, z_cm
        th1 = math.atan2(y, x)
        c1 = math.cos(th1)
        s1 = math.sin(th1)

        wx = x - self.L4 * c1
        wy = y - self.L4 * s1
        wz = z - self.L1

        c3 = (wx * wx + wy * wy + wz * wz - self.L2 * self.L2 - self.L3 * self.L3) / (2.0 * self.L2 * self.L3)
        if c3 < -1.0001 or c3 > 1.0001:
            return None
        c3 = float(np.clip(c3, -1.0, 1.0))

        s3_abs = math.sqrt(max(0.0, 1.0 - c3 * c3))
        th1_deg = math.degrees(th1)

        for s3 in (-s3_abs, s3_abs):
            th3 = math.atan2(s3, c3)

            m1 = c3 * self.L3 + self.L2
            m2 = wz
            m3 = s3 * self.L3
            m4 = c1 * wx + s1 * wy

            c2 = m1 * m2 - m3 * m4
            s2 = -m2 * m3 - m1 * m4
            th2 = math.atan2(s2, c2)

            th2_deg = math.degrees(th2)
            th3_deg = math.degrees(th3)
            th4_deg = -(th2_deg + th3_deg) - 90.0

            if th1_deg < -120.0 or th1_deg > 120.0:
                continue
            if th2_deg < -90.0 or th2_deg > 30.0:
                continue
            if th3_deg < -150.0 or th3_deg > 0.0:
                continue

            return [th1_deg, th2_deg, th3_deg, th4_deg]

        return None

    def move_to(self, x_cm: float, y_cm: float, z_cm: float, speed: int = 70) -> list[float]:
        angles = self._calc_inv_kinematics(float(x_cm), float(y_cm), float(z_cm))
        if angles is None:
            raise ValueError(f"[REAL_ROBOT] IK fail: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self._send_joint_degrees(angles[:4], speed=speed)
        return angles

    def open_gripper(self) -> None:
        if self.connected:
            if not self._try_call("open_gripper", tuple()):
                print("[REAL_ROBOT WARN] open_gripper API를 찾지 못했습니다.")

    def close_gripper(self) -> None:
        if self.connected:
            if not self._try_call("close_gripper", tuple()):
                print("[REAL_ROBOT WARN] close_gripper API를 찾지 못했습니다.")

    def lockh(self) -> None:
        if self.connected:
            if not (self._try_call("lock_horz", tuple()) or self._try_call("lockh", tuple())):
                print("[REAL_ROBOT WARN] gripper horizontal lock API를 찾지 못했습니다.")

    def lockv(self) -> None:
        if self.connected:
            if not (self._try_call("lock_vert", tuple()) or self._try_call("lockv", tuple())):
                print("[REAL_ROBOT WARN] gripper vertical lock API를 찾지 못했습니다.")

    def unlock(self) -> None:
        if self.connected:
            self._try_call("unlock", tuple())

    def go_home(self, wait_seconds: float = 0.0) -> None:
        if self.connected:
            self._send_joint_degrees(self.HOME_DEGREES, speed=50)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    def execute_from_exec_info(self, exec_info: Dict[str, Any], speed: int = 70) -> Dict[str, Any]:
        tx, ty, tz = [float(v) for v in exec_info["target_xyz"]]
        gripper = float(exec_info["gripper_cmd"])

        angles = self.move_to(tx * 100.0, ty * 100.0, tz * 100.0, speed=speed)

        if gripper >= 0.5:
            self.close_gripper()
            gripper_state = "close"
        else:
            self.open_gripper()
            gripper_state = "open"

        real_info = {
            "target_xyz_m": [tx, ty, tz],
            "target_xyz_cm": [tx * 100.0, ty * 100.0, tz * 100.0],
            "joint_degrees": [float(v) for v in angles[:4]],
            "gripper_state": gripper_state,
        }
        print(
            f"[REAL_ROBOT] target_cm={[round(v, 2) for v in real_info['target_xyz_cm']]} | "
            f"joint_deg={[round(v, 2) for v in real_info['joint_degrees']]} | "
            f"gripper={gripper_state}"
        )
        return real_info

    def close(self) -> None:
        pass


def print_success_log(step_idx: int, exec_info: Dict[str, Any]) -> None:
    final_delta_xyz = [round(float(v), 4) for v in exec_info["final_delta_xyz"]]
    move_xyz = [round(float(v), 4) for v in exec_info["actual_move_xyz"]]
    target_xyz = [round(float(v), 4) for v in exec_info["target_xyz"]]
    gripper = float(exec_info["gripper_cmd"])
    retries = int(exec_info["retry_count"])
    print(
        f"[{step_idx:03d}] OK | final_delta={final_delta_xyz} | "
        f"move={move_xyz} | target={target_xyz} | "
        f"gripper={gripper:.1f} | retries={retries}"
    )


def print_fail_log(step_idx: int, exc: Exception) -> None:
    print(f"[{step_idx:03d}] FAIL | {exc}")


def infer_color_from_instruction(instruction: Optional[str]) -> Optional[str]:
    if not instruction:
        return None

    text = instruction.lower()
    matches = []
    for color in TARGET_COLORS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            matches.append(color)

    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 색상이 들어 있습니다: {matches} | instruction={instruction!r}")
    return matches[0] if matches else None


def resolve_target_color_and_instruction(
    instruction: Optional[str],
    target_color_arg: str,
    task_type_arg: str,
    rng: np.random.Generator,
) -> Tuple[str, str]:
    instruction_color = infer_color_from_instruction(instruction)

    if instruction_color is not None:
        target_color = instruction_color
        if target_color_arg in TARGET_COLORS and target_color_arg != instruction_color:
            raise ValueError(
                f"--instruction 색상({instruction_color})과 --target_color({target_color_arg})가 다릅니다. "
            )
    elif target_color_arg in TARGET_COLORS:
        target_color = target_color_arg
    elif target_color_arg in ("auto", "random"):
        target_color = str(rng.choice(TARGET_COLORS))
    else:
        raise ValueError(f"지원하지 않는 --target_color 값입니다: {target_color_arg}")

    if instruction is None or instruction.strip() == "":
        color_name = target_color.split('_')[0]
        desc = OBJECT_MAP[target_color]["desc"]
        
        if task_type_arg == "lift":
            instruction = f"lift the {color_name} {desc}"
        elif task_type_arg == "push":
            instruction = f"push the {color_name} {desc}"
        else:
            instruction = f"grasp the {color_name} {desc}"

    return target_color, instruction


def make_default_object_specs() -> Dict[str, Dict[str, float]]:
    x_values = np.linspace(
        DEFAULT_OBJECT_X_RANGE[0] * 0.75,
        DEFAULT_OBJECT_X_RANGE[1] * 0.75,
        len(TARGET_COLORS),
    )
    y_center = float(sum(DEFAULT_OBJECT_Y_RANGE) / 2.0)
    return {
        color: {
            "body_name": OBJECT_MAP[color]["body"],
            "x": float(x_values[idx]),
            "y": y_center,
            "yaw": 0.0,
        }
        for idx, color in enumerate(TARGET_COLORS)
    }


def sample_object_specs(
    rng: np.random.Generator,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    yaw_range: Tuple[float, float] = DEFAULT_YAW_RANGE,
    min_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    max_tries: int = 1000,
) -> Dict[str, Dict[str, float]]:
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

    specs: Dict[str, Dict[str, float]] = {}
    placed_xy = []

    placement_order = list(TARGET_COLORS)
    rng.shuffle(placement_order)

    for color in placement_order:
        for _ in range(max_tries):
            x = float(rng.uniform(x_range[0], x_range[1]))
            y = float(rng.uniform(y_range[0], y_range[1]))
            xy = np.array([x, y], dtype=np.float64)

            if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                specs[color] = {
                    "body_name": OBJECT_MAP[color]["body"],
                    "x": x,
                    "y": y,
                    "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                }
                placed_xy.append(xy)
                break
        else:
            raise RuntimeError("물체 6개를 겹치지 않게 배치하지 못했습니다.")

    return {color: specs[color] for color in TARGET_COLORS}


def reset_freejoint_body_pose(env: SyncSimRaccoonEnv, body_name: str, x: float, y: float, z: float, yaw: float) -> None:
    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise AttributeError("SyncSimRaccoonEnv에 model/data 속성이 필요합니다.")

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"body not found: {body_name}. XML 파일 확인이 필요합니다.")

    jnt_adr = int(env.model.body_jntadr[body_id])
    jnt_num = int(env.model.body_jntnum[body_id])
    if jnt_num < 1:
        raise ValueError(f"{body_name} has no joint")

    joint_id = jnt_adr
    qpos_adr = int(env.model.jnt_qposadr[joint_id])

    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    env.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

    qvel_adr = int(env.model.jnt_dofadr[joint_id])
    env.data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def reset_multicolor_scene(
    env: SyncSimRaccoonEnv,
    object_specs: Dict[str, Dict[str, float]],
    target_color: str,
) -> None:
    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

    target_spec = object_specs[target_color]

    env.reset_episode(float(target_spec["x"]), float(target_spec["y"]), float(target_spec["yaw"]))

    for color, spec in object_specs.items():
        reset_freejoint_body_pose(
            env=env,
            body_name=str(spec["body_name"]),
            x=float(spec["x"]),
            y=float(spec["y"]),
            z=0.02,
            yaw=float(spec["yaw"]),
        )

    target_body_name = str(target_spec["body_name"])
    if hasattr(env, "active_object_body_name"):
        env.active_object_body_name = target_body_name
    if hasattr(env, "target_body_name"):
        env.target_body_name = target_body_name

    mujoco.mj_forward(env.model, env.data)


def object_specs_to_meta(object_specs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    return {
        color: {
            "body_name": str(spec["body_name"]),
            "xy": [float(spec["x"]), float(spec["y"])],
            "yaw": float(spec["yaw"]),
        }
        for color, spec in object_specs.items()
    }


def write_rollout_meta(
    out_dir: Path,
    instruction: str,
    target_color: str,
    object_specs: Dict[str, Dict[str, float]],
    args: Dict[str, Any],
) -> None:
    meta = {
        "instruction": instruction,
        "target_color": target_color,
        "target_body_name": OBJECT_MAP[target_color]["body"],
        "all_object_init_poses": object_specs_to_meta(object_specs),
        "args": args,
    }
    with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def rollout(
    xml_path: str,
    server_url: str,
    instruction: Optional[str],
    unnorm_key: str,
    output_dir: str,
    episode_id: int = 1,
    max_steps: int = 1000000,
    use_viewer: bool = True,
    camera_name: str = "front_view",
    speed: int = 70,
    settle_seconds_per_action: float = 0.8,
    initial_settle_seconds: float = 0.3,
    delta_scale: float = 1.0,
    randomize_objects: bool = True,
    request_timeout: float = 60.0,
    max_delta_xyz: float = 0.005,
    target_color_arg: str = "auto",
    task_type_arg: str = "grasp",
    seed: Optional[int] = None,
    object_x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    object_y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    min_object_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    use_real_robot: bool = False,
    allow_sim_only_on_hw_fail: bool = False,
    real_initial_wait_seconds: float = 5.0,
    real_settle_seconds: Optional[float] = None,
    real_go_home_on_exit: bool = False,
) -> None:
    out_dir = Path(output_dir) / f"episode_{episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    clear_existing_images(out_dir)

    rng = np.random.default_rng(seed)
    target_color, instruction = resolve_target_color_and_instruction(
        instruction=instruction,
        target_color_arg=target_color_arg,
        task_type_arg=task_type_arg,
        rng=rng,
    )

    if randomize_objects:
        object_specs = sample_object_specs(
            rng=rng,
            x_range=object_x_range,
            y_range=object_y_range,
            min_distance=min_object_distance,
        )
    else:
        object_specs = make_default_object_specs()

    env = SyncSimRaccoonEnv(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
    )
    real_robot: Optional[RealRaccoonController] = None

   
    # ==========================================
    # [WEB CAM] 실제 웹캠 장치 초기화
    # ==========================================
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 1번 카메라 강제 고정 및 윈도우 최적화  # 웹캠이 안 잡히면 1 또는 2로 변경
    if not cap.isOpened():
        print("[WEB CAM WARN] 실제 웹캠을 열 수 없습니다! 디바이스 연결을 확인하세요.")
    else:
        print("[WEB CAM] 실제 웹캠 장치 연결 성공 (Inference에 웹캠 이미지를 사용합니다.)")

    try:
        reset_multicolor_scene(
            env=env,
            object_specs=object_specs,
            target_color=target_color,
        )

        env.lockh()
        if use_real_robot:
            real_robot = RealRaccoonController(
                require_ready=not allow_sim_only_on_hw_fail,
                home_wait_seconds=real_initial_wait_seconds,
            )

        env.debug_check_current_ee_reachable()

        if initial_settle_seconds > 0:
            env.settle_steps(seconds=initial_settle_seconds)

        write_rollout_meta(
            out_dir=out_dir,
            instruction=instruction,
            target_color=target_color,
            object_specs=object_specs,
            args={
                "xml_path": xml_path,
                "unnorm_key": unnorm_key,
                "camera_name": camera_name,
                "speed": speed,
                "settle_seconds_per_action": settle_seconds_per_action,
                "initial_settle_seconds": initial_settle_seconds,
                "delta_scale": delta_scale,
                "max_delta_xyz": max_delta_xyz,
                "seed": seed,
                "object_x_range": list(object_x_range),
                "object_y_range": list(object_y_range),
                "min_object_distance": min_object_distance,
                "use_real_robot": use_real_robot,
                "allow_sim_only_on_hw_fail": allow_sim_only_on_hw_fail,
                "real_initial_wait_seconds": real_initial_wait_seconds,
                "real_settle_seconds": real_settle_seconds,
                "real_go_home_on_exit": real_go_home_on_exit,
            },
        )

        print(
            f"[SCENE] task_type={task_type_arg!r} | instruction={instruction!r} | target_color={target_color!r} | "
            f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f})"
        )

        step_idx = 0

        while True:
            # ==========================================
            # [WEB CAM] 웹캠 프레임 가져오기 및 전처리
            # ==========================================
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # 💡 [여기 수정됨] 비율이 찌그러지지 않도록 정중앙을 정사각형(1:1)으로 자르기
                    h, w, _ = frame.shape
                    size = min(h, w)
                    y1, x1 = (h - size) // 2, (w - size) // 2
                    cropped_frame = frame[y1:y1+size, x1:x1+size]
                    
                    # 자른 이미지를 AI 모델이 요구하는 256x256 크기 및 RGB로 변환
                    frame_resized = cv2.resize(cropped_frame, (256, 256))
                    input_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    # 실시간 모니터링 창 띄우기 (이제 정사각형 창이 뜹니다!)
                    cv2.imshow("Real WebCam Feed (AI Input)", frame_resized)
                    cv2.waitKey(1)
                else:
                    print("[WEB CAM WARN] 프레임을 읽지 못해 시뮬레이션 이미지로 대체합니다.")
                    input_image = env.get_observation()["image"]
            else:
                input_image = env.get_observation()["image"]

            # AI 서버에 실제 웹캠 이미지 전송
            response = request_action(
                server_url=server_url,
                instruction=instruction,
                image_rgb=input_image,
                unnorm_key=unnorm_key,
                timeout=request_timeout,
            )
            action = response["action"]

            try:
                exec_info = env.execute_delta_action7(
                    action=action,
                    speed=speed,
                    delta_scale=delta_scale,
                    max_delta_xyz=max_delta_xyz,
                )

                if real_robot is not None and real_robot.connected:
                    exec_info["real_robot"] = real_robot.execute_from_exec_info(exec_info, speed=speed)
                    wait_seconds = settle_seconds_per_action if real_settle_seconds is None else real_settle_seconds
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)

                print_success_log(step_idx, exec_info)

                env.settle_steps(seconds=settle_seconds_per_action)
                obs = env.get_observation()

                # 결과 폴더에 웹캠 이미지 저장
                frame_name = f"frame_{step_idx:06d}.png"
                Image.fromarray(input_image).save(out_dir / frame_name)

                # =========================================================
                # [Auto-Termination] 성공 판정 로직
                # =========================================================
                target_body = OBJECT_MAP[target_color]["body"]
                body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, target_body)
                
                if body_id != -1:
                    jnt_adr = int(env.model.body_jntadr[body_id])
                    qpos_adr = int(env.model.jnt_qposadr[jnt_adr])
                    
                    current_x = env.data.qpos[qpos_adr]
                    current_y = env.data.qpos[qpos_adr + 1]
                    current_z = env.data.qpos[qpos_adr + 2]
                    
                    if task_type_arg in ["lift", "grasp"]:
                        if current_z > 0.03 and action[-1] >= 0.5:
                            print(f"\n🎉 [TASK SUCCESS] {target_color} {task_type_arg} 완료! 물체를 성공적으로 집어 올렸습니다. (높이: {current_z:.3f}m)")
                            break
                            
                    elif task_type_arg == "push":
                        init_x = object_specs[target_color]["x"]
                        init_y = object_specs[target_color]["y"]
                        moved_dist = math.hypot(current_x - init_x, current_y - init_y)
                        if moved_dist > 0.015:
                            print(f"\n🎉 [TASK SUCCESS] {target_color} {task_type_arg} 완료! 물체를 충분히 밀어냈습니다. (이동 거리: {moved_dist:.3f}m)")
                            break
                # =========================================================

            except Exception as exc:
                print_fail_log(step_idx, exc)
                obs = env.get_observation()
                frame_name = f"frame_{step_idx:06d}_skipped.png"
                Image.fromarray(input_image).save(out_dir / frame_name)

            step_idx += 1
            if step_idx >= max_steps:
                print("\n[STOP] max_steps reached. 태스크에 실패했거나 목표 도달에 너무 오래 걸렸습니다.")
                break

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")

    finally:
        if cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
        if real_robot is not None:
            if real_go_home_on_exit and real_robot.connected:
                real_robot.go_home(wait_seconds=0.0)
            real_robot.close()
        env.close()


def clear_existing_images(out_dir: Path) -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    deleted_count = 0
    for file_path in out_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            file_path.unlink()
            deleted_count += 1

    print(f"[CLEANUP] removed {deleted_count} existing image files from {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--server_url", type=str, default=None, help="Direct HTTP URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="OpenVLA prompt. If omitted, automatically generated based on task and color.",
    )
    parser.add_argument(
        "--target_color",
        type=str,
        default="auto",
        choices=["auto", "random", *TARGET_COLORS],
        help="Target color. 'auto' uses the color in --instruction, or random if instruction has no color.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="grasp",
        choices=["grasp", "lift", "push"],
        help="학습시킨 다양한 태스크(grasp, lift, push)를 테스트할 수 있습니다.",
    )
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--output_dir", type=str, default="rollout_outputs")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=70)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.8)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.3)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--max_delta_xyz", type=float, default=0.005)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument("--use_real_robot", action="store_true", help="서버 action을 실제 라쿤봇 하드웨어에도 전송합니다.")
    parser.add_argument(
        "--allow_sim_only_on_hw_fail",
        action="store_true",
        help="--use_real_robot 상태에서 하드웨어 연결 실패 시 종료하지 않고 MuJoCo만 계속합니다.",
    )
    parser.add_argument("--real_initial_wait_seconds", type=float, default=5.0, help="실제 로봇 home 이동 후 대기 시간")
    parser.add_argument(
        "--real_settle_seconds",
        type=float,
        default=None,
        help="실제 로봇 action 전송 후 대기 시간. 생략하면 --settle_seconds_per_action 값을 사용합니다.",
    )
    parser.add_argument("--real_go_home_on_exit", action="store_true", help="종료 시 실제 로봇을 home 자세로 보냅니다.")
    parser.add_argument(
        "--no_randomize_objects",
        action="store_true",
        help="Disables randomization for all objects.",
    )

    parser.add_argument("--use_ssh_tunnel", action="store_true", help="Connect to the inference server through SSH local port forwarding")
    parser.add_argument("--ssh_host", type=str, default="qlak315.iptime.org")
    parser.add_argument("--ssh_port", type=int, default=24100)
    parser.add_argument("--ssh_user", type=str, default="root")
    parser.add_argument("--ssh_password", type=str, default=None, help="Prefer OPENVLA_SSH_PASSWORD or --ssh_ask_password")
    parser.add_argument("--ssh_ask_password", action="store_true", help="Prompt for the SSH password interactively")
    parser.add_argument("--remote_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--remote_server_port", type=int, default=8000)
    parser.add_argument("--local_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--local_server_port", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)

        if tunnel is not None:
            print(
                f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> "
                f"{args.remote_server_host}:{args.remote_server_port}"
            )

        rollout(
            xml_path=args.xml_path,
            server_url=server_url,
            instruction=args.instruction,
            unnorm_key=args.unnorm_key,
            output_dir=args.output_dir,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            use_viewer=args.use_viewer,
            camera_name=args.camera_name,
            speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            randomize_objects=not args.no_randomize_objects,
            request_timeout=args.request_timeout,
            max_delta_xyz=args.max_delta_xyz,
            target_color_arg=args.target_color,
            task_type_arg=args.task_type,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
            use_real_robot=args.use_real_robot,
            allow_sim_only_on_hw_fail=args.allow_sim_only_on_hw_fail,
            real_initial_wait_seconds=args.real_initial_wait_seconds,
            real_settle_seconds=args.real_settle_seconds,
            real_go_home_on_exit=args.real_go_home_on_exit,
        )

if __name__ == "__main__":
    main()