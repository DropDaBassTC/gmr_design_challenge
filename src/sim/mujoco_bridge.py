#!/usr/bin/env python3
"""MuJoCo simulation bridge with shared-state scanner nodes."""
import math
from pathlib import Path

import mujoco
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from motion.motion_node import MotionNode
from scanning.scanner_node import PointCloudRecorder, ScannerNode
from scan_sequence import ScanSequence

MODEL_PATH = Path(__file__).parent / "models" / "scene.xml"
PHYSICS_HZ = 500.0
CONFIG_PATH = Path(__file__).resolve().parents[2] / "sensors.yml"
SCANNERS = {
    "floor_rail_1": ("floor_lidar_1", "", "lidar"),
    "floor_rail_2": ("floor_lidar_2", "floor_rail_2", "floor_lidar_2"),
    "floor_rail_3": ("floor_lidar_3", "floor_rail_3", "floor_lidar_3"),
    "floor_rail_4": ("floor_lidar_4", "floor_rail_4", "floor_lidar_4"),
    "crane": ("lidar", "crane", "crane_lidar"),
}
MOTION_ACTUATORS = {
    "track_speed_cmd": "track_velocity",
    "floor_rail_2_speed_cmd": "floor_rail_2_velocity",
    "floor_rail_3_speed_cmd": "floor_rail_3_velocity",
    "floor_rail_4_speed_cmd": "floor_rail_4_velocity",
    "crane_bridge_speed_cmd": "crane_bridge_velocity",
    "crane_trolley_speed_cmd": "crane_trolley_velocity",
}


class MujocoBridge(Node):
    def __init__(
        self, model_path=MODEL_PATH, *, pan_actuators=None, auto_scan=True,
        output_dir=None, config_path=CONFIG_PATH,
    ):
        config = _load_sensor_config(config_path)
        super().__init__("mujoco_bridge")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if pan_actuators is None:
            pan_actuator = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pan_motor"
            )
            pan_actuators = [] if pan_actuator < 0 else [pan_actuator]
        self.pan_actuators = pan_actuators
        if any(actuator < 0 or actuator >= self.model.nu for actuator in self.pan_actuators):
            raise ValueError("pan actuator index is outside the MuJoCo model")
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.create_subscription(Float64MultiArray, "motor_cmd", self.on_cmd, 10)
        output_dir = output_dir or Path(__file__).resolve().parents[2] / "output"
        self.recorder = PointCloudRecorder(output_dir)
        self.scanner_nodes = {
            key: ScannerNode(
                self.model,
                self.data,
                site_name=site_name,
                node_name=f"{key}_scanner",
                topic_prefix=topic_prefix,
                frame_id=frame_id,
                activation_guard=lambda key=key: self._scanner_activation_guard(key),
                on_scan=self._record_scan,
                on_capture_change=self._capture_change,
                **config["scanner"],
            )
            for key, (site_name, topic_prefix, frame_id) in SCANNERS.items()
        }
        self.motion_node = MotionNode(
            self.model,
            self.data,
            MOTION_ACTUATORS,
            node_name="multi_axis_motion",
            motion_guard=self._motion_command_guard,
        )
        self.scan_sequence = ScanSequence(
            self.model, self.data, self.scanner_nodes, self.motion_node,
            enabled=auto_scan,
            on_complete=self._finalize_scans if auto_scan else None,
            speed=config["scan_sequence"]["speed_mps"],
            crane_y_step=config["scan_sequence"]["crane_y_step_m"],
            on_capture_start=lambda _name: self.recorder.start_capture(),
            on_capture_finish=lambda _name: self.recorder.finish_capture(),
        )
        self.sim_executor = SingleThreadedExecutor()
        for node in (self, self.motion_node, *self.scanner_nodes.values()):
            self.sim_executor.add_node(node)
        self.create_timer(1.0 / PHYSICS_HZ, self.step)
        self._shutdown = False

    def shutdown(self):
        if self._shutdown:
            return
        if self.recorder.capture_active:
            self.recorder.finish_capture()
        self._shutdown = True
        self.sim_executor.shutdown()
        for scanner in self.scanner_nodes.values():
            scanner.destroy_node()
        self.motion_node.destroy_node()
        self.destroy_node()

    def _record_scan(self, cloud, origin, rotation):
        self.recorder.record(cloud, origin, rotation)

    def _capture_change(self, active):
        if self.scan_sequence.running:
            return
        if active:
            self.recorder.start_capture()
        else:
            self.recorder.finish_capture()

    def _finalize_scans(self):
        path = self.recorder.finalize()
        self.get_logger().info(f"Merged point cloud saved to {path}")

    def _scanner_activation_guard(self, requested_scanner):
        if self.scan_sequence.running and requested_scanner != self.scan_sequence.authorized_scanner:
            return False, "automatic scan sequence owns scanner activation"
        if any(
            abs(self.data.ctrl[actuator]) > 1e-6
            for actuator in (
                *self.motion_node.actuator_ids.values(), *self.pan_actuators
            )
        ):
            return False, "all rail and crane motion commands must be zero before scanning"
        if any(abs(self.data.qvel[self.model.jnt_dofadr[joint]]) > 1e-3 for joint in range(self.model.njnt)):
            return False, "all rail and crane axes must be stationary before scanning"
        if any(
            scanner.active
            for name, scanner in self.scanner_nodes.items()
            if name != requested_scanner
        ):
            return False, "another scanner is already active"
        return True, "scanner activation allowed"

    def _motion_command_guard(self, topic, speed):
        if self.scan_sequence.running:
            return False, "automatic scan sequence owns all rail and crane motion"
        if any(scanner.active for scanner in self.scanner_nodes.values()) and abs(speed) > 1e-6:
            return False, "motion commands are disabled during a stationary scan"
        return True, "motion command allowed"

    def on_cmd(self, msg):
        if self.scan_sequence.running or any(scanner.active for scanner in self.scanner_nodes.values()):
            self.get_logger().warning("ignoring legacy motor command during automatic scanning")
            return
        for actuator, command in zip(self.pan_actuators, msg.data):
            self.data.ctrl[actuator] = command

    def step(self):
        self.scan_sequence.step()
        mujoco.mj_step(self.model, self.data)
        joint_states = JointState()
        joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_states.name = self.joint_names
        joint_states.position = [self.data.qpos[self.model.jnt_qposadr[i]] for i in range(self.model.njnt)]
        joint_states.velocity = [self.data.qvel[self.model.jnt_dofadr[i]] for i in range(self.model.njnt)]
        self.pub_js.publish(joint_states)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the MuJoCo scanner simulation")
    parser.add_argument(
        "--viewer", action="store_true",
        help="open the interactive MuJoCo window (requires a graphical display)",
    )
    args = parser.parse_args()

    rclpy.init()
    bridge = MujocoBridge()
    try:
        if args.viewer:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
                _run_viewer(bridge, viewer)
        else:
            bridge.sim_executor.spin()
    finally:
        bridge.shutdown()
        rclpy.shutdown()


def _run_viewer(bridge, viewer):
    while viewer.is_running():
        bridge.sim_executor.spin_once(timeout_sec=0.001)
        with viewer.lock():
            viewer.sync()


def _load_sensor_config(path):
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"cannot load sensor config {path}: {exc}") from exc

    keys = {
        "scan_sequence": {"speed_mps", "crane_y_step_m"},
        "scanner": {
            "horizontal_samples", "vertical_channels", "max_range_m",
            "range_noise_std_m", "scan_hz",
        },
    }
    if not isinstance(config, dict) or set(config) != set(keys):
        raise ValueError(f"sensor config {path} must contain scan_sequence and scanner mappings")
    for section, section_keys in keys.items():
        if not isinstance(config[section], dict) or set(config[section]) != section_keys:
            raise ValueError(f"sensor config {path} section '{section}' has missing or unknown keys")
    for key in ("horizontal_samples", "vertical_channels"):
        value = config["scanner"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"sensor config {path}: scanner.{key} must be a positive integer")
    numeric_settings = (
        ("scan_sequence", ("speed_mps", "crane_y_step_m")),
        ("scanner", ("max_range_m", "range_noise_std_m", "scan_hz")),
    )
    for section, names in numeric_settings:
        for key in names:
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"sensor config {path}: {section}.{key} must be a finite number")
    if any(config["scan_sequence"][key] <= 0 for key in keys["scan_sequence"]):
        raise ValueError(f"sensor config {path}: scan sequence settings must be positive")
    scanner = config["scanner"]
    if (
        scanner["max_range_m"] <= 0
        or scanner["scan_hz"] <= 0
        or scanner["range_noise_std_m"] < 0
    ):
        raise ValueError(f"sensor config {path}: scanner range/rate must be positive and noise nonnegative")
    return {
        "scan_sequence": dict(config["scan_sequence"]),
        "scanner": {
            "horizontal_samples": config["scanner"]["horizontal_samples"],
            "vertical_channels": config["scanner"]["vertical_channels"],
            "max_range": config["scanner"]["max_range_m"],
            "noise_std": config["scanner"]["range_noise_std_m"],
            "scan_hz": config["scanner"]["scan_hz"],
        },
    }


if __name__ == "__main__":
    main()
