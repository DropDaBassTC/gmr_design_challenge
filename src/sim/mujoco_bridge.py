#!/usr/bin/env python3
"""MuJoCo simulation bridge with four shared-state scanner nodes."""
from pathlib import Path

import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from motion.motion_node import MotionNode
from scanning.scanner_node import ScannerNode
from scan_sequence import ScanSequence

MODEL_PATH = Path(__file__).parent / "models" / "scene.xml"
PHYSICS_HZ = 500.0
SCANNERS = {
    "floor_rail_1": ("floor_lidar_1", "", "lidar"),
    "floor_rail_2": ("floor_lidar_2", "floor_rail_2", "floor_lidar_2"),
    "floor_rail_3": ("floor_lidar_3", "floor_rail_3", "floor_lidar_3"),
    "crane": ("lidar", "crane", "crane_lidar"),
}
MOTION_ACTUATORS = {
    "track_speed_cmd": "track_velocity",
    "floor_rail_2_speed_cmd": "floor_rail_2_velocity",
    "floor_rail_3_speed_cmd": "floor_rail_3_velocity",
    "crane_bridge_speed_cmd": "crane_bridge_velocity",
    "crane_trolley_speed_cmd": "crane_trolley_velocity",
}


class MujocoBridge(Node):
    def __init__(self, model_path=MODEL_PATH, *, pan_actuators=None, auto_scan=True):
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
        self.scanner_nodes = {
            key: ScannerNode(
                self.model,
                self.data,
                site_name=site_name,
                node_name=f"{key}_scanner",
                topic_prefix=topic_prefix,
                frame_id=frame_id,
                activation_guard=lambda key=key: self._scanner_activation_guard(key),
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
        )
        self.sim_executor = SingleThreadedExecutor()
        for node in (self, self.motion_node, *self.scanner_nodes.values()):
            self.sim_executor.add_node(node)
        self.create_timer(1.0 / PHYSICS_HZ, self.step)

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
    rclpy.init()
    bridge = MujocoBridge()
    try:
        bridge.sim_executor.spin()
    finally:
        bridge.sim_executor.shutdown()
        for scanner in bridge.scanner_nodes.values():
            scanner.destroy_node()
        bridge.motion_node.destroy_node()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
