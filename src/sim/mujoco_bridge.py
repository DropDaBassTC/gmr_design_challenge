#!/usr/bin/env python3
"""MuJoCo simulation bridge with shared scanning and motion nodes."""
from pathlib import Path

import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from motion.motion_node import MotionNode
from scanning.scanner_node import ScannerNode

MODEL_PATH = Path(__file__).parent / "models" / "scene.xml"
PHYSICS_HZ = 500.0


class MujocoBridge(Node):
    def __init__(self, model_path=MODEL_PATH, *, pan_actuators=None):
        super().__init__("mujoco_bridge")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.pan_actuators = (
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pan_motor")]
            if pan_actuators is None else pan_actuators
        )
        if any(actuator < 0 or actuator >= self.model.nu for actuator in self.pan_actuators):
            raise ValueError("pan actuator index is outside the MuJoCo model")
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.create_subscription(Float64MultiArray, "motor_cmd", self.on_cmd, 10)
        self.scanner_node = ScannerNode(self.model, self.data)
        self.motion_node = MotionNode(self.model, self.data)
        self.sim_executor = SingleThreadedExecutor()
        for node in (self, self.scanner_node, self.motion_node):
            self.sim_executor.add_node(node)
        self.create_timer(1.0 / PHYSICS_HZ, self.step)

    def on_cmd(self, msg):
        for actuator, command in zip(self.pan_actuators, msg.data):
            self.data.ctrl[actuator] = command

    def step(self):
        mujoco.mj_step(self.model, self.data)
        joint_states = JointState()
        joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_states.name = self.joint_names
        joint_states.position = self.data.qpos[: self.model.njnt].tolist()
        joint_states.velocity = self.data.qvel[: self.model.njnt].tolist()
        self.pub_js.publish(joint_states)


def main():
    rclpy.init()
    bridge = MujocoBridge()
    try:
        bridge.sim_executor.spin()
    finally:
        bridge.sim_executor.shutdown()
        bridge.scanner_node.destroy_node()
        bridge.motion_node.destroy_node()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
