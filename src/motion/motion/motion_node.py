from math import isfinite

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class MotionNode(Node):
    """Apply a bounded track-speed command to the shared MuJoCo state."""

    def __init__(self, model, data):
        import mujoco

        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity"
        )
        if actuator_id < 0:
            raise ValueError("MuJoCo scene is missing the 'track_velocity' actuator")
        if not model.actuator_ctrllimited[actuator_id]:
            raise ValueError("track velocity actuator must define ctrlrange limits")
        min_speed, max_speed = model.actuator_ctrlrange[actuator_id]

        super().__init__("sensor_motion")
        self.model = model
        self.data = data
        self.actuator_id = actuator_id
        self.min_speed, self.max_speed = min_speed, max_speed
        self.subscription = self.create_subscription(
            Float64, "track_speed_cmd", self._on_speed, 10
        )

    def _on_speed(self, msg):
        if not isfinite(msg.data):
            self.get_logger().warning("ignoring non-finite track speed command")
            return
        self.data.ctrl[self.actuator_id] = min(
            self.max_speed, max(self.min_speed, msg.data)
        )
