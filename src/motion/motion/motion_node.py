from math import isfinite

import mujoco
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class MotionNode(Node):
    """Command any named, speed-limited carriage in the shared MuJoCo model."""

    def __init__(self, model, data, actuators=None, *, node_name="sensor_motion", motion_guard=None):
        actuator_names = actuators or {"track_speed_cmd": "track_velocity"}
        actuator_ids = {}
        limits = {}
        for topic, name in actuator_names.items():
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if actuator_id < 0:
                raise ValueError(f"MuJoCo scene is missing the '{name}' actuator")
            if not model.actuator_ctrllimited[actuator_id]:
                raise ValueError(f"actuator '{name}' must define ctrlrange limits")
            actuator_ids[topic] = actuator_id
            limits[topic] = model.actuator_ctrlrange[actuator_id].copy()

        super().__init__(node_name)
        self.model = model
        self.data = data
        self.actuator_ids = actuator_ids
        self.speed_limits = limits
        self.motion_guard = motion_guard
        self.speed_subscriptions = [
            self.create_subscription(
                Float64, topic, self._callback_for(topic), 10
            )
            for topic in actuator_ids
        ]

    def _callback_for(self, topic):
        def callback(msg):
            self._set_speed(topic, msg)

        return callback

    def _on_speed(self, msg):
        self._set_speed("track_speed_cmd", msg)

    def _set_speed(self, topic, msg):
        if not isfinite(msg.data):
            self.get_logger().warning(f"ignoring non-finite speed command on {topic}")
            return
        if self.motion_guard is not None:
            allowed, reason = self.motion_guard(topic, msg.data)
            if not allowed:
                self.get_logger().warning(f"ignoring motion command on {topic}: {reason}")
                return
        low, high = self.speed_limits[topic]
        self.data.ctrl[self.actuator_ids[topic]] = min(low if msg.data < low else msg.data, high)
