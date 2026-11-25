import unittest
from pathlib import Path

import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64

from motion.motion_node import MotionNode

SCENE_PATH = Path(__file__).parents[2] / "sim" / "models" / "scene.xml"


class MotionSpeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.node = MotionNode(self.model, self.data)

    def tearDown(self):
        self.node.destroy_node()

    def test_speed_command_clamps_to_model_velocity_limits(self):
        actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity")

        self.node._on_speed(Float64(data=0.25))
        self.assertEqual(self.data.ctrl[actuator], 0.25)

        self.node._on_speed(Float64(data=5.0))
        self.assertEqual(self.data.ctrl[actuator], 0.5)

        self.node._on_speed(Float64(data=-5.0))
        self.assertEqual(self.data.ctrl[actuator], -0.5)

    def test_zero_speed_stops_and_nonfinite_command_is_ignored(self):
        actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity")
        self.node._on_speed(Float64(data=0.3))
        self.node._on_speed(Float64(data=0.0))
        self.assertEqual(self.data.ctrl[actuator], 0.0)

        self.node._on_speed(Float64(data=float("nan")))
        self.assertEqual(self.data.ctrl[actuator], 0.0)

    def test_speed_command_moves_the_carriage_when_mujoco_steps(self):
        track = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "track")
        qpos = self.model.jnt_qposadr[track]
        self.node._on_speed(Float64(data=0.25))
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.assertGreater(self.data.qpos[qpos], 0.05)

    def test_ros_topic_command_reaches_the_motion_node(self):
        actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity")
        source = Node("track_command_test")
        publisher = source.create_publisher(Float64, "track_speed_cmd", 10)
        executor = SingleThreadedExecutor()
        executor.add_node(source)
        executor.add_node(self.node)
        try:
            publisher.publish(Float64(data=0.4))
            for _ in range(10):
                executor.spin_once(timeout_sec=0.05)
                if self.data.ctrl[actuator] == 0.4:
                    break
            self.assertEqual(self.data.ctrl[actuator], 0.4)
        finally:
            executor.remove_node(source)
            executor.remove_node(self.node)
            source.destroy_node()

    def test_independent_rail_and_crane_speed_topics_control_separate_actuators(self):
        node = MotionNode(
            self.model,
            self.data,
            {
                "floor_rail_2_speed_cmd": "floor_rail_2_velocity",
                "crane_bridge_speed_cmd": "crane_bridge_velocity",
            },
            node_name="multi_axis_motion_test",
        )
        try:
            node._set_speed("floor_rail_2_speed_cmd", Float64(data=0.2))
            node._set_speed("crane_bridge_speed_cmd", Float64(data=-0.3))
            rail_id = node.actuator_ids["floor_rail_2_speed_cmd"]
            crane_id = node.actuator_ids["crane_bridge_speed_cmd"]
            self.assertEqual(self.data.ctrl[rail_id], 0.2)
            self.assertEqual(self.data.ctrl[crane_id], -0.3)
        finally:
            node.destroy_node()


if __name__ == "__main__":
    unittest.main()
