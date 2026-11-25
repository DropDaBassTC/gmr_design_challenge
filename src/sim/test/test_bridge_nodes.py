import unittest
from pathlib import Path

import mujoco
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool

from mujoco_bridge import MujocoBridge

SCENE_PATH = Path(__file__).parents[1] / "models" / "scene.xml"


class BridgeNodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.bridge = MujocoBridge(model_path=str(SCENE_PATH))

    def tearDown(self):
        self.bridge.sim_executor.shutdown()
        self.bridge.scanner_node.destroy_node()
        self.bridge.motion_node.destroy_node()
        self.bridge.destroy_node()

    def test_scanner_and_motion_share_bridge_model_and_data(self):
        self.assertIs(self.bridge.scanner_node.model, self.bridge.model)
        self.assertIs(self.bridge.scanner_node.data, self.bridge.data)
        self.assertIs(self.bridge.motion_node.model, self.bridge.model)
        self.assertIs(self.bridge.motion_node.data, self.bridge.data)
        self.assertEqual(len(self.bridge.sim_executor.get_nodes()), 3)

    def test_track_command_moves_the_lidar_pose_used_by_scanner(self):
        joint = mujoco.mj_name2id(self.bridge.model, mujoco.mjtObj.mjOBJ_JOINT, "track")
        qpos = self.bridge.model.jnt_qposadr[joint]
        actuator = mujoco.mj_name2id(
            self.bridge.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity"
        )
        site = self.bridge.scanner_node.site_id
        initial_sensor_x = self.bridge.data.site_xpos[site][0]
        source = Node("bridge_integration_command")
        publisher = source.create_publisher(Float64, "track_speed_cmd", 10)
        self.bridge.sim_executor.add_node(source)
        try:
            publisher.publish(Float64(data=0.25))
            for _ in range(10):
                self.bridge.sim_executor.spin_once(timeout_sec=0.05)
                if self.bridge.data.ctrl[actuator] == 0.25:
                    break
            self.assertEqual(self.bridge.data.ctrl[actuator], 0.25)
        finally:
            self.bridge.sim_executor.remove_node(source)
            source.destroy_node()
        for _ in range(250):
            self.bridge.step()
        self.assertGreater(self.bridge.data.qpos[qpos], 0.05)
        self.assertGreater(self.bridge.data.site_xpos[site][0] - initial_sensor_x, 0.05)
        request, response = SetBool.Request(), SetBool.Response()
        request.data = True
        self.bridge.scanner_node._set_active(request, response)
        cloud = self.bridge.scanner_node.scan()
        self.assertIsNotNone(cloud)
        self.assertEqual(cloud.header.frame_id, "lidar")

    def test_legacy_pan_command_does_not_overwrite_track_speed(self):
        track_actuator = mujoco.mj_name2id(
            self.bridge.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity"
        )
        self.bridge.motion_node._on_speed(Float64(data=0.3))
        self.bridge.on_cmd(Float64MultiArray(data=[0.1, 0.0]))
        self.assertEqual(self.bridge.data.ctrl[track_actuator], 0.3)


if __name__ == "__main__":
    unittest.main()
