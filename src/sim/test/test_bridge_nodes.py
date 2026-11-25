import unittest
from pathlib import Path

import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
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
        self.bridge = MujocoBridge(model_path=str(SCENE_PATH), auto_scan=False)

    def tearDown(self):
        self.bridge.sim_executor.shutdown()
        for scanner in self.bridge.scanner_nodes.values():
            scanner.destroy_node()
        self.bridge.motion_node.destroy_node()
        self.bridge.destroy_node()

    def test_scanner_and_motion_share_bridge_model_and_data(self):
        self.assertEqual(len(self.bridge.scanner_nodes), 4)
        self.assertEqual(len(self.bridge.sim_executor.get_nodes()), 6)
        for scanner in self.bridge.scanner_nodes.values():
            self.assertIs(scanner.model, self.bridge.model)
            self.assertIs(scanner.data, self.bridge.data)
        self.assertIs(self.bridge.motion_node.model, self.bridge.model)
        self.assertIs(self.bridge.motion_node.data, self.bridge.data)
        self.assertEqual(
            set(self.bridge.motion_node.actuator_ids),
            {
                "track_speed_cmd",
                "floor_rail_2_speed_cmd",
                "floor_rail_3_speed_cmd",
                "crane_bridge_speed_cmd",
                "crane_trolley_speed_cmd",
            },
        )

    def test_track_command_moves_the_lidar_pose_used_by_scanner(self):
        joint = mujoco.mj_name2id(self.bridge.model, mujoco.mjtObj.mjOBJ_JOINT, "track")
        qpos = self.bridge.model.jnt_qposadr[joint]
        actuator = mujoco.mj_name2id(
            self.bridge.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity"
        )
        site = self.bridge.scanner_nodes["floor_rail_1"].site_id
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
        self.bridge.motion_node._on_speed(Float64(data=0.0))
        for _ in range(1000):
            self.bridge.step()
            if abs(self.bridge.data.qvel[self.bridge.model.jnt_dofadr[joint]]) < 1e-3:
                break
        request, response = SetBool.Request(), SetBool.Response()
        request.data = True
        scanner = self.bridge.scanner_nodes["floor_rail_1"]
        scanner._set_active(request, response)
        cloud = scanner.scan()
        self.assertIsNotNone(cloud)
        self.assertEqual(cloud.header.frame_id, "lidar")

    def test_scanners_have_unique_topics_and_services(self):
        topic_names = {
            scanner.publisher.topic_name for scanner in self.bridge.scanner_nodes.values()
        }
        service_names = {
            scanner.service.srv_name for scanner in self.bridge.scanner_nodes.values()
        }
        self.assertEqual(topic_names, {
            "/points", "/floor_rail_2/points",
            "/floor_rail_3/points", "/crane/points",
        })
        self.assertEqual(service_names, {
            "/set_scanning", "floor_rail_2/set_scanning",
            "floor_rail_3/set_scanning", "crane/set_scanning",
        })

    def test_each_scanner_publishes_its_own_lidar_frame(self):
        for name, scanner in self.bridge.scanner_nodes.items():
            self.assertIsNone(scanner.scan())
            request, response = SetBool.Request(), SetBool.Response()
            request.data = True
            scanner._set_active(request, response)
            msg = scanner.scan()
            self.assertIsInstance(msg, PointCloud2)
            expected = "lidar" if name == "floor_rail_1" else (
                "crane_lidar" if name == "crane" else f"floor_lidar_{name[-1]}"
            )
            self.assertEqual(msg.header.frame_id, expected)
            request.data = False
            scanner._set_active(request, response)

    def test_only_one_scanner_can_be_active_at_a_time(self):
        floor = self.bridge.scanner_nodes["floor_rail_1"]
        crane = self.bridge.scanner_nodes["crane"]
        floor_request, floor_response = SetBool.Request(), SetBool.Response()
        crane_request, crane_response = SetBool.Request(), SetBool.Response()
        floor_request.data = True
        crane_request.data = True

        self.bridge.data.qvel[self.bridge.model.jnt_dofadr[0]] = 0.2
        self.assertFalse(floor._set_active(floor_request, floor_response).success)
        self.assertFalse(floor.active)
        self.bridge.data.qvel[self.bridge.model.jnt_dofadr[0]] = 0.0
        floor._set_active(floor_request, floor_response)
        bridge_actuator = self.bridge.motion_node.actuator_ids["crane_bridge_speed_cmd"]
        self.bridge.motion_node._set_speed("crane_bridge_speed_cmd", Float64(data=0.3))
        self.assertEqual(self.bridge.data.ctrl[bridge_actuator], 0.0)
        crane._set_active(crane_request, crane_response)

        self.assertTrue(floor_response.success)
        self.assertFalse(crane_response.success)
        self.assertTrue(floor.active)
        self.assertFalse(crane.active)

        floor_request.data = False
        floor._set_active(floor_request, floor_response)
        crane._set_active(crane_request, crane_response)
        self.assertTrue(crane_response.success)
        self.assertTrue(crane.active)

    def test_legacy_pan_command_does_not_overwrite_track_speed(self):
        track_actuator = mujoco.mj_name2id(
            self.bridge.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity"
        )
        pan_actuator = self.bridge.pan_actuators[0]
        self.bridge.motion_node._on_speed(Float64(data=0.3))
        self.bridge.on_cmd(Float64MultiArray(data=[0.1, 0.0]))
        self.assertEqual(self.bridge.data.ctrl[track_actuator], 0.3)
        self.assertEqual(self.bridge.data.ctrl[pan_actuator], 0.1)

    def test_legacy_pan_command_is_blocked_during_scanning(self):
        scanner = self.bridge.scanner_nodes["crane"]
        request, response = SetBool.Request(), SetBool.Response()
        request.data = True
        scanner._set_active(request, response)
        pan_actuator = self.bridge.pan_actuators[0]

        self.bridge.on_cmd(Float64MultiArray(data=[0.4]))

        self.assertEqual(self.bridge.data.ctrl[pan_actuator], 0.0)

    def test_automatic_sequence_owns_scanner_and_motion_commands(self):
        scanner = self.bridge.scanner_nodes["floor_rail_1"]
        self.bridge.scan_sequence.running = True
        request, response = SetBool.Request(), SetBool.Response()
        request.data = True

        self.assertFalse(scanner._set_active(request, response).success)
        self.bridge.motion_node._on_speed(Float64(data=0.25))
        actuator = self.bridge.motion_node.actuator_ids["track_speed_cmd"]
        self.assertEqual(self.bridge.data.ctrl[actuator], 0.0)


if __name__ == "__main__":
    unittest.main()
