import unittest
import tempfile
from contextlib import nullcontext
from pathlib import Path

import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool

from mujoco_bridge import MujocoBridge, _run_viewer

SCENE_PATH = Path(__file__).parents[1] / "models" / "scene.xml"


class BridgeNodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.output_dir = tempfile.TemporaryDirectory()
        self.bridge = MujocoBridge(
            model_path=str(SCENE_PATH), auto_scan=False,
            output_dir=self.output_dir.name,
        )

    def tearDown(self):
        self.bridge.sim_executor.shutdown()
        for scanner in self.bridge.scanner_nodes.values():
            scanner.destroy_node()
        self.bridge.motion_node.destroy_node()
        self.bridge.destroy_node()
        self.output_dir.cleanup()

    def test_scanner_and_motion_share_bridge_model_and_data(self):
        self.assertEqual(len(self.bridge.scanner_nodes), 5)
        self.assertEqual(len(self.bridge.sim_executor.get_nodes()), 7)
        for scanner in self.bridge.scanner_nodes.values():
            self.assertIs(scanner.model, self.bridge.model)
            self.assertIs(scanner.data, self.bridge.data)
        self.assertIs(self.bridge.motion_node.model, self.bridge.model)
        self.assertIs(self.bridge.motion_node.data, self.bridge.data)
        self.assertTrue(self.bridge.recorder.raw_dir.is_dir())
        self.assertEqual(
            set(self.bridge.motion_node.actuator_ids),
            {
                "track_speed_cmd",
                "floor_rail_2_speed_cmd",
                "floor_rail_3_speed_cmd",
                "floor_rail_4_speed_cmd",
                "crane_bridge_speed_cmd",
                "crane_trolley_speed_cmd",
            },
        )

    def test_yaml_config_controls_scan_rate_density_range_noise_and_route_speed(self):
        config = Path(self.output_dir.name) / "sensors.yml"
        config.write_text(
            "scan_sequence:\n"
            "  speed_mps: 0.2\n"
            "  crane_y_step_m: 4.0\n"
            "scanner:\n"
            "  horizontal_samples: 24\n"
            "  vertical_channels: 3\n"
            "  max_range_m: 18.0\n"
            "  range_noise_std_m: 0.025\n"
            "  scan_hz: 5.0\n",
            encoding="utf-8",
        )
        bridge = MujocoBridge(
            model_path=str(SCENE_PATH), auto_scan=False,
            output_dir=self.output_dir.name, config_path=config,
        )
        try:
            self.assertEqual(bridge.scan_sequence.speed, 0.2)
            self.assertEqual(bridge.scan_sequence.crane_y_step, 4.0)
            for scanner in bridge.scanner_nodes.values():
                self.assertEqual(len(scanner.directions), 72)
                self.assertEqual(scanner.max_range, 18.0)
                self.assertEqual(scanner.noise_std, 0.025)
                self.assertEqual(scanner.timer.timer_period_ns, 200_000_000)
        finally:
            bridge.sim_executor.shutdown()
            for scanner in bridge.scanner_nodes.values():
                scanner.destroy_node()
            bridge.motion_node.destroy_node()
            bridge.destroy_node()

    def test_yaml_config_rejects_duplicate_keys_and_invalid_values(self):
        config = Path(self.output_dir.name) / "sensors.yml"
        config.write_text(
            "scan_sequence:\n  speed_mps: 0.5\n  speed_mps: 0.2\n"
            "  crane_y_step_m: 8.0\n"
            "scanner:\n  horizontal_samples: 180\n  vertical_channels: 8\n"
            "  max_range_m: 10.0\n  range_noise_std_m: 0.01\n  scan_hz: 10.0\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate YAML key"):
            MujocoBridge(model_path=str(SCENE_PATH), output_dir=self.output_dir.name, config_path=config)

        config.write_text(
            "scan_sequence:\n  speed_mps: .nan\n  crane_y_step_m: 8.0\n"
            "scanner:\n  horizontal_samples: 180\n  vertical_channels: 8\n"
            "  max_range_m: 10.0\n  range_noise_std_m: 0.01\n  scan_hz: 10.0\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "finite number"):
            MujocoBridge(model_path=str(SCENE_PATH), output_dir=self.output_dir.name, config_path=config)

        for invalid_yaml, message in (("", "must contain"), ("!!python/object/apply:os.system ['true']", "could not determine a constructor")):
            config.write_text(invalid_yaml, encoding="utf-8")
            with self.subTest(invalid_yaml=invalid_yaml), self.assertRaisesRegex(ValueError, message):
                MujocoBridge(model_path=SCENE_PATH, output_dir=self.output_dir.name, config_path=config)

        with self.assertRaisesRegex(ValueError, "cannot load sensor config"):
            MujocoBridge(
                model_path=SCENE_PATH, output_dir=self.output_dir.name,
                config_path=config.with_name("missing.yml"),
            )

    def test_automatic_route_writes_one_file_per_floor_pass_and_excludes_return_frames(self):
        sequence = self.bridge.scan_sequence
        sequence.running = True
        for index in range(4):
            scanner_name = f"floor_rail_{index + 1}"
            sequence.floor_index = index
            sequence.phase = "floor_scan"
            sequence._begin_floor_scan()
            self.bridge.step()
            scanner = self.bridge.scanner_nodes[scanner_name]
            frames = [scanner.scan(), scanner.scan()]
            self.assertEqual(len(self.bridge.recorder.raw_paths), index)

            joint_name = "track" if index == 0 else scanner_name
            joint = mujoco.mj_name2id(self.bridge.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos = self.bridge.model.jnt_qposadr[joint]
            self.bridge.data.qpos[qpos] = self.bridge.model.jnt_range[joint][1]
            self.bridge.data.qvel[self.bridge.model.jnt_dofadr[joint]] = 0.0
            mujoco.mj_forward(self.bridge.model, self.bridge.data)
            sequence.step()

            self.assertEqual(sequence.phase, "floor_return")
            self.assertEqual(len(self.bridge.recorder.raw_paths), index + 1)
            header, payload = self.bridge.recorder.raw_paths[-1].read_bytes().split(b"end_header\n", 1)
            point_count = sum(frame.width for frame in frames)
            self.assertIn(f"element vertex {point_count}\n".encode(), header)
            self.assertEqual(len(payload), point_count * 12)

        sequence.running = False
        self.bridge.step()
        self.assertEqual(len(self.bridge.recorder.raw_paths), 4)

    def test_bridge_shutdown_persists_partial_manual_capture(self):
        scanner = self.bridge.scanner_nodes["floor_rail_1"]
        request, response = SetBool.Request(), SetBool.Response()
        request.data = True
        scanner._set_active(request, response)
        frame = scanner.scan()

        self.bridge.shutdown()

        self.assertEqual(len(self.bridge.recorder.raw_paths), 1)
        header, payload = self.bridge.recorder.raw_paths[0].read_bytes().split(b"end_header\n", 1)
        self.assertIn(f"element vertex {frame.width}\n".encode(), header)
        self.assertEqual(len(payload), frame.width * 12)

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
        self.assertLess(self.bridge.data.site_xpos[site][0] - initial_sensor_x, -0.05)
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
            "/floor_rail_4/points",
        })
        self.assertEqual(service_names, {
            "/set_scanning", "floor_rail_2/set_scanning",
            "floor_rail_3/set_scanning", "crane/set_scanning",
            "floor_rail_4/set_scanning",
        })

    def test_each_scanner_publishes_its_own_lidar_frame(self):
        for name, scanner in self.bridge.scanner_nodes.items():
            previous_count = len(self.bridge.recorder.raw_paths)
            self.assertIsNone(scanner.scan())
            request, response = SetBool.Request(), SetBool.Response()
            request.data = True
            scanner._set_active(request, response)
            msg = scanner.scan()
            second_msg = scanner.scan()
            self.assertIsInstance(msg, PointCloud2)
            self.assertIsInstance(second_msg, PointCloud2)
            expected = "lidar" if name == "floor_rail_1" else (
                "crane_lidar" if name == "crane" else f"floor_lidar_{name[-1]}"
            )
            self.assertEqual(msg.header.frame_id, expected)
            self.assertEqual(len(self.bridge.recorder.raw_paths), previous_count)
            request.data = False
            scanner._set_active(request, response)
            self.assertEqual(len(self.bridge.recorder.raw_paths), previous_count + 1)
            data = self.bridge.recorder.raw_paths[-1].read_bytes()
            header, points = data.split(b"end_header\n", 1)
            self.assertIn(f"element vertex {msg.width + second_msg.width}\n".encode(), header)
            self.assertEqual(len(points), (msg.width + second_msg.width) * 12)

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

    def test_viewer_loop_spins_ros_and_syncs_until_window_closes(self):
        class Viewer:
            def __init__(self):
                self.frames = 0

            def is_running(self):
                return self.frames < 3

            def lock(self):
                return nullcontext()

            def sync(self):
                self.frames += 1

        viewer = Viewer()
        spins = []
        self.bridge.sim_executor.spin_once = lambda timeout_sec: spins.append(timeout_sec)

        _run_viewer(self.bridge, viewer)

        self.assertEqual(len(spins), 3)
        self.assertTrue(all(timeout == 0.001 for timeout in spins))
        self.assertEqual(viewer.frames, 3)


if __name__ == "__main__":
    unittest.main()
