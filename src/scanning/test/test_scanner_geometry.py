import unittest

import mujoco
import numpy as np
import rclpy
from pathlib import Path
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import SetBool

from scanning.scanner_node import cloud_from_points, generate_directions
from scanning.scanner_node import ScannerNode

SCENE_PATH = Path(__file__).parents[2] / "sim" / "models" / "scene.xml"


class ScannerGeometryTest(unittest.TestCase):
    def test_focus_dome_covers_full_azimuth_and_300_degree_vertical_fov(self):
        directions = generate_directions(4, 3, 360.0, 300.0)
        expected = np.array(
            [
                [0.8660254, 0.0, -0.5],
                [0.0, 0.8660254, -0.5],
                [-0.8660254, 0.0, -0.5],
                [0.0, -0.8660254, -0.5],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(directions[:4], expected, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
        self.assertEqual(directions.shape, (12, 3))

    def test_point_cloud_uses_xyz_float32_and_handles_empty_returns(self):
        points = np.array([[1.25, -2.0, 3.5]], dtype=np.float32)
        msg = cloud_from_points(points, "lidar")
        self.assertEqual((msg.height, msg.width), (1, 1))
        self.assertEqual((msg.point_step, msg.row_step), (12, 12))
        self.assertEqual([(f.name, f.offset, f.datatype) for f in msg.fields], [
            ("x", 0, 7), ("y", 4, 7), ("z", 8, 7)
        ])
        self.assertEqual(len(msg.data), 12)

        empty = cloud_from_points(np.empty((0, 3), dtype=np.float32), "lidar")
        self.assertEqual((empty.height, empty.width, empty.row_step), (1, 0, 0))
        self.assertEqual(bytes(empty.data), b"")

    def test_scanner_is_a_ros_node_and_starts_inactive(self):
        rclpy.init()
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        node = ScannerNode(model, mujoco.MjData(model))
        try:
            self.assertIsInstance(node, Node)
            self.assertFalse(node.active)

            request = SetBool.Request()
            response = SetBool.Response()
            request.data = True
            node._set_active(request, response)
            self.assertTrue(node.active)
            self.assertTrue(response.success)

            request.data = False
            node._set_active(request, response)
            self.assertFalse(node.active)
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_pointcloud_publishes_only_while_scanning_is_active(self):
        rclpy.init()
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scanner = ScannerNode(
            model, data, horizontal_samples=4, vertical_channels=3,
            max_range=5.0, noise_std=0.0, scan_hz=0.01,
        )
        listener = Node("pointcloud_test_listener")
        received = []
        listener.create_subscription(PointCloud2, "points", received.append, 10)
        executor = SingleThreadedExecutor()
        executor.add_node(listener)
        try:
            executor.spin_once(timeout_sec=0.2)
            scanner.scan()
            executor.spin_once(timeout_sec=0.1)
            self.assertEqual(received, [])

            request, response = SetBool.Request(), SetBool.Response()
            request.data = True
            scanner._set_active(request, response)
            scanner.scan()
            executor.spin_once(timeout_sec=0.2)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].header.frame_id, "lidar")

            request.data = False
            scanner._set_active(request, response)
            scanner.scan()
            executor.spin_once(timeout_sec=0.1)
            self.assertEqual(len(received), 1)
        finally:
            executor.remove_node(listener)
            listener.destroy_node()
            scanner.destroy_node()
            rclpy.shutdown()

    def test_measured_cloud_stays_inside_configured_maximum_range(self):
        rclpy.init()
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scanner = ScannerNode(
            model, data, horizontal_samples=4, vertical_channels=1,
            max_range=5.0, noise_std=0.0, scan_hz=0.01,
        )
        try:
            request, response = SetBool.Request(), SetBool.Response()
            request.data = True
            scanner._set_active(request, response)
            msg = scanner.scan()
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
            self.assertTrue(np.all(np.linalg.norm(points, axis=1) <= 5.0))
        finally:
            scanner.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
