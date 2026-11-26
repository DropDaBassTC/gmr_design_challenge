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
    def test_directions_are_centered_on_local_positive_z_axis(self):
        directions = generate_directions(4, 3, 360.0, 300.0, center_axis="z")
        np.testing.assert_allclose(directions[6], [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
        self.assertEqual(directions.shape, (12, 3))

    def test_crane_rays_hit_aircraft_not_hangar_visual_hull(self):
        rclpy.init()
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        crane_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crane_bridge_x")
        crane_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crane_trolley_y")
        data.qpos[model.jnt_qposadr[crane_x]] = -20.0
        data.qpos[model.jnt_qposadr[crane_y]] = -10.0
        mujoco.mj_forward(model, data)
        scanner = ScannerNode(
            model, data, horizontal_samples=72, vertical_channels=8,
            vertical_fov=60.0, max_range=10.0, noise_std=0.0,
            ray_geomgroup=[1, 0, 0, 0, 0, 0], center_axis="z",
            site_name="lidar", node_name="crane_ray_target_test",
        )
        try:
            hits = set()
            origin = data.site_xpos[scanner.site_id]
            rotation = data.site_xmat[scanner.site_id].reshape(3, 3)
            geom_id = np.zeros(1, dtype=np.int32)
            for direction in scanner.directions:
                distance = mujoco.mj_ray(
                    model, data, origin, direction @ rotation.T,
                    scanner.ray_geomgroup, 1,
                    int(model.site_bodyid[scanner.site_id]), geom_id,
                )
                if 0.0 <= distance <= scanner.max_range:
                    hits.add(mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])
                    ))
            self.assertEqual(hits, {"input_mesh_visual"})
            self.assertNotIn("hangar_visual", hits)
            self.assertNotIn("hangar_roof_y_pos", hits)
            self.assertNotIn("hangar_roof_y_neg", hits)
        finally:
            scanner.destroy_node()
            rclpy.shutdown()

    def test_scanner_rejects_invalid_vertical_fov(self):
        rclpy.init()
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        try:
            with self.assertRaisesRegex(ValueError, "vertical FOV"):
                ScannerNode(model, mujoco.MjData(model), vertical_fov=0.0)
        finally:
            rclpy.shutdown()

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
