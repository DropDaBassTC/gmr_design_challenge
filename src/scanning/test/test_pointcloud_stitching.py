import unittest
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from sensor_msgs.msg import PointCloud2

from scanning.scanner_node import (
    PointCloudRecorder,
    cloud_from_points,
    stitch_pointclouds,
)


class PointCloudStitchingTest(unittest.TestCase):
    def test_recorder_writes_one_cloud_per_capture_not_per_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PointCloudRecorder(directory)
            recorder.start_capture()
            recorder.record(
                cloud_from_points([[1.0, 0.0, 0.0]], "lidar"),
                np.zeros(3), np.eye(3),
            )
            recorder.record(
                cloud_from_points([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], "lidar"),
                np.array([4.0, 0.0, 0.0]), np.eye(3),
            )
            self.assertEqual(list(recorder.raw_dir.glob("*.ply")), [])
            spool = list(recorder.raw_dir.glob("*.capture.tmp"))
            self.assertEqual(len(spool), 1)
            self.assertEqual(spool[0].stat().st_size, 36)

            path = recorder.finish_capture()

            self.assertEqual(len(list(recorder.raw_dir.glob("*.ply"))), 1)
            self.assertEqual(path, recorder.raw_paths[0])
            header, payload = path.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"element vertex 3\n", header)
            np.testing.assert_allclose(
                np.frombuffer(payload, dtype="<f4").reshape(-1, 3),
                [[1, 0, 0], [6, 0, 0], [7, 0, 0]],
            )

    def test_recorder_refuses_to_finalize_an_open_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PointCloudRecorder(directory)
            recorder.start_capture()
            with self.assertRaisesRegex(RuntimeError, "capture is active"):
                recorder.finalize()

    def test_recorder_saves_timestamped_raw_cloud_and_merged_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 11, 25, 9, 5, 7)
            recorder = PointCloudRecorder(directory, clock=lambda: now)
            cloud = cloud_from_points([[1.25, -2.0, 3.5]], "lidar")

            recorder.record(cloud, np.zeros(3), np.eye(3))
            recorder.finalize()
            raw_path = Path(directory) / "raw_scans" / "2026-11-25-09-05-07.000000.ply"
            merged_path = Path(directory) / "mergedPointcloud.ply"

            self.assertTrue(raw_path.is_file())
            self.assertTrue(merged_path.is_file())
            self.assertEqual(raw_path.read_bytes(), merged_path.read_bytes())
            data = merged_path.read_bytes()
            header, points = data.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 1\n", header)
            np.testing.assert_allclose(np.frombuffer(points, dtype="<f4"), [1.25, -2.0, 3.5])

    def test_recorder_accumulates_scanner_frames_in_world_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PointCloudRecorder(directory)
            first = cloud_from_points([[1.0, 0.0, 0.0]], "first_lidar")
            second = cloud_from_points([[1.0, 0.0, 0.0]], "second_lidar")
            transform = np.eye(4)
            transform[1, 3] = 4.0

            recorder.record(first, np.zeros(3), np.eye(3))
            recorder.record(second, transform[:3, 3], transform[:3, :3])
            recorder.finalize()

            data = (Path(directory) / "mergedPointcloud.ply").read_bytes()
            header, points = data.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 2\n", header)
            np.testing.assert_allclose(
                np.frombuffer(points, dtype="<f4").reshape(-1, 3),
                [[1, 0, 0], [1, 4, 0]],
            )

    def test_recorder_disambiguates_same_timestamp_without_losing_scans(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 11, 25, 9, 5, 7)
            recorder = PointCloudRecorder(directory, clock=lambda: now)
            cloud = cloud_from_points([[0.0, 0.0, 0.0]], "lidar")

            first = recorder.record(cloud, np.zeros(3), np.eye(3))
            second = recorder.record(cloud, np.zeros(3), np.eye(3))

            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

    def test_recorder_writes_separate_captures_with_one_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 11, 25, 9, 5, 7)
            recorder = PointCloudRecorder(directory, clock=lambda: now)
            recorder.start_capture()
            recorder.record(cloud_from_points([[1, 0, 0]], "lidar"), np.zeros(3), np.eye(3))
            first = recorder.finish_capture()
            recorder.start_capture()
            recorder.record(cloud_from_points([[2, 0, 0]], "lidar"), np.zeros(3), np.eye(3))
            second = recorder.finish_capture()

            self.assertNotEqual(first, second)
            self.assertEqual(len(list(recorder.raw_dir.glob("*.ply"))), 2)

    def test_transforms_clouds_into_target_frame_before_concatenating(self):
        first = cloud_from_points([[1.0, 0.0, 0.0]], "lidar")
        second = cloud_from_points([[1.0, 0.0, 0.0]], "lidar")
        identity = np.eye(4)
        translated_and_rotated = np.array(
            [
                [0.0, -1.0, 0.0, 2.0],
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        stitched = stitch_pointclouds(
            [first, second], [identity, translated_and_rotated], "map"
        )
        points = np.frombuffer(stitched.data, dtype=np.float32).reshape(-1, 3)

        self.assertEqual(stitched.header.frame_id, "map")
        self.assertEqual(stitched.width, 2)
        np.testing.assert_allclose(points, [[1, 0, 0], [2, 4, 0]], atol=1e-6)

    def test_rejects_mismatched_cloud_and_transform_counts(self):
        cloud = cloud_from_points([[0.0, 0.0, 0.0]], "lidar")
        with self.assertRaisesRegex(ValueError, "same number"):
            stitch_pointclouds([cloud], [], "map")

    def test_empty_input_returns_an_empty_cloud_in_the_target_frame(self):
        stitched = stitch_pointclouds([], [], "map")
        self.assertEqual(stitched.header.frame_id, "map")
        self.assertEqual(stitched.width, 0)
        self.assertEqual(bytes(stitched.data), b"")

    def test_rejects_missing_source_frame_and_invalid_transform(self):
        cloud = cloud_from_points([[0.0, 0.0, 0.0]], "")
        with self.assertRaisesRegex(ValueError, "source frame"):
            stitch_pointclouds([cloud], [np.eye(4)], "map")

        cloud.header.frame_id = "lidar"
        with self.assertRaisesRegex(ValueError, "finite 4x4"):
            stitch_pointclouds([cloud], [np.eye(3)], "map")


if __name__ == "__main__":
    unittest.main()
