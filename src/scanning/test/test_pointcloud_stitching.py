import unittest

import numpy as np

from scanning.scanner_node import cloud_from_points, stitch_pointclouds


class PointCloudStitchingTest(unittest.TestCase):
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
