import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import SetBool


def generate_directions(horizontal_samples, vertical_channels, horizontal_fov, vertical_fov):
    azimuth = np.radians(
        np.linspace(-horizontal_fov / 2, horizontal_fov / 2, horizontal_samples, endpoint=False)
    )
    elevation = np.radians(np.linspace(-vertical_fov / 2, vertical_fov / 2, vertical_channels))
    azimuth, elevation = np.meshgrid(azimuth, elevation)
    return np.stack(
        [np.cos(elevation) * np.cos(azimuth),
         np.cos(elevation) * np.sin(azimuth),
         np.sin(elevation)], axis=-1,
    ).reshape(-1, 3)


def cloud_from_points(points, frame_id):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.height, msg.width = 1, points.shape[0]
    msg.fields = [
        PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate("xyz")
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * points.shape[0]
    msg.is_dense = True
    msg.data = points.tobytes()
    return msg


def stitch_pointclouds(clouds, transforms, target_frame):
    """Transform XYZ float32 clouds into target_frame and concatenate them.

    Each 4x4 matrix maps points from its cloud's frame into target_frame.
    This applies known poses; it does not estimate registration.
    """
    if len(clouds) != len(transforms):
        raise ValueError("clouds and transforms must have the same number of items")
    if not target_frame:
        raise ValueError("target_frame must not be empty")

    stitched = []
    for cloud, transform in zip(clouds, transforms):
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("each transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("each transform must be homogeneous")
        if not cloud.header.frame_id:
            raise ValueError("each cloud must have a source frame")
        if not target_frame:
            raise ValueError("target_frame must not be empty")

        fields = {field.name: field for field in cloud.fields}
        if any(
            name not in fields
            or fields[name].datatype != PointField.FLOAT32
            or fields[name].count != 1
            for name in "xyz"
        ):
            raise ValueError("clouds must have scalar float32 x, y, and z fields")
        offsets = [fields[name].offset for name in "xyz"]
        if min(offsets) < 0 or max(offsets) + 4 > cloud.point_step:
            raise ValueError("XYZ fields must fit inside point_step")
        if cloud.row_step < cloud.width * cloud.point_step:
            raise ValueError("row_step is smaller than the point data")
        if len(cloud.data) < cloud.height * cloud.row_step:
            raise ValueError("cloud data is shorter than its declared dimensions")

        byte_order = ">" if cloud.is_bigendian else "<"
        dtype = np.dtype({
            "names": list("xyz"),
            "formats": [byte_order + "f4"] * 3,
            "offsets": offsets,
            "itemsize": cloud.point_step,
        })
        records = np.ndarray(
            (cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
            strides=(cloud.row_step, cloud.point_step),
        )
        points = np.column_stack([records[name].reshape(-1) for name in "xyz"])
        transformed = points @ matrix[:3, :3].T + matrix[:3, 3]
        stitched.append(transformed.astype(np.float32, copy=False))

    points = np.concatenate(stitched, axis=0) if stitched else np.empty((0, 3), dtype=np.float32)
    return cloud_from_points(points, target_frame)


class ScannerNode(Node):
    """Raycast shared MuJoCo state only while activated by SetBool service."""

    def __init__(
        self, model, data, *, horizontal_samples=180, vertical_channels=8,
        max_range=10.0, noise_std=0.01, scan_hz=10.0,
    ):
        import mujoco

        if horizontal_samples < 1 or vertical_channels < 1 or max_range <= 0 or noise_std < 0 or scan_hz <= 0:
            raise ValueError("sample counts, range, rate must be positive and noise nonnegative")
        if not all(math.isfinite(value) for value in (max_range, noise_std, scan_hz)):
            raise ValueError("scan range, rate, and noise must be finite")
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        if site_id < 0:
            raise ValueError("MuJoCo scene is missing the 'lidar' site")

        super().__init__("faro_scanner")
        self.model = model
        self.data = data
        self.mujoco = mujoco
        self.site_id = site_id

        self.directions = generate_directions(horizontal_samples, vertical_channels, 360.0, 300.0)
        self.max_range = max_range
        self.noise_std = noise_std
        self.active = False
        self.publisher = self.create_publisher(PointCloud2, "points", 10)
        self.service = self.create_service(SetBool, "set_scanning", self._set_active)
        self.timer = self.create_timer(1.0 / scan_hz, self.scan)
        self.get_logger().info(
            "FARO Focus simulation: 360x300 degree dome; sampling/range/noise are placeholders"
        )

    def _set_active(self, request, response):
        self.active = request.data
        response.success = True
        response.message = "scanning activated" if self.active else "scanning deactivated"
        return response

    def scan(self):
        if not self.active:
            return None
        origin = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        world_directions = self.directions @ rotation.T
        geomid = np.zeros(1, dtype=np.int32)
        points = []
        for local, world in zip(self.directions, world_directions):
            distance = self.mujoco.mj_ray(
                self.model, self.data, origin, world, None, 1,
                int(self.model.site_bodyid[self.site_id]), geomid,
            )
            if 0.0 <= distance <= self.max_range:
                measured_distance = max(0.0, distance + np.random.normal(0.0, self.noise_std))
                if measured_distance <= self.max_range:
                    points.append(local * measured_distance)
        msg = cloud_from_points(points, "lidar")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)
        return msg
