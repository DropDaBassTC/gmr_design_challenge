ROS 2 package for simulated FARO Focus-style scanning.

`ScannerNode(model, data)` raycasts the shared MuJoCo scene and publishes XYZ
`sensor_msgs/msg/PointCloud2` on `/points` while active. Call
`/set_scanning` (`std_srvs/srv/SetBool`) with `true` to activate and `false`
to stop. The simulated dome is 360° × 300°; sampling (180 × 8 rays), range
(10 m), and noise (1 cm standard deviation) are temporary placeholders, not
verified Focus Core SKU settings. Construct this node with the simulator's
live model/data and a single-threaded executor.

Use `stitch_pointclouds(clouds, transforms, target_frame)` to combine captures
in a common coordinate frame. Supply one 4×4 homogeneous transform per cloud,
mapping that cloud's `header.frame_id` into the target frame. The function
transforms and concatenates XYZ float32 points; it does not estimate poses,
register overlapping scans, or remove duplicates. Get transforms from the
simulator/TF tree; do not concatenate clouds captured at different carriage
poses without their respective poses.
