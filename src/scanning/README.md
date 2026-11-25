ROS 2 package for simulated FARO Focus-style scanning.

`ScannerNode(model, data)` raycasts the shared MuJoCo scene and publishes XYZ
`sensor_msgs/msg/PointCloud2` on `/points` while active. Call
`/set_scanning` (`std_srvs/srv/SetBool`) with `true` to activate and `false`
to stop. The simulated dome is 360° × 300°; sampling (180 × 8 rays), range
(10 m), and noise (1 cm standard deviation) are temporary placeholders, not
verified Focus Core SKU settings. Construct this node with the simulator's
live model/data and a single-threaded executor.
