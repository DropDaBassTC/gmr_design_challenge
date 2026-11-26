ROS 2 package for simulated FARO Focus-style scanning.

`ScannerNode(model, data)` raycasts the shared MuJoCo scene and publishes XYZ
`sensor_msgs/msg/PointCloud2` on `/points` while active. Call
`/set_scanning` (`std_srvs/srv/SetBool`) with `true` to activate and `false`
to stop. Frames during the activation interval are streamed to a disk spool
and assembled into one raw PLY at deactivation, avoiding an unbounded
in-memory capture. An orderly simulator shutdown writes an active partial
capture before closing scanner nodes; abrupt process termination is not
guaranteed to produce a valid PLY.
Each frame is transformed to world coordinates using its acquisition-time pose.
The ray directions are centered on the scanner site's local +Z axis. The four
floor scanners use the full simulated 360° × 300° dome; the overhead crane
scanner uses a 360° × 60° downward cone and raycasts the aircraft mesh only, so
the hangar roof and its convex visual hull do not occlude the airplane. Sampling
(180 × 8 rays), range (10 m), and noise (1 cm standard deviation) are temporary
placeholders, not verified Focus Core SKU settings. Construct this node with
the simulator's live model/data and a single-threaded executor.

Use `stitch_pointclouds(clouds, transforms, target_frame)` to combine captures
in a common coordinate frame. Supply one 4×4 homogeneous transform per cloud,
mapping that cloud's `header.frame_id` into the target frame. The function
transforms and concatenates XYZ float32 points; it does not estimate poses,
register overlapping scans, or remove duplicates. Get transforms from the
simulator/TF tree; do not concatenate clouds captured at different carriage
poses without their respective poses.

The MuJoCo bridge can run an automatic route that keeps each scanner active
during its simulated rail pass. This continuous-moving capture is a simulation
approximation only; the design brief's intended Focus Core workflow uses
stationary scan stations. Each rail outbound and overhead X traversal now
produces one raw file; return and reposition frames are excluded. The bridge
reads simulator route/scanner settings from root `sensors.yml`. Optical
exposure is not modeled or configurable in this simulator.
