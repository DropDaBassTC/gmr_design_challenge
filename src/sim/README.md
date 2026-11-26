MuJoCo simulation bridge

Run locally from the ROS 2 Jazzy workspace after `colcon build` and sourcing
`install/setup.bash`:

    python3 src/sim/mujoco_bridge.py

Or run the containerized simulation with:

    docker compose --profile sim up --build

To open the interactive MuJoCo viewer and run the same automatic route, allow
the container to connect to your host X11 display (`DISPLAY` and a valid
`XAUTHORITY` file must be available in this shell), then run:

    xhost +si:localuser:root
    docker compose --profile viewer run --build --rm sim_viewer

When finished, stop the viewer with Ctrl-C and revoke that local display access:

    xhost -si:localuser:root

The viewer profile requires an X11 display (`DISPLAY`, X11 socket and
`XAUTHORITY`). Wayland desktops normally provide XWayland; remote/headless
sessions need an X11-forwarded or virtual display. Do not start both `sim`
and `viewer` profiles at once: they run independent MuJoCo worlds, though they
use the same simulation ROS domain.

Run the complete Python unittest suite headless (no display or viewer):

    docker compose --profile test run --build --rm test_headless

This test profile sets `MUJOCO_GL=egl`, mounts source read-only, and does not
mount a display socket or Xauthority file.

The bridge owns one MuJoCo model/data instance and shares it between the
scanning and motion nodes in one single-threaded ROS executor. It publishes
`/joint_states` and (only when active) `/points`. Activate/deactivate scans via
`/set_scanning` (`std_srvs/srv/SetBool`). Command carriage speed on
`/track_speed_cmd` (`std_msgs/msg/Float64`), in m/s from -0.5 to 0.5. The
existing `/motor_cmd` (`std_msgs/msg/Float64MultiArray`) remains the pan
actuator command.

The scene uses the IVTAA rail and carriage STLs for four independent
full-length floor rails along X with upward-facing scanners at
`Y=-27.375, -9.125, +9.125, +27.375 m`, evenly spaced across the 73 m work area.
The rail's positive-X end is its home datum; the carriages start there and
travel toward negative X. The overhead scanner remains on the crane bridge/trolley. Scanner services are
`/floor_rail_1/set_scanning`, `/floor_rail_2/set_scanning`,
`/floor_rail_3/set_scanning`, `/floor_rail_4/set_scanning`, and
`/crane/set_scanning`; their clouds publish
on the matching `/.../points` topics. The first floor rail retains the
backwards-compatible `/set_scanning` and `/points` names. Motion commands are
`/track_speed_cmd`, `/floor_rail_2_speed_cmd`, `/floor_rail_3_speed_cmd`,
`/floor_rail_4_speed_cmd`, `/crane_bridge_speed_cmd`, and
`/crane_trolley_speed_cmd`. The configured
default pose has the crane at X=-12 m, trolley at Y=6 m, and hook block below
the bridge as in the design brief. Scanner body dimensions, rail spacing,
travel, and crane structures are simplified placeholders, not a validated
hardware-clearance model.

By default, simulation startup runs an automatic scan route. It homes all
carriages, scans floor rails 1–4 one at a time while each carriage traverses
outbound, returns each carriage with scanning disabled, then scans the crane
along X from one side of Y. It shifts the trolley 8 m at a time with scanning
disabled, alternating X direction at each Y lane until the opposite travel
limit is reached. The 8 m lane spacing is a simulation placeholder that needs
coverage/overlap validation. The route commands the shared MuJoCo actuators directly; ROS motion
commands and manual scanner activation are locked out until the route ends.
This intentionally simulates continuous capture while moving and is not a
Focus Core operating mode. Set `auto_scan=False` when constructing
`MujocoBridge` for tests or manual ROS control.

Each scanner activation interval produces one binary PLY in
`output/raw_scans/` using a local timestamp
(`YYYY-MM-DD-HH-MM-SS.ffffff.ply`); colliding timestamps get a numeric suffix.
Frames are transformed with their acquisition-time scanner pose into the
MuJoCo world frame, then aggregated and written once at the interval's end.
The automatic route produces one floor-rail PLY for each complete outbound
pass and one crane PLY for each complete X traversal; return and reposition
frames are excluded. When the automatic route completes, raw captures are
concatenated into `output/mergedPointcloud.ply`. Compose mounts the project
output directory at `/ros_ws/output`. Manual runs with `auto_scan=False` write
one raw PLY on each scanner deactivation but do not automatically finalize the
merged file.

Floor scanner rays use a 360° × 300° dome. The overhead scanner uses a 360° ×
60° cone centered downward on the aircraft top, and raycasts the aircraft mesh
without the hangar visual hull or roof. The aircraft visual mesh is assigned a
MuJoCo geom group visible in the viewer and selected by the overhead ray filter. Ray density, range,
and noise remain placeholder values pending the exact Focus Core SKU. Simple
collision boxes approximate the hangar shell because MuJoCo convexifies mesh
collision geometry.

Simulator settings live in root `sensors.yml`, bind-mounted read-only to
`/ros_ws/sensors.yml`. Supported values and units are `scan_sequence.speed_mps`
(m/s; scene actuators cap speed at ±0.5 m/s), `scan_sequence.crane_y_step_m`
(m), `scanner.horizontal_samples` (rays around the dome),
`scanner.vertical_channels` (rays vertically), `scanner.max_range_m` (m),
`scanner.range_noise_std_m` (m), and `scanner.scan_hz` (Hz). Optical exposure
is neither modeled nor configurable; real exposure requires a verified
hardware driver/SKU interface.

Raw PLY captures are written once per scanner activation interval, not once
per published frame. Manual `set_scanning` activation/deactivation likewise
opens and closes one capture. Each frame is transformed using its
acquisition-time scanner pose before aggregation.
