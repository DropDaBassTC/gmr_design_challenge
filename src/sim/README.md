MuJoCo simulation bridge

Run locally from the ROS 2 Jazzy workspace after `colcon build` and sourcing
`install/setup.bash`:

    python3 src/sim/mujoco_bridge.py

Or run the containerized simulation with:

    docker compose --profile sim up --build

The bridge owns one MuJoCo model/data instance and shares it between the
scanning and motion nodes in one single-threaded ROS executor. It publishes
`/joint_states` and (only when active) `/points`. Activate/deactivate scans via
`/set_scanning` (`std_srvs/srv/SetBool`). Command carriage speed on
`/track_speed_cmd` (`std_msgs/msg/Float64`), in m/s from -0.5 to 0.5. The
existing `/motor_cmd` (`std_msgs/msg/Float64MultiArray`) remains the pan
actuator command.

The scene models three independent full-length floor rails along X with
upward-facing scanners at `Y=-24, 0, +24 m` (placeholder spacing), plus the
overhead scanner on the crane bridge/trolley. Scanner services are
`/floor_rail_1/set_scanning`, `/floor_rail_2/set_scanning`,
`/floor_rail_3/set_scanning`, and `/crane/set_scanning`; their clouds publish
on the matching `/.../points` topics. The first floor rail retains the
backwards-compatible `/set_scanning` and `/points` names. Motion commands are
`/track_speed_cmd`, `/floor_rail_2_speed_cmd`, `/floor_rail_3_speed_cmd`,
`/crane_bridge_speed_cmd`, and `/crane_trolley_speed_cmd`. The configured
default pose has the crane at X=-12 m, trolley at Y=6 m, and hook block below
the bridge as in the design brief. Scanner body dimensions, rail spacing,
travel, and crane structures are simplified placeholders, not a validated
hardware-clearance model.

By default, simulation startup runs an automatic scan route. It homes all
carriages, scans floor rails 1–3 one at a time while each carriage traverses
outbound, returns each carriage with scanning disabled, then scans the crane
along X from one side of Y. It shifts the trolley 8 m at a time with scanning
disabled, alternating X direction at each Y lane until the opposite travel
limit is reached. The 8 m lane spacing is a simulation placeholder that needs
coverage/overlap validation. The route commands the shared MuJoCo actuators directly; ROS motion
commands and manual scanner activation are locked out until the route ends.
This intentionally simulates continuous capture while moving and is not a
Focus Core operating mode. Set `auto_scan=False` when constructing
`MujocoBridge` for tests or manual ROS control.

The FARO Focus family dome is simulated as 360° × 300°. Ray density, range,
and noise remain placeholder values pending the exact Focus Core SKU. The
Hangar STL is used for scan rays and visualization; simple collision boxes
approximate its shell because MuJoCo convexifies mesh collision geometry.
