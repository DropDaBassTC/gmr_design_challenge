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

The FARO Focus family dome is simulated as 360° × 300°. Ray density, range,
and noise remain placeholder values pending the exact Focus Core SKU. The
Hangar STL is used for scan rays and visualization; simple collision boxes
approximate its shell because MuJoCo convexifies mesh collision geometry.
