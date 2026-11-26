ROS 2 package for simulated sensor-carriage motion.

`MotionNode(model, data)` shares the simulator's MuJoCo state and subscribes to
`/track_speed_cmd` (`std_msgs/msg/Float64`). The command is a target carriage
speed in m/s, clamped to the scene's actuator range of -0.5 to 0.5 m/s. Send
zero to stop. The simulation owner must keep stepping MuJoCo and should create
this node with the same model/data used by the scanner.

When constructed by the simulation bridge, the node also consumes
`/floor_rail_2_speed_cmd`, `/floor_rail_3_speed_cmd`,
`/floor_rail_4_speed_cmd`,
`/crane_bridge_speed_cmd`, and `/crane_trolley_speed_cmd`, each controlling its
own range-limited actuator. `/track_speed_cmd` remains the first floor rail's
backwards-compatible command.
