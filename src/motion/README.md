ROS 2 package for simulated sensor-carriage motion.

`MotionNode(model, data)` shares the simulator's MuJoCo state and subscribes to
`/track_speed_cmd` (`std_msgs/msg/Float64`). The command is a target carriage
speed in m/s, clamped to the scene's actuator range of -0.5 to 0.5 m/s. Send
zero to stop. The simulation owner must keep stepping MuJoCo and should create
this node with the same model/data used by the scanner.
