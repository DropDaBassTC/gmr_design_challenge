#!/usr/bin/env bash
set -eo pipefail

export MUJOCO_GL=egl
unset DISPLAY WAYLAND_DISPLAY
source /opt/ros/jazzy/setup.bash
[ -f /ros_ws/install/setup.bash ] && source /ros_ws/install/setup.bash
export PYTHONPATH="/ros_ws/src/sim:/ros_ws/src/scanning:/ros_ws/src/motion${PYTHONPATH:+:${PYTHONPATH}}"

for package in sim scanning motion; do
    python3 -m unittest discover -s "/ros_ws/src/${package}/test" -v
done
