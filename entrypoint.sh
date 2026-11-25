#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
[ -f /ros_ws/install/setup.bash ] && source /ros_ws/install/setup.bash
exec "$@"
