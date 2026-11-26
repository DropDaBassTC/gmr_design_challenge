# ROS 2 Jazzy (Ubuntu 24.04, LTS to 2029)
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_WS=/ros_ws

# Add driver/hardware packages you need here (examples: serial, lidar, ros2_control)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-yaml \
        python3-rosdep \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-tf2-ros \
        python3-serial \
        i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo (prebuilt wheel, headless rendering via EGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip libegl1 libgl1 libosmesa6 \
    && pip install --no-cache-dir --break-system-packages mujoco \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=egl

WORKDIR ${ROS_WS}
COPY src/ src/

# Resolve dependencies declared in your package.xml files, then build
RUN . /opt/ros/jazzy/setup.sh \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "/ros_ws/src/sim/mujoco_bridge.py"]