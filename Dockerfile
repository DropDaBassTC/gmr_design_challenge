ARG ROS_DISTRO=humble
FROM gmr.jfrog.io/gmr-remote/ros:${ROS_DISTRO}-ros-base AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install \
    python3-pip \
    git \
    git-lfs \
    python3-tk \
    # pdf reading tpp_viz_tool
    evince \
    wget \
    # outlier removal tool
    python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-pcl-ros \
    # afd calibration
    iputils-ping \
    netcat \
    bc \
    # afd calibration + ubuntu 22
    netcat-traditional \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && apt-get clean && \
    rm -rf /tmp/* /var/cache/* /var/lib/apt/lists/*

ARG USERNAME=nathan
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
# Create a non-root user
RUN --mount=type=secret,id=jfrog_creds,dst=/etc/apt/auth.conf.d/jfrog.conf \
    groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \  
    # [Optional] Add sudo support for the non-root user
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc \
    && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc
ENV USER=$USERNAME
ENV DEBIAN_FRONTEND=
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Adding dependencies for platform for segmentation_interface
RUN mkdir -p /tmp/build
WORKDIR /tmp/build

# Nlohmann
RUN git clone --branch v3.11.2 --depth 1 https://github.com/nlohmann/json.git && \
    cd json && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install


# boost
RUN wget https://archives.boost.io/release/1.74.0/source/boost_1_74_0.tar.bz2 && \
    tar --bzip2 -xf boost_1_74_0.tar.bz2 && \
    cd boost_1_74_0 && \
    ./bootstrap.sh --prefix=/usr --with-toolset=gcc && \
    ./b2 install && \
    cd .. && rm -rf boost_1_74_0*

# Clean up the build directory
RUN rm -rf /tmp/build

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility,display


USER ${USERNAME}:${USERNAME}

RUN pip3 install --upgrade pip

# Install 
