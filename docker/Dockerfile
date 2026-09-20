FROM osrf/ros:humble-desktop

# System deps for MuJoCo (GL/GLFW for the viewer) + Python tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir mujoco numpy

# Open-source robot models (UR5e, etc.)
RUN git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /opt/mujoco_menagerie

WORKDIR /workspace

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["bash"]
