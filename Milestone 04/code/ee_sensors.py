"""
End-effector sensors in the MuJoCo model: what the simulator "measures".

add_ee_sensors() puts six MuJoCo frame sensors on the flange site
(attachment_site): position, orientation, linear and angular velocity,
linear and angular acceleration. read_ee() returns their readings in the
robot base frame, so they can be compared directly with our kinematic
equations.

Two details:
  - MuJoCo's linear acceleration sensor behaves like a real accelerometer:
    it also feels gravity and reads a - g. We add g back to get a.
  - The base sits at ROBOT_BASE_POS with no rotation, so world and base
    frames differ only by that offset: positions shift, velocities and
    accelerations do not.
"""
import mujoco
import numpy as np

from kinematics import ROBOT_BASE_POS

EE_SITE = "attachment_site"
SENSORS = {
    "ee_pos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
    "ee_quat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    "ee_linvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
    "ee_angvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
    "ee_linacc": mujoco.mjtSensor.mjSENS_FRAMELINACC,
    "ee_angacc": mujoco.mjtSensor.mjSENS_FRAMEANGACC,
}


def add_ee_sensors(spec):
    """Add the six flange sensors to an MjSpec that contains the UR5e."""
    for name, kind in SENSORS.items():
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=EE_SITE)


def read_ee(model, data):
    """Sensed flange state in the robot base frame: p, R, v, w, a, alpha."""
    s = lambda name: data.sensor(name).data.copy()  # noqa: E731
    w, x, y, z = s("ee_quat")
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return {
        "p": s("ee_pos") - ROBOT_BASE_POS,
        "R": R,
        "v": s("ee_linvel"),
        "w": s("ee_angvel"),
        "a": s("ee_linacc") + model.opt.gravity,   # accelerometer reads a - g
        "alpha": s("ee_angacc"),
    }
