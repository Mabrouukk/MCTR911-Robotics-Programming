"""
Import panda_kinematics.trajectory_node on a machine without ROS 2.

trajectory_node.py imports rclpy and the message types at the top (that part
of the template must not change). The simulator and the verify script only
need its maths and its Q_START / Q_END, so when ROS 2 is missing we put
empty stand-in modules in place before importing it.
"""
import os
import sys
import types

PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panda_kinematics")


def load_trajectory_node():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        stubs = {
            "rclpy": {}, "rclpy.node": {"Node": object},
            "sensor_msgs": {}, "sensor_msgs.msg": {"JointState": object},
            "std_msgs": {}, "std_msgs.msg": {"Float64MultiArray": object},
        }
        for name, attrs in stubs.items():
            module = sys.modules.setdefault(name, types.ModuleType(name))
            for key, value in attrs.items():
                setattr(module, key, value)
    if PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)
    from panda_kinematics import trajectory_node
    return trajectory_node
