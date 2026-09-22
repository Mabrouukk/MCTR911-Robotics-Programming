import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PickPlaceStatusSubscriber(Node):
    def __init__(self):
        super().__init__("ur5e_pick_place_monitor")
        self.subscription = self.create_subscription(
            String, "/ur5e/task_status", self.on_status, 10
        )

    def on_status(self, msg: String):
        self.get_logger().info(f"[MONITOR] UR5e status: {msg.data}")


def main():
    rclpy.init()
    node = PickPlaceStatusSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
