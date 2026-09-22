import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CYCLE = [
    "Waiting for part on conveyor",
    "Part detected - computing pick pose",
    "Moving to pick pose",
    "Gripper closing on part",
    "Moving to place pose",
    "Gripper opening - part released",
    "Returning to home pose",
]

class PickPlaceStatusPublisher(Node):
    def __init__(self):
        super().__init__("ur5e_pick_place_publisher")
        self.publisher_ = self.create_publisher(String, "/ur5e/task_status", 10)
        self.step_index = 0
        self.timer = self.create_timer(1.5, self.publish_next_step)

    def publish_next_step(self):
        msg = String()
        msg.data = self.next_cycle_step()
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing: '{msg.data}'")

    def next_cycle_step(self):
        step = CYCLE[self.step_index % len(CYCLE)]
        self.step_index += 1
        return step


def main():
    rclpy.init()
    node = PickPlaceStatusPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
