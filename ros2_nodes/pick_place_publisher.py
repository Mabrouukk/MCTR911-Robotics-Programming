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
] # de el cycles el 3andna el betwada7 el 7 stages bta3et el pick and place msh contoller ha2e2e

class PickPlaceStatusPublisher(Node): 
    def __init__(self):
        super().__init__("ur5e_pick_place_publisher") # hena el program el by send w ye receive el status messages
        self.publisher_ = self.create_publisher(String, "/ur5e/task_status", 10) # hena el topic el by send el status messages 3ala named channel esmaha /ur5e/task_status we el 10 el buffur
        self.step_index = 0 
        self.timer = self.create_timer(1.5, self.publish_next_step) # publish_next_step hya el function el by send el status messages every 1.5 seconds

    def publish_next_step(self): # pick el second item men el CYCLE list w send el message
        msg = String()
        msg.data = self.next_cycle_step()
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing: '{msg.data}'")

    def next_cycle_step(self): # pick up el next step men el CYCLE list w return it
        step = CYCLE[self.step_index % len(CYCLE)]
        self.step_index += 1
        return step


def main():
    rclpy.init()
    node = PickPlaceStatusPublisher()
    try:
        rclpy.spin(node) # de el function el by keep the node running w listening for messages 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
