import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')
        self.publisher_ = self.create_publisher(Twist, '/controller/cmd_vel', 10)

    def move_in_circle(self, linear_speed, angular_speed, duration):
        twist = Twist()
        twist.linear.x = linear_speed  # Forward
        twist.angular.z = angular_speed  # Turning rate    	
        self.get_logger().info(f"Moving in a circle: linear={linear_speed} m/s, angular={angular_speed} rad/s for {duration} sec")
        end_time = time.time() + duration
        while time.time() < end_time:
            self.publisher_.publish(twist)
            time.sleep(0.1)  # Publish at 10 Hz

    def stop(self):
    	twist = Twist()  # All zeros = stop everything
    	self.get_logger().info("Sending stop commands to straighten wheels...")
    	for _ in range(10):  # Publish 10 times over 1 second
        	self.publisher_.publish(twist)
        	time.sleep(0.1)
    	self.get_logger().info("Robot fully stopped and wheels straightened.")


def main(args=None):
    rclpy.init(args=args)
    mover = RobotMover()
    mover.move_in_circle(0.1, 0.0, 2) # Straighten the robot legs
    # mover.move_in_circle(0.1, 0.5, 3)  # gentle forward-left curve for 4 sec
    # mover.move_in_circle(-0.1, 0.0, 1) 
    # mover.move_in_circle(0.1, -0.3, 3) 
    # mover.move_in_circle(-0.1, 0.0, 1) 
    mover.stop()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main()




