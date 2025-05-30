import rclpy
from rclpy.node import Node
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from servo_controller_msgs.msg import ServoStateList
import time

class ArmController(Node):

    def __init__(self):
        super().__init__('arm_controller')
        self.publisher_ = self.create_publisher(ServosPosition, 'servo_controller', 10)

        self.current_pose = {}
        self.pose_received = False

        self.create_subscription(
            ServoStateList,
            '/servo_states',
            self.servo_state_callback,
            10
        )

    def servo_state_callback(self, msg):
        for servo in msg.servo_state:
            self.current_pose[servo.id] = servo.position
        self.pose_received = True
    
    def is_default_pose(self, default_pose: dict, tolerance: int = 10):
        """
        Check if the current servo pose is close enough to the default.
        :param default_pose: Dict of {servo_id: target_position}
        :param tolerance: Allowed pulse error (e.g. ±10)
        :return: True if all match, False otherwise
        """
        for servo_id, target in default_pose.items():
            current = self.current_pose.get(servo_id, None)
            if current is None:
                self.get_logger().warn(f"No reading for servo ID {servo_id}")
                return False
            if abs(current - target) > tolerance:
                return False
            return True

        

    def move_arm(self, duration_ms=1000, **servo_positions):
        """
        Send pulse commands to multiple servos with optional duration.
        Usage:
            move_arm(duration_ms=1000, joint1=500, joint2=600, gripper=700)
        Servo IDs must match your hardware mapping.
        """
        msg = ServosPosition()
        msg.position_unit = 'pulse'
        msg.duration = duration_ms/1000.0

        # Mapping from friendly names to actual servo IDs
        servo_map = {
            'joint1': 1,
            'joint2': 2,
            'joint3': 3,
            'joint4': 4,
            'joint5': 5,
            'gripper': 10,
        }

        for name, pulse in servo_positions.items():
            if name in servo_map:
                servo = ServoPosition()
                servo.id = servo_map[name]
                servo.position = float(pulse)
                msg.position.append(servo)
            else:
                self.get_logger().warn(f"Unknown joint name: {name}")

        self.publisher_.publish(msg)
        self.get_logger().info(f"Sent command: {servo_positions} with duration {duration_ms} ms")

    def default_move(self):
        # Just one example: move to a certain pose
        self.move_arm(
            duration_ms=1500,
            joint1=500,
            joint2=500,
            joint3=500,
            joint4=500,
            joint5=500,
            gripper=750
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    time.sleep(1.0)
    
    try:
        while rclpy.ok():
            node.move_arm(duration_ms=1500, joint5=500, joint4=150, joint3=500, joint2=500, joint1=920)
            # Let the node process callbacks and sleep between moves
            rclpy.spin_once(node, timeout_sec=1.5)
    except KeyboardInterrupt:
        pass
        
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main()
    





