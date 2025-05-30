import sys
import os
import signal
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import argparse


class ROS2CamCapture(Node):
    def __init__(self, name, save_path):
        super().__init__(name)
        # ROS2 subscription
        self.cam_subscription = self.create_subscription(
            Image, 
            '/depth_cam/rgb/image_raw', 
            self.image_callback, 
            1
        )
        self.cv_bridge = CvBridge()
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize variables for UI
        self.capture_counter = 0
        self.button_pressed = False
        self.frame = None
        self.button_width = 200
        self.button_height = 50
        self.button_x = 0  # Will be set when first frame is received
        self.button_y = 0  # Will be set when first frame is received
        self.button_color = (0, 255, 0)  # Green
        self.button_text = "Capture"
        self.window_name = "ROS2 Camera Capture"
        
        # Set up window and mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Log setup complete
        self.get_logger().info("Camera capture node initialized. Press 'q' to quit or click the 'Capture' button to save an image.")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within button area
            if (self.button_x <= x <= self.button_x + self.button_width and 
                self.button_y <= y <= self.button_y + self.button_height):
                self.button_pressed = True
    
    def image_callback(self, msg):
        # Convert ROS image message to OpenCV image
        image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.frame = image_bgr.copy()  # Store original bgr frame for saving
        
        # Set button position if not set yet
        if self.button_x == 0:
            height, width = image_rgb.shape[:2]
            self.button_x = (width - self.button_width) // 2
            self.button_y = height - self.button_height - 20
        
        # Add button to the display frame
        display_frame = image_rgb.copy()
        cv2.rectangle(display_frame, 
                     (self.button_x, self.button_y), 
                     (self.button_x + self.button_width, self.button_y + self.button_height), 
                     self.button_color, -1)
        cv2.putText(display_frame, self.button_text, 
                   (self.button_x + 50, self.button_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Display the frame
        cv2.imshow(self.window_name, display_frame)
        
        # Save the frame if button was pressed
        if self.button_pressed:
            # Generate random numbers for filename
            random_num1 = np.random.randint(1000, 9999)
            random_num2 = np.random.randint(1000, 9999)
            filename = os.path.join(self.save_path, 
                                   f'image_{random_num1}_{random_num2}_{self.capture_counter}.jpg')
            cv2.imwrite(filename, self.frame)  # Save original BGR image
            self.get_logger().info(f"Image saved as {filename}")
            self.capture_counter += 1
            self.button_pressed = False
        
        # Check for key press to exit
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            # Signal the main loop to exit
            self.get_logger().info("Exit key pressed, shutting down...")
            rclpy.shutdown()


def main(args=None):
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print('Exiting ROS2 camera application...')
        cv2.destroyAllWindows()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Save images from ROS2 camera topic')
    parser.add_argument('--save_path', type=str, 
                        default='/home/ubuntu/robowalle/src/catinator/catinator/captured_images',
                        help='Directory to save captured images')
    parsed_args, _ = parser.parse_known_args(args=args)
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create and spin node
        node = ROS2CamCapture("ros2_camera_capture", parsed_args.save_path)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting...")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
