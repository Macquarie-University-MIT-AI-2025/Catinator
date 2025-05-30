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
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage
import time
from catinator.move_arm import ArmController  # Import the ArmController class

class CatBreedClassifier:
    def __init__(self, model_path, class_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # ===================================================================
        # IMPORTANT: This is where the model is loaded from the checkpoint
        # The model architecture is automatically detected and initialized
        # ===================================================================
        self.model = self.load_model(model_path)
        
        # Define image transforms with augmentation for better real-world performance
        self.transform = transforms.Compose([
            transforms.Resize(358),
            transforms.CenterCrop(299),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Add color augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225]),
        ])
        
        # Load class names
        self.class_names = class_names
        
        # Initialize prediction history for temporal smoothing
        self.prediction_history = []
        self.history_size = 5

    def load_model(self, model_path):
        # Load the model state dictionary from the .pth file
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Print top-level keys for debugging
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        # Extract state dictionary
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
                print("Found model_state_dict key")
            elif 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
                print("Found state_dict key")
            else:
                state = checkpoint
                print("Using checkpoint directly as state_dict")
        else:
            state = checkpoint
            print("Checkpoint is not a dict, using directly")
        
        # Print inner keys for debugging
        print("State dict keys:", list(state.keys())[:10], "..." if len(state.keys()) > 10 else "")
        
        # Auto-detect model architecture
        if 'fc.weight' in state:
            print("Detected ResNet architecture")
            num_classes = state['fc.weight'].size(0)
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'classifier.2.weight' in state:
            print("Detected ConvNeXt architecture")
            num_classes = state['classifier.2.weight'].size(0)
            model = models.convnext_base(weights=None)
            model.classifier = nn.Sequential(
                nn.LayerNorm2d(model.classifier[0].normalized_shape),
                nn.Flatten(1),
                nn.Linear(model.classifier[2].in_features, num_classes)
            )
        else:
            classifier_patterns = ['head.weight', 'classifier.weight', 'linear.weight', 'output.weight', 'fc.bias']
            found_key = None
            for pattern in classifier_patterns:
                for key in state.keys():
                    if pattern in key:
                        found_key = key
                        break
                if found_key:
                    break
            
            if found_key:
                print(f"Found classifier pattern: {found_key}")
                num_classes = state[found_key].size(0)
                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                raise ValueError("Could not determine model architecture from state dictionary.")
        
        # Load weights into model
        try:
            model.load_state_dict(state)
            print(f"Model loaded successfully, num_classes={num_classes}")
        except Exception as e:
            print(f"Warning: Error loading state dict: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(state, strict=False)
            print("Model loaded with strict=False")
        
        # Finalize model setup
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, cv2_image):
        """Predict the cat breed from a CV2 image"""
        # Enhance image quality
        lab = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert to RGB and continue with pipeline
        cv2_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv2_rgb)
        
        # Apply transformations
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Model inference
        with torch.no_grad():
            start_time = time.time()
            output = self.model(x)
            inference_time = time.time() - start_time
            
            # Process output
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(prob, 1)
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(prob[0], 3)
            top3_results = []
            for i in range(3):
                idx = top3_indices[i].item()
                class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
                top3_results.append({
                    "class_name": class_name,
                    "confidence": top3_prob[i].item()
                })
        
        # Get prediction with confidence threshold
        predicted_class_idx = prediction.item()
        confidence_val = confidence.item()
        
        if confidence_val < 0.6:  # Confidence threshold
            class_name = "Uncertain"
        else:
            class_name = self.class_names[predicted_class_idx] if self.class_names else f"Class {predicted_class_idx}"
            
            # Add to prediction history for temporal smoothing
            self.prediction_history.append(class_name)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            # Use most common prediction from history
            from collections import Counter
            if len(self.prediction_history) >= 3:  # Only use history after collecting enough predictions
                class_name = Counter(self.prediction_history).most_common(1)[0][0]
        
        return {
            "class_name": class_name,
            "confidence": confidence_val,
            "inference_time": inference_time,
            "top3": top3_results
        }

class ROS2CatBreedClassifier(Node):
    def __init__(self, name, model_path, class_names=None):
        super().__init__(name)
        # ROS2 subscription
        self.cam_subscription = self.create_subscription(
            Image, 
            '/depth_cam/rgb/image_raw', 
            self.image_callback, 
            1
        )
        self.cv_bridge = CvBridge()
        
        # Initialize classifier
        self.classifier = CatBreedClassifier(model_path, class_names)
        
        # Initialize arm controller
        self.arm_controller = ArmController()
        self.last_arm_movement = 0
        self.arm_movement_cooldown = 3.0  # Wait 3 seconds between movements
        self.temporary_pause_until = 0  # Timestamp until when classification should be paused
        
        # Initialize display variables
        self.window_name = "ROS2 Cat Breed Classifier"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        self.result_text = "Waiting for detection..."
        self.confidence = 0.0
        self.inference_time = 0.0
        self.top3_results = []
        self.fullscreen = False
        self.pause_classification = False
        self.frame_count = 0
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5
        
        # Set up keyboard handler
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Add button for pausing/resuming classification
        self.button_width = 200
        self.button_height = 50
        self.button_x = 20
        self.button_y = 20
        
        # Log setup complete
        self.get_logger().info(f"Cat breed classifier initialized using model: {model_path}")
        self.get_logger().info(f"Available classes: {class_names}")
        self.get_logger().info("Press 'q' to quit, 'f' to toggle fullscreen, 'p' to pause/resume classification")
    
    def perform_arm_movement(self):
        """Perform a small greeting movement with the arm"""
        current_time = time.time()
        if current_time - self.last_arm_movement >= self.arm_movement_cooldown:
            self.get_logger().info("Performing arm movement")
            
            # Set temporary pause
            self.temporary_pause_until = time.time() + 5.0  # Pause for 5 seconds
            self.get_logger().info("Pausing classification for 5 seconds")
            
            # Move to initial position
            self.arm_controller.move_arm(
                duration_ms=1000,
                joint1=920,  # Center position
                joint2=500,  # Slightly raised
                joint3=500,  # Neutral
                joint4=150,  # Neutral
                joint5=500   # Neutral
            )
            time.sleep(1.0)
            
            # Wave motion
            self.arm_controller.move_arm(
                duration_ms=1000,
                joint1=700,  # Move right
                joint2=350,
                joint3=450,
                joint4=500,
                joint5=700  # Wave "hand"
            )
            time.sleep(1.0)
            
            # Return to neutral
            self.arm_controller.move_arm(
                duration_ms=1000,
                joint1=920,
                joint2=500,
                joint3=500,
                joint4=150,
                joint5=500
            )
            
            self.last_arm_movement = current_time
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.button_x <= x <= self.button_x + self.button_width and 
                self.button_y <= y <= self.button_y + self.button_height):
                self.pause_classification = not self.pause_classification
    
    def image_callback(self, msg):
        # Convert ROS image message to OpenCV image
        image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Get frame dimensions
        height, width = image_bgr.shape[:2]
        
        # Draw ROI rectangle in center of frame
        roi_size = min(height, width) // 2
        x1 = (width - roi_size) // 2
        y1 = (height - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Prepare display frame
        display_frame = image_bgr.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Check if temporary pause is active
        current_time = time.time()
        is_temp_paused = current_time < self.temporary_pause_until
        
        # Draw pause/resume button
        button_color = (0, 0, 255) if (self.pause_classification or is_temp_paused) else (0, 255, 0)
        button_text = "Paused (5s)" if is_temp_paused else ("Classification Paused" if self.pause_classification else "Classification Active")
        cv2.rectangle(display_frame, 
                     (self.button_x, self.button_y), 
                     (self.button_x + self.button_width, self.button_y + self.button_height), 
                     button_color, -1)
        cv2.putText(display_frame, button_text, 
                   (self.button_x + 10, self.button_y + self.button_height//2 + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Run prediction if not paused and temporary pause is not active
        if (not self.pause_classification and not is_temp_paused and 
            current_time - self.last_prediction_time >= self.prediction_interval):
            
            roi_frame = image_bgr[y1:y2, x1:x2]
            
            if roi_frame.size > 0:
                try:
                    result = self.classifier.predict(roi_frame)
                    self.result_text = result["class_name"]
                    self.confidence = result["confidence"]
                    self.inference_time = result["inference_time"]
                    self.top3_results = result["top3"]
                    self.last_prediction_time = current_time
                    
                    # Trigger arm movement if confidence is high enough
                    if self.confidence >= 0.6 and self.result_text != "Uncertain":
                        self.perform_arm_movement()
                        
                except Exception as e:
                    self.get_logger().error(f"Error during prediction: {e}")
        
        # Add overlay and text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, height-180), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, f"Top Prediction: {self.result_text}", 
                   (30, height-140), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Confidence: {self.confidence:.2%}", 
                   (30, height-100), font, 0.4, (255, 255, 255), 1)
        
        if self.top3_results:
            for i, result in enumerate(self.top3_results):
                cv2.putText(display_frame, f"{i+1}. {result['class_name']}: {result['confidence']:.2%}", 
                           (width//2 + 30, height-140+i*30), font, 0.4, (255, 255, 255), 1)
        
        cv2.putText(display_frame, f"Time: {self.inference_time*1000:.0f}ms", 
                   (width-300, height-40), font, 0.4, (255, 255, 255), 1)
        
        self.frame_count += 1
        cv2.putText(display_frame, f"Frame: {self.frame_count}", 
                   (width-300, 30), font, 0.3, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info("Exit key pressed, shutting down...")
            rclpy.shutdown()
        elif key == ord('f'):
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 1280, 720)
        elif key == ord('p'):
            self.pause_classification = not self.pause_classification

def main(args=None):
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print('Exiting cat classification application...')
        cv2.destroyAllWindows()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Real-time cat breed classification using ROS2 camera')
    parser.add_argument('--model', type=str, 
                       default="/home/ubuntu/robowalle/src/catinator/catinator/ck_resnet50_all_0.0005_cosine.pth",
                       help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--class_file', type=str,
                       help='Path to a text file with class names, one per line')
    parsed_args, _ = parser.parse_known_args(args=args)
    
    # Load class names if provided
    class_names = [
        "abyssinian",
        "americanshorthair",
        "birman",
        "bombay",
        "britishshorthair",
        "burmese",
        "chausie",
        "devonrex",
        "egyptianmau",
        "mainecoon",
        "munchkin",
        "norwegianforest",
        "orientalshorthair",
        "persian",
        "ragdoll",
        "russianblue",
        "selkirkrex",
        "siamese",
        "siberian",
        "sphynx",
        "toyger"
    ]

    if parsed_args.class_file and os.path.exists(parsed_args.class_file):
        with open(parsed_args.class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create and spin node
        node = ROS2CatBreedClassifier(
            "ros2_cat_classifier", 
            parsed_args.model, 
            class_names
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting...")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 