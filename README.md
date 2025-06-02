# Catinator: Real-Time Cat Breed Classification Robot

## Overview

Catinator is an advanced ROS2-based robotic system that combines computer vision, machine learning, and robotic control to perform real-time cat breed classification. The system integrates a deep learning model with live camera feeds and robotic arm control to create an interactive cat detection and recognition platform.

## Project Structure

```
Catinator/
├── src/catinator/                    # ROS2 package root
│   ├── catinator/                    # Main package modules
│   │   ├── camera_capture.py         # Camera image capture utility
│   │   ├── classifier.py             # Model evaluation and testing
│   │   ├── realtime_classification.py # Real-time classification node
│   │   ├── move_arm.py               # Robotic arm controller
│   │   ├── move_robot.py             # Mobile robot controller
│   │   ├── ck_resnet50_all_0.0005_cosine.pth # Pre-trained model weights
│   │   └── captured_images/          # Image storage directory
│   ├── setup.py                      # Package configuration
│   ├── package.xml                   # ROS2 package metadata
│   └── resource/                     # Package resources
├── captured_images/                  # Global image storage
├── log/                             # System logs
├── install/                         # Build artifacts
└── .git/                           # Version control
```

## Features

- **Real-time Cat Breed Classification**: Classify 21 different cat breeds using a ResNet50-based deep learning model
- **ROS2 Integration**: Full ROS2 ecosystem compatibility with message passing and node communication
- **Robotic Arm Control**: Automated arm movements triggered by successful cat detection
- **Interactive Camera Interface**: Live video feed with clickable controls and visual feedback
- **Temporal Smoothing**: Prediction history analysis for improved accuracy
- **Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for better image quality

## Supported Cat Breeds

The system can classify the following 21 cat breeds:
- Abyssinian, American Shorthair, Birman, Bombay, British Shorthair
- Burmese, Chausie, Devon Rex, Egyptian Mau, Maine Coon
- Munchkin, Norwegian Forest, Oriental Shorthair, Persian, Ragdoll
- Russian Blue, Selkirk Rex, Siamese, Siberian, Sphynx, Toyger

## Installation

### Prerequisites

- ROS2 (Humble or later)
- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.x
- CUDA (optional, for GPU acceleration)

### Dependencies

```bash
# ROS2 dependencies
sudo apt install ros-humble-sensor-msgs ros-humble-geometry-msgs

# Python dependencies
pip install torch torchvision opencv-python pillow numpy

# ROS2 Python packages
pip install rclpy cv-bridge
```

### Build Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd Catinator
```

2. Build the ROS2 package:
```bash
colcon build --packages-select catinator
source install/setup.bash
```

3. Verify the model file exists:
```bash
ls src/catinator/catinator/ck_resnet50_all_0.0005_cosine.pth
```

## Usage

### Real-time Classification

Start the real-time cat breed classification system:

```bash
ros2 run catinator realtime_classifier
```

**Controls:**
- `q` or `ESC`: Quit application
- `f`: Toggle fullscreen mode
- `p`: Pause/resume classification
- Click the button: Toggle classification state

### Camera Capture

Capture images from the camera feed:

```bash
ros2 run catinator camera_cap --save_path ./captured_images
```

### Robotic Arm Control

Control the robotic arm independently:

```bash
ros2 run catinator move_arm
```

## Core Components

### 1. Real-time Classification System (`realtime_classification.py`)

The real-time classification system is the heart of the Catinator project, implementing a sophisticated computer vision pipeline that processes live camera feeds and performs cat breed classification.

#### Architecture Overview

The system consists of two main classes:

**CatBreedClassifier Class:**
- **Model Loading**: Automatically detects and loads ResNet50 or ConvNeXt architectures from checkpoint files
- **Image Preprocessing**: Implements a comprehensive transformation pipeline including resizing, cropping, color augmentation, and normalization
- **Prediction Engine**: Performs inference with confidence thresholding and temporal smoothing
- **Performance Optimization**: GPU acceleration support with CUDA backend

**ROS2CatBreedClassifier Node:**
- **Camera Integration**: Subscribes to `/depth_cam/rgb/image_raw` topic for live video feeds
- **Interactive Interface**: Provides real-time visual feedback with overlays and controls
- **Robotic Integration**: Triggers arm movements when high-confidence detections occur
- **Performance Monitoring**: Tracks inference times and frame rates

#### Key Technical Features

**Advanced Image Enhancement:**
```python
# CLAHE enhancement for better image quality
lab = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
enhanced = cv2.merge((cl,a,b))
```

**Temporal Smoothing Algorithm:**
The system maintains a prediction history buffer and uses majority voting to reduce classification noise:
```python
self.prediction_history.append(class_name)
if len(self.prediction_history) >= 3:
    class_name = Counter(self.prediction_history).most_common(1)[0][0]
```

**Confidence-Based Filtering:**
- Confidence threshold of 60% for valid predictions
- "Uncertain" classification for low-confidence detections
- Top-3 prediction display for comprehensive analysis

**Integration with Robotic Systems:**
- Automatic arm movement triggers on high-confidence detections
- 3-second cooldown period between movements
- Temporary classification pause during arm operations

#### Real-time Performance Optimizations

- **ROI Processing**: Focuses classification on center region of frame
- **Adaptive Prediction Intervals**: Configurable inference frequency (default: 0.5s)
- **GPU Memory Management**: Efficient tensor operations with proper device handling
- **Frame Skipping**: Processes every nth frame to maintain real-time performance

### 2. Model Evaluation System (`classifier.py`)

The classifier module implements a sophisticated zero-shot evaluation system designed for model testing and validation across different cat breed datasets.

#### Core Functionality

**Dynamic Model Architecture Detection:**
The system automatically identifies the model architecture from checkpoint files:
```python
# Auto-detect based on state dictionary keys
if 'fc.weight' in state:
    print("Detected ResNet architecture")
    num_classes = state['fc.weight'].size(0)
    model = models.resnet50(weights=None)
```

**Advanced Data Pipeline:**
- **Multi-scale Transforms**: 358px resize followed by 299px center crop for optimal model input
- **Normalization**: ImageNet-standard normalization for transfer learning compatibility
- **Batch Processing**: Efficient DataLoader implementation with configurable batch sizes

**Zero-shot Evaluation Framework:**
The system performs comprehensive model evaluation without additional training:

**Class Mapping System:**
```python
# Intelligent class mapping from training to test sets
train_root = Path("/path/to/training/data")
orig_ds = datasets.ImageFolder(train_root, transform=val_tfms)
orig_classes = orig_ds.classes
test_indices = [orig_classes.index(c) for c in dataset.classes]
```

**Dynamic Head Slicing:**
Creates subset classifiers by extracting relevant neurons from the full model:
```python
# Extract subset of classification head
new_fc = nn.Linear(in_f, num_test).to(device)
with torch.no_grad():
    new_fc.weight.copy_(model.fc.weight[test_indices])
    new_fc.bias.copy_(model.fc.bias[test_indices])
```

#### Evaluation Metrics

**Comprehensive Performance Analysis:**
- **Loss Calculation**: Cross-entropy loss across test dataset
- **Accuracy Metrics**: Per-class and overall accuracy computation
- **Per-image Predictions**: Detailed classification results for individual samples

**Statistical Reporting:**
```python
loss = running_loss / total
acc = running_corrects / total
print(f"Zero‐shot loss: {loss:.4f}, accuracy: {acc:.4%}")
```

#### Technical Implementation Details

**Memory Efficient Processing:**
- Gradient-free inference using `torch.no_grad()`
- Efficient tensor operations for large datasets
- Proper device management for GPU/CPU compatibility

**Flexible Dataset Handling:**
- Support for custom dataset paths
- Automatic class discovery from directory structure
- Compatible with ImageFolder dataset format

**Model Compatibility:**
- Supports various ResNet architectures
- Extensible design for other CNN architectures
- Checkpoint format validation and error handling

## Technical Specifications

### Model Architecture
- **Base Model**: ResNet50 with ImageNet pre-training
- **Input Size**: 299×299 RGB images
- **Output Classes**: 21 cat breeds
- **Model Size**: ~90MB (ck_resnet50_all_0.0005_cosine.pth)

### Performance Metrics
- **Inference Time**: ~50-100ms per frame (GPU)
- **Classification Accuracy**: >85% on validation set
- **Confidence Threshold**: 60% for positive classifications
- **Frame Rate**: 10-20 FPS depending on hardware

### Hardware Requirements
- **Minimum**: Intel i5/AMD Ryzen 5, 8GB RAM, integrated graphics
- **Recommended**: Intel i7/AMD Ryzen 7, 16GB RAM, dedicated GPU (GTX 1060+)
- **Camera**: ROS2-compatible RGB camera
- **Robot**: ROS2-compatible robotic platform with servo control

## Configuration

### Model Parameters
- Modify class names in `realtime_classification.py` line 425-446
- Adjust confidence threshold in `realtime_classification.py` line 146
- Change prediction interval in `realtime_classification.py` line 246

### Robotic Control
- Servo mappings configured in `move_arm.py` lines 53-60
- Arm movement sequences in `realtime_classification.py` lines 268-296
- Movement cooldown period in `realtime_classification.py` line 182

## Troubleshooting

### Common Issues

**Model Loading Errors:**
- Verify model file path and permissions
- Check PyTorch version compatibility
- Ensure sufficient memory for model loading

**Camera Connection Issues:**
- Verify ROS2 camera topic: `ros2 topic list | grep image`
- Check camera permissions and drivers
- Test with: `ros2 topic echo /depth_cam/rgb/image_raw`

**Performance Issues:**
- Monitor GPU memory usage
- Reduce batch size or image resolution
- Check system resource utilization

### Debug Mode

Enable detailed logging:
```python
# Add to realtime_classification.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow ROS2 coding standards
2. Maintain compatibility with existing interfaces
3. Add comprehensive tests for new features
4. Update documentation for significant changes

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- Built on ROS2 framework
- Utilizes PyTorch and torchvision libraries
- ResNet50 architecture from "Deep Residual Learning for Image Recognition"
- OpenCV for computer vision operations 