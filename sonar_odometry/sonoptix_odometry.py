#! /usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import torch

# Import our custom Base Class (adjust import path based on your package structure)
from .sonar_odometry_lib import BaseSonarOdometryNode

class SonoptixOdometryNode(BaseSonarOdometryNode):
    """
    Hardware-specific implementation for the Sonoptix Echo Sonar.
    Handles the `Image` topic and extracts the embedded range.
    """
    def __init__(self):
        super().__init__('sonoptix_odometry')

        # Sonoptix-specific parameters
        self.declare_parameter('topic_in', '/sonar/echo/data')
        topic_in = self.get_parameter('topic_in').value

        self.br = CvBridge()
        
        # Subscribe to the Sonoptix image topic
        self.sub = self.create_subscription(
            Image, 
            topic_in, 
            self.image_callback, 
            qos_profile_sensor_data
        )

        self.get_logger().info('Sonoptix Odometry Subscriber Initialized.')

    def image_callback(self, msg):
        # 1. Parse hardware-specific message into an OpenCV array
        cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = cv_img.shape

        # 2. Extract embedded meta-data (Sonoptix stores range in pixel [0,0])
        max_range_m = 12 #float(cv_img[0, 0])
        fov = 120 if max_range_m <= 30 else 90

        # 3. Clean the artifact pixel to avoid interfering with gradient calculations
        cv_img[0, 0] = 0

        # 4. Convert to PyTorch tensor
        # We ensure the tensor lives on the correct device by asking the base class estimator 
        # (Defaulting to CPU if the estimator hasn't fully initialized yet)
        device = self.estimator.device if self.estimator else torch.device('cpu')
        scan_tensor = torch.tensor(cv_img, dtype=torch.float32, device=device)

        # 5. Hand the standardized data to the Base Class for Odometry Optimization
        self.process_sonar_scan(
            scan_tensor=scan_tensor,
            max_range_m=max_range_m,
            h=h,
            w=w,
            fov_degrees=fov,
            stamp=msg.header.stamp
        )

def main(args=None):
    rclpy.init(args=args)
    node = SonoptixOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()