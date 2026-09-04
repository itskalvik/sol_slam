#! /usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np

from .sol_slam_ros import SOLSLAMROSWrapper
from .sol_slam_core import SonarFrame, unwarp_sonar, get_best_device

class SonoptixEchoSLAMNode(SOLSLAMROSWrapper):
    """Hardware-specific frontend for Sonoptix Echo."""
    def __init__(self):
        super().__init__('sonoptix_echo_slam')

        self.declare_parameter('topic_in', '/sonar/echo/data')
        self.declare_parameter('unwarp_image', False)
        
        self.needs_unwarp = self.get_parameter('unwarp_image').value
        self.br = CvBridge()
        self.sub = self.create_subscription(
            Image, 
            self.get_parameter('topic_in').value, 
            self.image_callback, 
            qos_profile_sensor_data
        )
        self.get_logger().info('Sonoptix Echo SOL-SLAM Frontend Initialized.')

    def image_callback(self, msg):
        cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Extract the 8-bit encoded max range from the bottom-left pixels
        bits = (cv_img[-1, :8] > 127).astype(np.uint8)
        max_range_m = int(np.packbits(bits)[0])
        
        # Black out the data pixels so they aren't processed as part of the sonar scan
        cv_img[-1, :8] = 0

        # Sonoptix Echo generic meta-data rule
        fov = 120 if max_range_m <= 30 else 90
        
        # Unwarp the sonar image if needed
        if self.needs_unwarp:
            cv_img = unwarp_sonar(cv_img, fov=fov, num_beams=768)

        h, w = cv_img.shape
        device = self.estimator.device if self.estimator else get_best_device()
        scan_tensor = torch.as_tensor(cv_img, dtype=torch.float32, device=device)
        stamp_sec = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

        frame = SonarFrame(
            scan_tensor=scan_tensor,
            max_range=max_range_m,
            fov_degrees=fov,
            timestamp=stamp_sec,
            height=h,
            width=w
        )
        self.process_sonar_frame(frame)

def main(args=None):
    rclpy.init(args=args)
    node = SonoptixEchoSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()