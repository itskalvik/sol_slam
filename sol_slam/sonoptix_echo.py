#! /usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import torch
from .sol_slam_lib import SOLSLAMBaseNode, unwarp_sonar, get_best_device

class SonoptixEchoSOLSLAM(SOLSLAMBaseNode):
    """
    Hardware-specific implementation for the Sonoptix Echo Sonar.
    Handles the `Image` topic and extracts the embedded range.
    """
    def __init__(self):
        super().__init__('sonoptix_echo')

        self.declare_parameter('topic_in', '/sonar/echo/data')
        self.declare_parameter('unwarp_image', False)
        
        topic_in = self.get_parameter('topic_in').value
        self.needs_unwarp = self.get_parameter('unwarp_image').value

        self.br = CvBridge()
        self.sub = self.create_subscription(
            Image, 
            topic_in, 
            self.image_callback, 
            qos_profile_sensor_data
        )

        self.get_logger().info('Sonoptix Echo SOL-SLAM Initialized.')

    def image_callback(self, msg):
        cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = cv_img.shape

        # Sonoptix generic meta-data rule
        max_range_m = 12 
        fov = 120 if max_range_m <= 30 else 90

        # Clean the artifact pixel
        cv_img[0, 0] = 0
        
        # Apply the Unwarp method if the image is a fan/wedge 
        if self.needs_unwarp:
            cv_img = unwarp_sonar(cv_img, fov=fov, num_beams=768)
            h, w = cv_img.shape # Update to the new polar dimensions

        # Force layout to the highest performance device available before dropping into base class
        device = self.estimator.device if self.estimator else get_best_device()
        scan_tensor = torch.as_tensor(cv_img, dtype=torch.float32, device=device)

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
    node = SonoptixEchoSOLSLAM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()