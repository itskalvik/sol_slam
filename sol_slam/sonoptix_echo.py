#! /usr/bin/env python3

import math
import numpy as np
import torch

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Quaternion
from nav_msgs.msg import Path
from cv_bridge import CvBridge

from .sol_slam_ros import SOLSLAMROSWrapper
from .sol_slam_core import SonarFrame, unwarp_sonar, get_best_device


def yaw_from_quaternion(q: Quaternion) -> float:
    """Extract yaw angle (around z-axis) from a ROS Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """Construct a ROS Quaternion representing a pure 2D yaw rotation."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class SonoptixEchoSLAMNode(SOLSLAMROSWrapper):
    """Hardware-specific frontend for Sonoptix Echo with zeroed and flipped DVL republishing."""
    def __init__(self):
        super().__init__('sonoptix_echo_slam')

        self.declare_parameter('topic_in', '/sonar/echo/data')
        self.declare_parameter('dvl_topic_in', '/a14/dvl/position_estimate')
        self.declare_parameter('unwarp_image', False)
        
        self.needs_unwarp = self.get_parameter('unwarp_image').value
        self.dvl_topic = self.get_parameter('dvl_topic_in').value
        self.br = CvBridge()

        # DVL Zeroing State & Path Buffers
        self.is_synced = False
        self.dvl_origin_pos = None  # (x, flipped_y, z)
        self.dvl_origin_yaw = 0.0   # flipped_yaw
        self.latest_raw_dvl = None

        # DVL Publishers
        self.dvl_pose_pub = self.create_publisher(PoseStamped, '/dvl/zeroed_pose', 10)
        self.dvl_path_pub = self.create_publisher(Path, '/dvl/zeroed_path', 10)

        self.dvl_path_msg = Path()
        self.dvl_path_msg.header.frame_id = self.odom_frame_id

        # Subscriptions
        self.sub = self.create_subscription(
            Image, 
            self.get_parameter('topic_in').value, 
            self.image_callback, 
            qos_profile_sensor_data
        )

        self.dvl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.dvl_topic,
            self.dvl_callback,
            10
        )

        self.get_logger().info('Sonoptix Echo SOL-SLAM Frontend Initialized.')

    def dvl_callback(self, msg: PoseWithCovarianceStamped):
        """Stores incoming DVL data and republishes zeroed path/pose with flipped y and yaw."""
        self.latest_raw_dvl = msg

        if not self.is_synced or self.dvl_origin_pos is None:
            return

        # Extract current raw DVL pose and apply axis/rotation flips
        curr_x = msg.pose.pose.position.x
        curr_y = -msg.pose.pose.position.y
        curr_z = msg.pose.pose.position.z
        curr_yaw = -yaw_from_quaternion(msg.pose.pose.orientation)

        # Delta position relative to origin in the flipped frame
        dx = curr_x - self.dvl_origin_pos[0]
        dy = curr_y - self.dvl_origin_pos[1]
        dz = curr_z - self.dvl_origin_pos[2]

        # Rotate displacement into reference coordinate frame (zero yaw at trigger time)
        cos_yaw = math.cos(-self.dvl_origin_yaw)
        sin_yaw = math.sin(-self.dvl_origin_yaw)
        
        rel_x = cos_yaw * dx - sin_yaw * dy
        rel_y = sin_yaw * dx + cos_yaw * dy
        rel_z = dz
        rel_yaw = curr_yaw - self.dvl_origin_yaw

        # Construct and publish zeroed PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.header.frame_id = self.odom_frame_id
        pose_stamped.pose.position.x = rel_x
        pose_stamped.pose.position.y = rel_y
        pose_stamped.pose.position.z = rel_z
        pose_stamped.pose.orientation = quaternion_from_yaw(rel_yaw)
        
        self.dvl_pose_pub.publish(pose_stamped)

        # Construct and publish zeroed Path
        self.dvl_path_msg.header = pose_stamped.header
        self.dvl_path_msg.poses.append(pose_stamped)
        self.dvl_path_msg.poses = self.dvl_path_msg.poses[-150:]
        self.dvl_path_pub.publish(self.dvl_path_msg)

    def reset_odom(self):
        """Zeroes/resets SLAM odometry and synchronizes zero frame for DVL."""
        # 1. Reset SLAM internal states and path
        if self.estimator is not None:
            self.estimator.step_counter = 0
            self.estimator.state.zero_()
            self.estimator.motion_model.reset()
            self.estimator.local_map.zero_()
        self.path_msg.poses.clear()

        # 2. Reset and latch DVL zero reference with flipped y and yaw
        if self.latest_raw_dvl is not None:
            pos = self.latest_raw_dvl.pose.pose.position
            self.dvl_origin_pos = (pos.x, -pos.y, pos.z)
            self.dvl_origin_yaw = -yaw_from_quaternion(self.latest_raw_dvl.pose.pose.orientation)
        else:
            self.dvl_origin_pos = (0.0, 0.0, 0.0)
            self.dvl_origin_yaw = 0.0

        self.dvl_path_msg.poses.clear()
        self.is_synced = True
        self.get_logger().info('SLAM initialized & compiled: Both SLAM and DVL origins successfully zeroed/synced.')

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

        # Flag whether this call triggers the initial compile/construction
        was_uncompiled = (self.estimator is None)

        self.process_sonar_frame(frame)

        # Once compiled and initialized, reset both odometry pipelines to sync origins
        if was_uncompiled and self.estimator is not None and not self.is_synced:
            self.reset_odom()

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