import rclpy
from rclpy.node import Node
import threading
import math
import numpy as np
import cv2
import dataclasses

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from .sol_slam_core import SOLSLAMCore, SonarFrame, SOLSLAMConfig

class SOLSLAMROSWrapper(Node):
    def __init__(self, node_name='sol_slam_base'):
        super().__init__(node_name)

        # 1. Base Node Parameters
        self.declare_parameter('enable_tensorrt', False)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'map')
        self.declare_parameter('enable_debug_viz', True)
        self.declare_parameter('contrast', 10.0)

        self.enable_tensorrt = self.get_parameter('enable_tensorrt').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value
        self.contrast = self.get_parameter('contrast').value

        # 2. Automated SLAM Tuning Parameters (handles missing types safely)
        self.slam_config = self._load_slam_parameters()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/sol_slam/odom', 10)
        self.path_pub = self.create_publisher(Path, '/sol_slam/path', 10)
        
        self.viz_lock = threading.Lock()
        self.viz_event = threading.Event()
        self.latest_viz_data = None
        
        if self.enable_debug_viz:
            self.peaks_pub = self.create_publisher(PointCloud2, '/sol_slam/peaks', 10)
            self.scan_img_pub = self.create_publisher(Image, '/sol_slam/denoised_scan', 10)
            self.map_img_pub = self.create_publisher(Image, '/sol_slam/map', 10)
            self.bridge = CvBridge()
            self.viz_thread = threading.Thread(target=self.viz_worker_loop, daemon=True)
            self.viz_thread.start()

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame_id
        
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.reset_srv = self.create_service(Trigger, '~/reset_odometry', self.reset_odometry_callback)

        self.estimator = None

    def _load_slam_parameters(self) -> SOLSLAMConfig:
        config = SOLSLAMConfig()
        for field_name, field_def in config.__dataclass_fields__.items():
            if field_def.default is not dataclasses.MISSING:
                default_val = field_def.default
            else:
                default_val = field_def.default_factory()
            self.declare_parameter(field_name, default_val)
            setattr(config, field_name, self.get_parameter(field_name).value)
        return config

    def parameter_callback(self, params):
        result = SetParametersResult(successful=True)
        for param in params:
            if param.name == 'contrast':
                self.contrast = param.value
            elif param.name == 'enable_debug_viz':
                self.enable_debug_viz = param.value
            elif hasattr(self.slam_config, param.name):
                setattr(self.slam_config, param.name, param.value)
                if self.estimator is not None:
                    if param.name == 'update_alpha':
                        self.estimator.alpha = torch.tensor(param.value, device=self.estimator.device)
                    elif param.name == 'lr':
                        self.estimator.solver_kernel.lr = float(param.value)
        return result

    def reset_odometry_callback(self, request, response):
        if self.estimator:
            self.estimator.step_counter = 0
            self.estimator.state.zero_()
            self.estimator.motion_model.reset()
            self.estimator.local_map.zero_()
            self.path_msg.poses.clear()
            response.success = True
        return response

    def process_sonar_frame(self, frame: SonarFrame):
        if self.estimator is None:
            res = float(frame.max_range / frame.height)
            self.estimator = SOLSLAMCore(
                self.slam_config, res, frame.height, frame.width,
                self.enable_tensorrt, frame.max_range, frame.fov_degrees
            )

        success = self.estimator.process_frame(frame)
        if not success: 
            return

        pos = self.estimator.motion_model.pos
        rot = self.estimator.motion_model.rot
        vx = self.estimator.motion_model.vx
        vy = self.estimator.motion_model.vy
        vth = self.estimator.motion_model.vth

        # 1. Publish odom, TF, and path
        self._publish_odometry(frame.timestamp, pos[0].item(), -pos[1].item(), -rot.item(), vx, -vy, -vth)
        
        # 2. Hand off tensors to the async visualization thread
        if self.enable_debug_viz:
            map_clone = self.estimator.local_map.detach().clone()
            scan_clone = self.estimator.last_cart_scan.detach().clone()
            with self.viz_lock:
                self.latest_viz_data = (frame.timestamp, map_clone, scan_clone)
            self.viz_event.set()

    def _publish_odometry(self, stamp_sec, x, y, yaw, vx, vy, vth):
        stamp = rclpy.time.Time(seconds=stamp_sec).to_msg()
        q = Quaternion()
        q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

        # Odometry message
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = vx, vy, vth
        self.odom_pub.publish(odom)

        # TF broadcast
        t = TransformStamped()
        t.header = odom.header
        t.child_frame_id = self.child_frame_id
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)

        # Path publishing
        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self.path_msg.header = odom.header
        self.path_msg.poses.append(pose_stamped)
        self.path_pub.publish(self.path_msg)

    def publish_peaks(self, stamp, map_tensor):
        """Extracts peak points from the local map and publishes a PointCloud2."""
        try:
            threshold = self.slam_config.min_intensity_threshold
            mask = map_tensor > threshold
            if not mask.any():
                return

            valid_coords = self.estimator.map_coords[mask].cpu().numpy()
            valid_intensities = map_tensor[mask].cpu().numpy()

            # Filter points within 1.0 m (sonar body artifacts)
            r2 = valid_coords[:, 0]**2 + valid_coords[:, 1]**2
            valid_mask = r2 > 1.0

            valid_coords = valid_coords[valid_mask]
            valid_intensities = valid_intensities[valid_mask]

            num_points = len(valid_coords)
            if num_points == 0:
                return

            points_np = np.zeros((num_points, 4), dtype=np.float32)
            points_np[:, 0] = valid_coords[:, 0]
            points_np[:, 1] = -valid_coords[:, 1]
            points_np[:, 3] = valid_intensities

            header = Header()
            header.stamp = stamp
            header.frame_id = self.child_frame_id

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            cloud_msg = PointCloud2()
            cloud_msg.header = header
            cloud_msg.height = 1
            cloud_msg.width = num_points
            cloud_msg.fields = fields
            cloud_msg.is_bigendian = False
            cloud_msg.point_step = 16
            cloud_msg.row_step = 16 * num_points
            cloud_msg.is_dense = True
            cloud_msg.data = points_np.tobytes()

            self.peaks_pub.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Peaks PointCloud Error: {e}")

    def viz_worker_loop(self):
        while rclpy.ok():
            self.viz_event.wait(timeout=0.1)
            if not self.viz_event.is_set(): 
                continue
            self.viz_event.clear()
            
            with self.viz_lock:
                if self.latest_viz_data is None: 
                    continue
                stamp_sec, map_tensor, scan_tensor = self.latest_viz_data
                self.latest_viz_data = None 
                
            try:
                stamp = rclpy.time.Time(seconds=stamp_sec).to_msg()
                
                # 1. Publish PointCloud2 Peaks
                self.publish_peaks(stamp, map_tensor)

                # 2. Render Map Image
                map_arr = np.clip(map_tensor.cpu().numpy() * self.contrast, 0, 1)
                map_c = cv2.applyColorMap((cv2.flip(cv2.rotate(map_arr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                map_msg = self.bridge.cv2_to_imgmsg(map_c, encoding="bgr8")
                map_msg.header.stamp = stamp
                map_msg.header.frame_id = self.odom_frame_id
                self.map_img_pub.publish(map_msg)

                # 3. Render Scan Image
                scan_arr = scan_tensor.cpu().numpy()
                scan_arr = np.clip(scan_arr[:, int(scan_arr.shape[0]/2):] * self.contrast, 0, 1)
                scan_c = cv2.applyColorMap((cv2.flip(cv2.rotate(scan_arr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                scan_msg = self.bridge.cv2_to_imgmsg(scan_c, encoding="bgr8")
                scan_msg.header.stamp = stamp
                scan_msg.header.frame_id = self.child_frame_id
                self.scan_img_pub.publish(scan_msg)

            except Exception as e:
                self.get_logger().error(f"Async Viz Error: {e}")