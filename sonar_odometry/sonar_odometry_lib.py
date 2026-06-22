import rclpy
from rclpy.node import Node
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch._dynamo
import torchvision

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped, Point
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

torch._dynamo.config.suppress_errors = True

# ==========================================
# Helper Functions & Device Management
# ==========================================
def get_best_device():
    """Returns the most performant available torch device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_rotation_matrix(rot, dtype=torch.float32, device='cpu'):
    c, s = torch.cos(rot), torch.sin(rot)
    rot_mat = torch.empty((2, 2), dtype=dtype, device=device)
    rot_mat[0, 0] = c; rot_mat[0, 1] = -s
    rot_mat[1, 0] = s; rot_mat[1, 1] = c
    return rot_mat

def unwarp_sonar(warped_image, fov=130, num_beams=768):
    """Utility to convert a fan-shaped sonar wedge into a raw polar matrix using OpenCV (CPU)"""
    h, d = warped_image.shape[:2]
    canvas = np.zeros((h * 2, h * 2), dtype=warped_image.dtype)
    s = (h * 2 - d) // 2
    canvas[0:h, s:s+d] = warped_image
    canvas = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    px_per_deg = num_beams / fov
    total_360_px = int(px_per_deg * 360)
    
    polar = cv2.warpPolar(canvas, dsize=(h, total_360_px), center=(h, h), 
                          maxRadius=h, flags=cv2.WARP_POLAR_LINEAR)
    raw_scan = polar.T
    padding = (total_360_px - num_beams) // 2
    raw_scan = raw_scan[:, padding:padding+num_beams]
    raw_scan = cv2.flip(raw_scan, 1)
        
    return raw_scan


# ==========================================
# Optimized Gradient Kernel
# ==========================================
@torch.compile(mode="reduce-overhead")
def compute_loss_and_grad(state, base_cart_scan, J_flat, M_flat, map_coords, map_res, map_size, flat_mask, N_valid):
    tx, ty, theta = state[0], state[1], state[2]
    cos_r, sin_r = torch.cos(theta), torch.sin(theta)
    
    x_shifted = map_coords[..., 0] - tx
    y_shifted = map_coords[..., 1] - ty
    
    x_old = x_shifted * cos_r + y_shifted * sin_r
    y_old = -x_shifted * sin_r + y_shifted * cos_r
    
    half_extent = map_res * (map_size - 1) / 2.0
    norm_x = x_old / half_extent
    norm_y = y_old / half_extent
    
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
    
    warped_scan = F.grid_sample(
        base_cart_scan.view(1, 1, map_size, map_size), 
        grid, 
        align_corners=True, 
        padding_mode='zeros'
    ).squeeze()
    
    warped_flat = warped_scan.view(-1)[flat_mask]
    
    grad = -(J_flat @ warped_flat) / N_valid
    loss = -torch.dot(M_flat, warped_flat) / N_valid
    
    return loss, grad


# ==========================================
# Robot Motion Model
# ==========================================
class RobotMotionModel:
    def __init__(self, dtype=torch.float32, device='cpu'):
        self.dtype = dtype
        self.device = device
        
        # Global Pose State
        self.pos = torch.zeros(2, dtype=self.dtype, device=self.device)
        self.rot = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        
        # Velocity State (Local Frame)
        self.vx = 0.0
        self.vy = 0.0
        self.vth = 0.0
        
        self.last_time = None

    def reset(self):
        self.pos.zero_()
        self.rot.zero_()
        self.vx, self.vy, self.vth = 0.0, 0.0, 0.0
        self.last_time = None

    def get_state_guess(self, current_time, default_guess):
        state_guess = torch.zeros(3, dtype=self.dtype, device=self.device)
        dt = current_time - self.last_time
        
        if dt > 1e-4:
            state_guess[0] = self.vx * dt
            state_guess[1] = self.vy * dt
            state_guess[2] = self.vth * dt
        else:
            state_guess = default_guess.clone().to(self.device)
            
        return state_guess

    def update(self, frame_pos, frame_rot, current_time):
        dt = current_time - self.last_time
        
        if dt > 1e-4:
            self.vx = frame_pos[0].item() / dt
            self.vy = frame_pos[1].item() / dt
            self.vth = frame_rot.item() / dt
        
        self.last_time = current_time
        
        rot_m = get_rotation_matrix(self.rot, dtype=self.dtype, device=self.device)
        self.pos += rot_m @ frame_pos
        self.rot += frame_rot
        
        return self.pos, self.rot


# ==========================================
# State Estimator 
# ==========================================
class SonarStateEstimator:
    def __init__(self, opts, res, node_logger):
        with torch.no_grad():
            self.device = get_best_device()
            self.dtype = torch.float32
            self.logger = node_logger
            self.logger.info(f"SonarStateEstimator initializing on device: {self.device.type.upper()}")
            
            self.sonar_res = torch.tensor(res, dtype=self.dtype, device=self.device)
            self.step_counter = 0

            self.state = torch.zeros(3, dtype=self.dtype, device=self.device)
            self.motion_model = RobotMotionModel(dtype=self.dtype, device=self.device)

            # Mapping parameters
            self.map_res = float(opts['mapping']['res'])
            self.max_range = float(opts['mapping']['max_range'])
            self.alpha = torch.tensor(opts['mapping']['update_alpha'], dtype=self.dtype, device=self.device)
            self.is_warped = bool(opts['mapping'].get('is_warped', False))
            
            self.blur_kernel_size = opts['mapping'].get('blur_kernel_size', [1, 1])
            self.blur_sigma = float(opts['mapping'].get('blur_sigma', 0.0))
            self.noise_decay_factor = float(opts['mapping'].get('noise_decay_factor', 4.0))
            self.denoise_factor = float(opts['mapping'].get('denoise_factor', 3.25))
            
            self.min_intensity_threshold = float(opts['mapping'].get('min_intensity_threshold', 0.1))
            self.min_valid_fraction = float(opts['mapping'].get('min_valid_fraction', 0.01))

            fov_deg = float(opts['mapping'].get('fov', 130.0))
            self.fov_rad = torch.tensor(np.radians(fov_deg), dtype=self.dtype, device=self.device)
            self.half_width_phys = self.max_range * torch.sin(self.fov_rad / 2.0)
            
            num_beams = int(opts['mapping'].get('num_beams', 768))
            angle_bins = torch.arange(num_beams, dtype=self.dtype, device=self.device)
            self.beam_angles = (angle_bins / float(num_beams - 1)) * self.fov_rad - (self.fov_rad / 2.0)

            self.map_size = int(self.max_range / self.map_res) * 2 + 1
            
            # Solver parameters
            solver_opts = opts.get('solver', {})
            self.iterations = int(solver_opts.get('iterations', 15)) 
            self.lr = float(solver_opts.get('lr', 1.0)) 
            self.grad_tol = float(solver_opts.get('grad_tol', 1e-6))
            self.loss_tol = float(solver_opts.get('loss_tol', 1e-9))
            self.max_grad_norm = float(solver_opts.get('max_grad_norm', 10.0))
            self.lm_damping = float(solver_opts.get('lm_damping', 1e-3)) 
            
            self.bound_factor = float(solver_opts.get('bound_factor', 1.75))
            self.fallback_bound = torch.tensor(solver_opts.get('fallback_bound', [0.25, 0.25, 0.08725]), 
                                               dtype=self.dtype, device=self.device)

            self.local_map = torch.zeros((self.map_size, self.map_size), dtype=self.dtype, device=self.device)
            self.local_map_blurred = torch.zeros_like(self.local_map)
            self.last_cart_scan = None
            self.last_polar_scan = None # Added to extract peaks natively
            
            axis = (torch.arange(self.map_size, dtype=self.dtype, device=self.device) - self.map_size // 2) * self.map_res
            grid_y, grid_x = torch.meshgrid(axis, axis, indexing='ij')
            self.map_coords = torch.stack([grid_x, grid_y], dim=-1).to(self.device)

            x_p, y_p = self.map_coords[..., 0], self.map_coords[..., 1]
            eps = 1e-6
            r2 = x_p**2 + y_p**2
            safe = r2 > eps**2
            x_safe = torch.where(safe, x_p, torch.full_like(x_p, eps, dtype=self.dtype, device=self.device))
            y_safe = torch.where(safe, y_p, torch.zeros_like(y_p, dtype=self.dtype, device=self.device))
            
            self.r = torch.sqrt(x_safe**2 + y_safe**2)
            self.theta = torch.atan2(y_safe, x_safe)
            a_min, a_max = self.beam_angles[0], self.beam_angles[-1]
            
            self.sonar_mask = (self.r < self.max_range) & (self.theta >= a_min) & (self.theta <= a_max)
            self.flat_mask = self.sonar_mask.view(-1)
            self.N_valid = self.flat_mask.sum().item()

    def preprocess_sonar(self, scan_img):
        # Explicitly move to the correct device and strip dimensions
        scan = scan_img.clone().detach().to(device=self.device, dtype=self.dtype).squeeze()
        if scan.max() > 2.0: # Normalize if 0-255 scale
            scan /= 255.0
        scan[scan < scan.mean() + self.denoise_factor * scan.std()] = 0.0
        return scan

    def get_cartesian_projection(self, scan_img):
        local_x, local_y = self.map_coords[..., 0], self.map_coords[..., 1]
        H_img, W_img = scan_img.shape 

        if self.is_warped:
            norm_y = 1.0 - 2.0 * (local_x / self.max_range)
            norm_x = -(local_y / self.half_width_phys)
            coords = torch.stack([norm_x, norm_y], dim=-1).to(self.device)
        else:
            a_min, a_max = self.beam_angles[0], self.beam_angles[-1]
            a_idx = (self.theta - a_min) / (a_max - a_min) * (W_img - 1)
            r_idx = self.r / self.sonar_res
            norm_a = (a_idx / (W_img - 1) * 2) - 1
            norm_r = (r_idx / (H_img - 1) * 2) - 1
            coords = torch.stack([norm_a, norm_r], dim=-1).to(self.device)

        cart = F.grid_sample(
            scan_img.view(1, 1, H_img, W_img), 
            coords.unsqueeze(0), 
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze()
        
        return torch.where(self.sonar_mask, cart, 0.0)

    def move_local_map(self, delta_pos, delta_rot):
        if torch.norm(delta_pos) < 1e-4 and torch.abs(delta_rot) < 1e-5: 
            return
        
        rot_mat = get_rotation_matrix(delta_rot, dtype=self.dtype, device=self.device)
        shifted = (self.map_coords.view(-1, 2) @ rot_mat.T) + delta_pos
        
        pixel_idx = (shifted / self.map_res) + (self.map_size // 2)
        pixel_idx = (pixel_idx / (self.map_size - 1) * 2) - 1
        
        self.local_map = F.grid_sample(
            self.local_map.view(1, 1, self.map_size, self.map_size), 
            pixel_idx.view(1, self.map_size, self.map_size, 2),
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze()
        
    def solve(self, state_init, base_cart_scan):
        res = self.map_res
        M = self.local_map_blurred.detach()
        
        M_x = torch.zeros_like(M, dtype=self.dtype, device=self.device)
        M_y = torch.zeros_like(M, dtype=self.dtype, device=self.device)
        M_x[:, 1:-1] = (M[:, 2:] - M[:, :-2]) / (2 * res)
        M_y[1:-1, :] = (M[2:, :] - M[:-2, :]) / (2 * res)
        
        x_grid = self.map_coords[..., 0]
        y_grid = self.map_coords[..., 1]
        M_theta = -y_grid * M_x + x_grid * M_y

        M_x_valid = M_x.view(-1)[self.flat_mask]
        M_y_valid = M_y.view(-1)[self.flat_mask]
        M_theta_valid = M_theta.view(-1)[self.flat_mask]
        
        J_flat = torch.stack([M_x_valid, M_y_valid, M_theta_valid]).to(self.device)
        M_flat = M.view(-1)[self.flat_mask]

        H = (J_flat @ J_flat.T) / self.N_valid
        H += torch.eye(3, dtype=self.dtype, device=self.device) * self.lm_damping
        H_inv = torch.linalg.inv(H)

        state = state_init.detach().clone().flatten().to(self.device)
        prev_loss = None

        for _ in range(1, self.iterations + 1):
            loss, grad = compute_loss_and_grad(
                state, base_cart_scan, J_flat, M_flat, 
                self.map_coords, self.map_res, self.map_size,
                self.flat_mask, self.N_valid
            )

            if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                break

            grad_norm = torch.linalg.vector_norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)
                grad_norm = self.max_grad_norm

            if grad_norm < self.grad_tol:
                break

            delta_q = H_inv @ grad
            state = state - self.lr * delta_q

            loss_value = loss.item()
            if prev_loss is not None and abs(prev_loss - loss_value) < self.loss_tol:
                break
            prev_loss = loss_value

        return state

    def odometry_step(self, scan_img, current_time):
        with torch.no_grad():           
            clean_scan = self.preprocess_sonar(scan_img)
            base_cart_scan = self.get_cartesian_projection(clean_scan)

            footprint_pixels = base_cart_scan[self.sonar_mask]
            active_pixels_count = (footprint_pixels > self.min_intensity_threshold).sum().float()
            active_fraction = (active_pixels_count / self.N_valid).item()

            if active_fraction < self.min_valid_fraction:
                self.logger.warn(f"Empty scan detected (Fraction: {active_fraction:.4f}). Skipping optimization.", throttle_duration_sec=1.0)
                self.motion_model.last_time = current_time
                self.motion_model.vx, self.motion_model.vy, self.motion_model.vth = 0.0, 0.0, 0.0
                return

            if self.step_counter > 0:
                state_guess = self.motion_model.get_state_guess(current_time, default_guess=self.state)
                proposed_state = self.solve(state_guess, base_cart_scan)
                
                # Bounding & Rejection
                current_bound = self.fallback_bound
                if torch.norm(state_guess) > 1e-5:
                    dynamic_bound = torch.abs(state_guess) * self.bound_factor
                    current_bound = torch.max(dynamic_bound, self.fallback_bound)

                if (torch.abs(proposed_state[0]) > current_bound[0]) or \
                   (torch.abs(proposed_state[1]) > current_bound[1]) or \
                   (torch.abs(proposed_state[2]) > current_bound[2]):
                    
                    self.logger.warn("Optimization Rejected. Movement exceeded bounds.", throttle_duration_sec=1.0)
                    self.state = state_guess 
                else:
                    self.state = proposed_state

                frame_pos, frame_rot = self.state[0:2], self.state[2]

                self.motion_model.update(frame_pos, frame_rot, current_time)
                self.move_local_map(frame_pos, frame_rot)
                
                mask_2d = self.sonar_mask
                valid_scan = base_cart_scan > 0
                update_mask = mask_2d & valid_scan
                
                self.local_map[update_mask] = (1 - self.alpha) * self.local_map[update_mask] + self.alpha * base_cart_scan[update_mask]
                self.local_map[self.local_map < (self.local_map.mean() + self.noise_decay_factor * self.local_map.std())] = 0.0
            else:
                self.local_map = base_cart_scan.clone().to(self.device)
                self.motion_model.last_time = current_time

            if self.blur_kernel_size[0] > 1:
                self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(
                    self.local_map.unsqueeze(0).unsqueeze(0), self.blur_kernel_size, self.blur_sigma
                ).squeeze()
            else:
                self.local_map_blurred = self.local_map

            self.step_counter += 1
            self.last_cart_scan = base_cart_scan
            self.last_polar_scan = clean_scan # Store to make peak extraction highly efficient


# ==========================================
# ROS 2 Base Class
# ==========================================
class BaseSonarOdometryNode(Node):
    def __init__(self, node_name='base_sonar_odometry'):
        super().__init__(node_name)

        # Frame Parameters
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('topic_out', '/odom')
        self.declare_parameter('path_topic', '/sonar_path')

        # Visual Debug Parameters
        self.declare_parameter('scan_img_topic', '/sonar_scan_img')
        self.declare_parameter('map_img_topic', '/sonar_map_img')
        self.declare_parameter('peaks_topic', '/sonar_peaks')
        self.declare_parameter('contrast', 10.0)

        # Core Parameters (Matching ROS 1 Method)
        self.declare_parameter('mapping.update_alpha', 0.3)
        self.declare_parameter('mapping.is_warped', False)
        self.declare_parameter('mapping.noise_decay_factor', 3.0)
        self.declare_parameter('mapping.denoise_factor', 3.0)
        self.declare_parameter('mapping.min_intensity_threshold', 0.0008)
        self.declare_parameter('mapping.min_valid_fraction', 0.006)

        self.declare_parameter('solver.iterations', 15)
        self.declare_parameter('solver.lr', 1.0)
        self.declare_parameter('solver.grad_tol', 1e-6)
        self.declare_parameter('solver.loss_tol', 1e-9)
        self.declare_parameter('solver.max_grad_norm', 10.0)
        self.declare_parameter('solver.lm_damping', 1e-3)
        self.declare_parameter('solver.bound_factor', 1.75)

        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.contrast = self.get_parameter('contrast').value
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('topic_out').value, 10)
        self.path_pub = self.create_publisher(Path, self.get_parameter('path_topic').value, 10)
        self.peaks_pub = self.create_publisher(PointCloud2, self.get_parameter('peaks_topic').value, 10)

        # Vision Debug Publishers
        self.scan_img_pub = self.create_publisher(Image, self.get_parameter('scan_img_topic').value, 10)
        self.map_img_pub = self.create_publisher(Image, self.get_parameter('map_img_topic').value, 10)
        self.bridge = CvBridge()

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame_id

        self.add_on_set_parameters_callback(self.parameter_callback)
        self.reset_srv = self.create_service(Trigger, '~/reset_odometry', self.reset_odometry_callback)

        self.estimator = None
        self.current_range = -1.0
        self.current_shape = None

    def parameter_callback(self, params):
        result = SetParametersResult(successful=True)
        for param in params:
            if param.name == 'contrast':
                self.contrast = param.value
                self.get_logger().info(f"Visual contrast updated to {self.contrast}")
            if self.estimator is not None:
                if param.name == 'mapping.update_alpha':
                    self.estimator.alpha = torch.tensor(param.value, device=self.estimator.device)
                elif param.name == 'solver.lr':
                    self.estimator.lr = param.value
                self.get_logger().info(f"Param Updated -> '{param.name}': {param.value}")
        return result

    def reset_odometry_callback(self, request, response):
        if self.estimator is not None:
            self.estimator.step_counter = 0
            self.estimator.state.zero_()
            self.estimator.motion_model.reset()
            self.estimator.local_map.zero_()
            self.estimator.local_map_blurred.zero_()
            
            response.success = True
            response.message = "Odometry and maps reset."
        else:
            response.success = False
            response.message = "Estimator not initialized."
        return response

    def initialize_estimator(self, h, w, max_range, fov_degrees):
        self.get_logger().info(f'Initializing Estimator: Range {max_range}m, Shape ({h}, {w}), FOV {fov_degrees}°')
        res = float(max_range / h)

        opts = {
            'mapping': {
                'max_range': max_range, 'res': res, 'fov': fov_degrees, 'num_beams': w,
                'update_alpha': self.get_parameter('mapping.update_alpha').value,
                'is_warped': self.get_parameter('mapping.is_warped').value,
                'noise_decay_factor': self.get_parameter('mapping.noise_decay_factor').value,
                'denoise_factor': self.get_parameter('mapping.denoise_factor').value,
                'min_intensity_threshold': self.get_parameter('mapping.min_intensity_threshold').value,
                'min_valid_fraction': self.get_parameter('mapping.min_valid_fraction').value,
                'blur_kernel_size': [1, 1], 'blur_sigma': 0.0
            },
            'solver': {
                'iterations': self.get_parameter('solver.iterations').value, 
                'lr': self.get_parameter('solver.lr').value, 
                'grad_tol': self.get_parameter('solver.grad_tol').value, 
                'loss_tol': self.get_parameter('solver.loss_tol').value, 
                'max_grad_norm': self.get_parameter('solver.max_grad_norm').value, 
                'lm_damping': self.get_parameter('solver.lm_damping').value,
                'bound_factor': self.get_parameter('solver.bound_factor').value,
                'fallback_bound': [0.25, 0.25, 0.08725]
            }
        }
        
        self.estimator = SonarStateEstimator(opts, res, self.get_logger())
        self.current_range = max_range
        self.current_shape = (h, w)
        
        # JIT Warmup
        self.get_logger().info('Warming up JIT compiler...')
        dummy_scan = torch.zeros((h, w), dtype=torch.float32, device=self.estimator.device)
        self.estimator.odometry_step(dummy_scan, 0.0)
        self.estimator.odometry_step(dummy_scan, 0.1)
        
        self.estimator.step_counter = 0
        self.estimator.state.zero_()
        self.estimator.motion_model.reset()
        self.estimator.local_map.zero_()
        self.get_logger().info('Warmup complete.')

    def process_sonar_scan(self, scan_tensor, max_range_m, h, w, fov_degrees, stamp):
        if self.estimator is None or self.current_range != max_range_m or self.current_shape != (h, w):
            self.initialize_estimator(h, w, max_range_m, fov_degrees)

        current_time = stamp.sec + (stamp.nanosec * 1e-9) if stamp else 0.0
        
        # Ensure scan is on the designated device before optimization
        scan_tensor = scan_tensor.to(self.estimator.device)
        self.estimator.odometry_step(scan_tensor, current_time)

        # Convert to standard ROS global frame 
        pos_x = self.estimator.motion_model.pos[0].item()
        pos_y = -self.estimator.motion_model.pos[1].item()
        rot_theta = -self.estimator.motion_model.rot.item()
        
        vx = self.estimator.motion_model.vx
        vy = -self.estimator.motion_model.vy
        vth = -self.estimator.motion_model.vth

        self.publish_odometry(stamp, pos_x, pos_y, rot_theta, vx, vy, vth)
        self.publish_debug_images(stamp)
        self.publish_peaks(stamp)

    def publish_peaks(self, stamp):
        """Extracts the maximum intensity range per beam angle and publishes it as a PointCloud2."""
        if getattr(self.estimator, 'last_polar_scan', None) is None:
            return

        try:
            # We use the polar matrix (W beams, H ranges) directly as it computes peaks flawlessly
            # without Cartesian grid blurring artifacts, giving the exact equivalent of last_cart_scan peaks
            scan = self.estimator.last_polar_scan
            
            # Find the max intensity and its distance index for each angle beam
            max_vals, max_indices = scan.max(dim=0)
            
            max_vals = max_vals.cpu().numpy()
            max_indices = max_indices.cpu().numpy()
            angles = self.estimator.beam_angles.cpu().numpy()
            
            H, W = scan.shape
            max_range = self.current_range
            threshold = self.get_parameter('mapping.min_intensity_threshold').value
            
            points = []
            
            for i in range(W):
                if max_vals[i] > threshold:
                    # Convert range index to physical distance
                    r = (max_indices[i] / float(H - 1)) * max_range
                    theta = angles[i]
                    
                    # Convert to cartesian coordinates local to the sonar base frame
                    # standard ROS coordinates: x forward, y left
                    x = float(r * math.cos(theta))
                    y = float(r * math.sin(theta))
                    z = 0.0
                    
                    points.append([x, y, z])
                    
            if points:
                header = Header()
                if stamp: 
                    header.stamp = stamp
                header.frame_id = self.child_frame_id
                
                # pc2 handles the exact byte formatting needed for the pointcloud
                cloud_msg = pc2.create_cloud_xyz32(header, points)
                self.peaks_pub.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish sonar peaks point cloud: {e}")

    def publish_debug_images(self, stamp):
        """Publishes Viridis-colored OpenCV images for map debugging."""
        if not hasattr(self.estimator, 'local_map') or getattr(self.estimator, 'last_cart_scan', None) is None:
            return

        try:
            # 1. Local Map
            map_array = self.estimator.local_map.detach().cpu().numpy()
            map_array = np.clip(map_array * self.contrast, 0, 1)
            map_array = cv2.rotate(map_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            map_array = cv2.flip(map_array, 1)
            
            map_uint8 = (map_array * 255).astype(np.uint8)
            map_color = cv2.applyColorMap(map_uint8, cv2.COLORMAP_VIRIDIS)
            map_img_msg = self.bridge.cv2_to_imgmsg(map_color, encoding="bgr8")
            if stamp: map_img_msg.header.stamp = stamp
            map_img_msg.header.frame_id = self.odom_frame_id
            self.map_img_pub.publish(map_img_msg)

            # 2. Cartesian Scan
            scan_array = self.estimator.last_cart_scan.detach().cpu().numpy()
            # Crop to matching active half 
            scan_array = scan_array[:, int(scan_array.shape[0]/2):]
            scan_array = np.clip(scan_array * self.contrast, 0, 1)
            scan_array = cv2.rotate(scan_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            scan_array = cv2.flip(scan_array, 1)
            
            scan_uint8 = (scan_array * 255).astype(np.uint8)
            scan_color = cv2.applyColorMap(scan_uint8, cv2.COLORMAP_VIRIDIS)
            scan_img_msg = self.bridge.cv2_to_imgmsg(scan_color, encoding="bgr8")
            if stamp: scan_img_msg.header.stamp = stamp
            scan_img_msg.header.frame_id = self.child_frame_id
            self.scan_img_pub.publish(scan_img_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish debug images: {e}")

    def publish_odometry(self, stamp, x, y, yaw, vx, vy, vth):
        q = Quaternion()
        q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

        odom = Odometry()
        if stamp: odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.child_frame_id
        
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation = q
        
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = vth

        self.odom_pub.publish(odom)

        t = TransformStamped()
        if stamp: t.header.stamp = stamp
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.child_frame_id
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)

        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self.path_msg.header = odom.header
        self.path_msg.poses.append(pose_stamped)
        self.path_pub.publish(self.path_msg)