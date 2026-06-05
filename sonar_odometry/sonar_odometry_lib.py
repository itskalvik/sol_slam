import rclpy
from rclpy.node import Node
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch._dynamo
import torchvision

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import Trigger

torch._dynamo.config.suppress_errors = True

# ==========================================
# Optimized Gradient Kernel & Motion Models
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

class MotionModel:
    def __init__(self, state_size, device='cpu'):
        self.state_size = torch.tensor(state_size, device=device)
        self.device = device
        self.time = 0.0
        self.t0 = 0.0

    def setTime(self, time, t0):
        # Now expects a single scalar timestamp for multi-beam sonars
        self.time = (float(time) - float(t0)) * 1.0e-6
        self.t0 = float(t0)

    def getInitialState(self):
        return torch.zeros(self.state_size, device=self.device)

class ConstVel(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(3, device)

    def getPosRotSingle(self, state, time):
        if state.numel() < 3:
            state = F.pad(state, (0, 3 - state.numel()))
        return state[:2], state[2]

# ==========================================
# State Estimator 
# ==========================================
class SonarStateEstimator:
    def __init__(self, opts, res, beam_angles):
        with torch.no_grad():
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.sonar_res = torch.tensor(res, device=self.device)
            self.step_counter = 0
            self.beam_angles = torch.tensor(beam_angles, device=self.device).float()

            self.motion_model = ConstVel(device=self.device)
            self.state_init = self.motion_model.getInitialState()
            self.current_pos = torch.zeros(2, device=self.device).double()
            self.current_rot = torch.tensor(0.0, device=self.device).double()

            self.map_res = float(opts['mapping']['res'])
            max_range = float(opts['mapping']['max_range'])
            self.map_size = int(max_range / self.map_res) * 2 + 1
            self.alpha = torch.tensor(opts['mapping']['update_alpha'], device=self.device)
            
            solver_opts = opts.get('solver', {})
            self.iterations = int(solver_opts.get('iterations', 15)) 
            self.lr = float(solver_opts.get('lr', 1.0)) 
            self.grad_tol = float(solver_opts.get('grad_tol', 1e-6))
            self.loss_tol = float(solver_opts.get('loss_tol', 1e-9))
            self.max_grad_norm = float(solver_opts.get('max_grad_norm', 10.0))
            self.lm_damping = float(solver_opts.get('lm_damping', 1e-3)) 

            self.local_map = torch.zeros((self.map_size, self.map_size), device=self.device)
            self.local_map_blurred = torch.zeros_like(self.local_map)
            
            axis = (torch.arange(self.map_size, device=self.device) - self.map_size // 2) * self.map_res
            grid_y, grid_x = torch.meshgrid(axis, axis, indexing='ij')
            self.map_coords = torch.stack([grid_x, grid_y], dim=-1).float() 

            x_p, y_p = self.map_coords[..., 0], self.map_coords[..., 1]
            eps = 1e-6
            r2 = x_p**2 + y_p**2
            safe = r2 > eps**2
            x_safe = torch.where(safe, x_p, torch.full_like(x_p, eps))
            y_safe = torch.where(safe, y_p, torch.zeros_like(y_p))
            
            r = torch.sqrt(x_safe**2 + y_safe**2)
            theta = torch.atan2(y_safe, x_safe)
            a_min, a_max = self.beam_angles[0], self.beam_angles[-1]
            
            self.sonar_mask = (r < max_range) & (theta >= a_min) & (theta <= a_max)
            self.flat_mask = self.sonar_mask.view(-1)
            self.N_valid = self.flat_mask.sum().item()

    def get_rotation_matrix(self, rot, dtype=torch.float32):
        c, s = torch.cos(rot), torch.sin(rot)
        rot_mat = torch.empty((2, 2), device=self.device, dtype=dtype)
        rot_mat[0, 0] = c; rot_mat[0, 1] = -s
        rot_mat[1, 0] = s; rot_mat[1, 1] = c
        return rot_mat

    def get_cartesian_projection(self, polar_img):
        pts = self.map_coords 
        x_p, y_p = pts[..., 0], pts[..., 1]
        eps = 1e-6
        r2 = x_p**2 + y_p**2

        safe = r2 > eps**2
        x_safe = torch.where(safe, x_p, torch.full_like(x_p, eps))
        y_safe = torch.where(safe, y_p, torch.zeros_like(y_p))

        r = torch.sqrt(x_safe**2 + y_safe**2)
        theta = torch.atan2(y_safe, x_safe)

        H_p, W_p = polar_img.shape 
        a_min, a_max = self.beam_angles[0], self.beam_angles[-1]
        
        a_idx = (theta - a_min) / (a_max - a_min) * (W_p - 1)
        r_idx = r / self.sonar_res

        norm_a = (a_idx / (W_p - 1) * 2) - 1
        norm_r = (r_idx / (H_p - 1) * 2) - 1
        coords = torch.stack([norm_a, norm_r], dim=-1)

        cart = F.grid_sample(
            polar_img.view(1, 1, H_p, W_p), 
            coords.unsqueeze(0), 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        ).squeeze()
        
        return cart * self.sonar_mask.float()

    def move_local_map(self, delta_pos, delta_rot):
        if torch.norm(delta_pos) < 1e-4 and torch.abs(delta_rot) < 1e-5: 
            return
        
        rot_mat = self.get_rotation_matrix(delta_rot, self.map_coords.dtype)
        shifted = (self.map_coords.view(-1, 2) @ rot_mat.T) + delta_pos.to(device=self.device, dtype=self.map_coords.dtype)
        
        pixel_idx = (shifted / self.map_res) + (self.map_size // 2)
        pixel_idx = (pixel_idx / (self.map_size - 1) * 2) - 1
        
        self.local_map = F.grid_sample(
            self.local_map.view(1, 1, self.map_size, self.map_size), 
            pixel_idx.view(1, self.map_size, self.map_size, 2),
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        ).squeeze()
        
    def solve(self, state_init, base_cart_scan):
        res = self.map_res
        M = self.local_map_blurred.detach()
        
        M_x = torch.zeros_like(M)
        M_y = torch.zeros_like(M)
        M_x[:, 1:-1] = (M[:, 2:] - M[:, :-2]) / (2 * res)
        M_y[1:-1, :] = (M[2:, :] - M[:-2, :]) / (2 * res)
        
        x_grid = self.map_coords[..., 0]
        y_grid = self.map_coords[..., 1]
        M_theta = -y_grid * M_x + x_grid * M_y

        M_x_valid = M_x.view(-1)[self.flat_mask]
        M_y_valid = M_y.view(-1)[self.flat_mask]
        M_theta_valid = M_theta.view(-1)[self.flat_mask]
        
        J_flat = torch.stack([M_x_valid, M_y_valid, M_theta_valid])
        M_flat = M.view(-1)[self.flat_mask]

        H = (J_flat @ J_flat.T) / self.N_valid
        H += torch.eye(3, device=self.device, dtype=torch.float32) * self.lm_damping
        H_inv = torch.linalg.inv(H)

        state = state_init.detach().to(device=self.device, dtype=torch.float32).flatten()
        if state.numel() < 3:
            state = F.pad(state, (0, 3 - state.numel()))

        prev_loss = None

        for i in range(1, self.iterations + 1):
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

    def odometry_step(self, polar_img, timestamp_us):
        with torch.no_grad():
            self.timestamp = float(timestamp_us)
            self.motion_model.setTime(self.timestamp, self.timestamp)
            
            clean_polar = polar_img.detach().clone().float().to(self.device).squeeze()
            base_cart_scan = self.get_cartesian_projection(clean_polar)

            if self.step_counter > 0:
                self.state_init = self.solve(self.state_init, base_cart_scan)
                frame_pos, frame_rot = self.motion_model.getPosRotSingle(self.state_init, self.timestamp)
                
                rot_m = self.get_rotation_matrix(self.current_rot, torch.double)
                self.current_pos += rot_m @ frame_pos.double()
                self.current_rot += frame_rot.double()

                self.move_local_map(frame_pos, frame_rot)
                
                mask_2d = self.sonar_mask
                valid_scan = base_cart_scan > 0
                update_mask = mask_2d & valid_scan
                
                self.local_map[update_mask] = (1 - self.alpha) * self.local_map[update_mask] + self.alpha * base_cart_scan[update_mask]
            else:
                self.local_map = base_cart_scan.clone()

            self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(
                self.local_map.unsqueeze(0).unsqueeze(0), [5, 5], 1.5
            ).squeeze()

            self.step_counter += 1
            return self.state_init.detach().cpu().numpy()

# ==========================================
# ROS 2 Base Class (Inherit from this)
# ==========================================
class BaseSonarOdometryNode(Node):
    """
    Base class for Sonar Odometry. 
    Handles odometry publishing, TF broadcasting, parameter updates, and resetting state.
    """
    def __init__(self, node_name='base_sonar_odometry'):
        super().__init__(node_name)

        # Standard Frame Parameters
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'sonar_link')
        self.declare_parameter('topic_out', '/odom')

        # Odometry Algorithm Parameters
        self.declare_parameter('mapping.update_alpha', 0.5)
        self.declare_parameter('solver.iterations', 15)
        self.declare_parameter('solver.lr', 1.0)
        self.declare_parameter('solver.grad_tol', 1e-6)
        self.declare_parameter('solver.loss_tol', 1e-9)
        self.declare_parameter('solver.max_grad_norm', 10.0)
        self.declare_parameter('solver.lm_damping', 1e-3)

        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        topic_out = self.get_parameter('topic_out').value

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, topic_out, 10)

        # Dynamic Parameter Updates Callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Reset Odometry Service
        self.reset_srv = self.create_service(Trigger, '~/reset_odometry', self.reset_odometry_callback)

        self.estimator = None
        self.current_range = -1.0
        self.current_shape = None

    def parameter_callback(self, params):
        """Allows tuning PyTorch solver & map hyperparameters at runtime."""
        result = SetParametersResult(successful=True)
        for param in params:
            # We apply changes to the estimator directly if it is actively running
            if self.estimator is not None:
                if param.name == 'mapping.update_alpha':
                    self.estimator.alpha = torch.tensor(param.value, device=self.estimator.device)
                elif param.name == 'solver.iterations':
                    self.estimator.iterations = param.value
                elif param.name == 'solver.lr':
                    self.estimator.lr = param.value
                elif param.name == 'solver.grad_tol':
                    self.estimator.grad_tol = param.value
                elif param.name == 'solver.loss_tol':
                    self.estimator.loss_tol = param.value
                elif param.name == 'solver.max_grad_norm':
                    self.estimator.max_grad_norm = param.value
                elif param.name == 'solver.lm_damping':
                    self.estimator.lm_damping = param.value
            
            self.get_logger().info(f"Param Updated -> '{param.name}': {param.value}")
            
        return result

    def reset_odometry_callback(self, request, response):
        """Zeroes out the position and strictly clears the local map to prevent ghost gradients."""
        if self.estimator is not None:
            self.estimator.step_counter = 0
            self.estimator.current_pos.zero_()
            self.estimator.current_rot.zero_()
            self.estimator.state_init = self.estimator.motion_model.getInitialState()
            
            # Wiping the maps is mathematically required so the new origin starts with a clean slate
            self.estimator.local_map.zero_()
            self.estimator.local_map_blurred.zero_()
            
            response.success = True
            response.message = "Odometry state and local maps successfully reset."
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = "Reset failed: Estimator is not initialized yet."
            self.get_logger().warn(response.message)
            
        return response

    def get_quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def initialize_estimator(self, h, w, max_range, fov_degrees):
        self.get_logger().info(f'Initializing Estimator: Range {max_range}m, Shape ({h}, {w}), FOV {fov_degrees}°')
        
        res = float(max_range / h)
        fov_rad = np.radians(fov_degrees)
        angle_bins = np.arange(w)
        beam_angles = (angle_bins / float(w - 1)) * fov_rad - (fov_rad / 2.0)

        opts = {
            'mapping': {
                'max_range': max_range, 
                'res': res, 
                'update_alpha': self.get_parameter('mapping.update_alpha').value
            },
            'solver': {
                'iterations': self.get_parameter('solver.iterations').value, 
                'lr': self.get_parameter('solver.lr').value, 
                'grad_tol': self.get_parameter('solver.grad_tol').value, 
                'loss_tol': self.get_parameter('solver.loss_tol').value, 
                'max_grad_norm': self.get_parameter('solver.max_grad_norm').value, 
                'lm_damping': self.get_parameter('solver.lm_damping').value
            }
        }
        
        self.estimator = SonarStateEstimator(opts, res, beam_angles)
        self.current_range = max_range
        self.current_shape = (h, w)
        
        # JIT Warmup (Pass 0.0 for the scalar timestamp)
        self.get_logger().info('Warming up JIT compiler...')
        dummy_scan = torch.zeros((h, w), dtype=torch.float32, device=self.estimator.device)
        self.estimator.odometry_step(dummy_scan, 0.0)
        self.estimator.odometry_step(dummy_scan, 0.0)
        
        self.estimator.step_counter = 0
        self.estimator.current_pos.zero_()
        self.estimator.current_rot.zero_()
        self.get_logger().info('Warmup complete. Odometry active.')

    def process_sonar_scan(self, scan_tensor, max_range_m, h, w, fov_degrees, stamp=None):
        """
        Main entry point for child classes to pass parsed sonar data.
        """
        if self.estimator is None or self.current_range != max_range_m or self.current_shape != (h, w):
            self.initialize_estimator(h, w, max_range_m, fov_degrees)

        # Process timestamp (Motion model expects microseconds as a scalar float)
        if stamp is not None and (stamp.sec > 0 or stamp.nanosec > 0):
            time_us = (stamp.sec * 1_000_000.0) + (stamp.nanosec / 1_000.0)
        else:
            time_us = 0.0

        # Run odometry optimization with the single global timestamp
        self.estimator.odometry_step(scan_tensor, time_us)

        # Extract Results
        x = float(self.estimator.current_pos[0].cpu())
        y = float(self.estimator.current_pos[1].cpu())
        yaw = float(self.estimator.current_rot.cpu())

        self.publish_odometry(stamp, x, y, yaw)

    def publish_odometry(self, stamp, x, y, yaw):
        q = self.get_quaternion_from_yaw(yaw)

        # Publish Odometry
        odom = Odometry()
        if stamp:
            odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q
        self.odom_pub.publish(odom)

        # Publish TF
        t = TransformStamped()
        if stamp:
            t.header.stamp = stamp
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.child_frame_id
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)