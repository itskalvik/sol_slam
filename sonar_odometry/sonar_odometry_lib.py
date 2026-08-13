import os
import rclpy
from rclpy.node import Node
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import onnxruntime as ort

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped, Point
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

# ==========================================
# Helper Functions & Device Management
# ==========================================
def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_rotation_matrix(rot, dtype=torch.float32, device='cpu'):
    c, s = torch.cos(rot), torch.sin(rot)
    rot_mat = torch.empty((2, 2), dtype=dtype, device=device)
    rot_mat[0, 0] = c; rot_mat[0, 1] = -s
    rot_mat[1, 0] = s; rot_mat[1, 1] = c
    return rot_mat

def unwarp_sonar(warped_image, fov=130, num_beams=768):
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
# TensorRT Kernels
# ==========================================
class FullSolverKernel(nn.Module):
    """
    Computes all iterations and returns the state that achieved the lowest loss.
    This guarantees mathematically stable alignment even if the solver diverges later.
    """
    def __init__(self, map_res, map_size, iterations, lr, max_grad_norm, N_valid):
        super().__init__()
        self.map_res = float(map_res)
        self.map_size = int(map_size)
        self.half_extent = self.map_res * (self.map_size - 1) / 2.0
        
        self.iterations = int(iterations)
        self.lr = float(lr)
        self.max_grad_norm = float(max_grad_norm)
        self.N_valid = float(N_valid)

    def forward(self, state_init, base_cart_scan, J_flat, M_flat, H_inv, map_coords, flat_indices):
        state = state_init.clone()
        
        best_state = state.clone()
        best_loss = torch.tensor(float('inf'), dtype=state.dtype, device=state.device)
        
        for _ in range(self.iterations):
            tx, ty, theta = state[0], state[1], state[2]
            cos_r, sin_r = torch.cos(theta), torch.sin(theta)
            
            x_shifted = map_coords[..., 0] - tx
            y_shifted = map_coords[..., 1] - ty
            
            x_old = x_shifted * cos_r + y_shifted * sin_r
            y_old = -x_shifted * sin_r + y_shifted * cos_r
            
            norm_x = x_old / self.half_extent
            norm_y = y_old / self.half_extent
            
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
            
            warped_scan = F.grid_sample(
                base_cart_scan.view(1, 1, self.map_size, self.map_size), 
                grid, align_corners=True, padding_mode='zeros'
            ).squeeze()
            
            warped_flat = warped_scan.view(-1)[flat_indices]
            
            grad = -(J_flat @ warped_flat) / self.N_valid
            loss = -torch.dot(M_flat, warped_flat) / self.N_valid
            
            # --- Trajectory Selection Logic ---
            is_better = loss < best_loss
            best_loss = torch.where(is_better, loss, best_loss)
            best_state = torch.where(is_better, state, best_state)
            
            # --- Continue the Optimization Path ---
            grad_norm = torch.linalg.vector_norm(grad)
            grad_norm_safe = torch.clamp(grad_norm, min=1e-8)
            clip_factor = torch.clamp(self.max_grad_norm / grad_norm_safe, max=1.0)
            grad = grad * clip_factor
            
            delta_q = torch.matmul(H_inv, grad)
            state = state - self.lr * delta_q
            
        return best_state


class MapUpdateKernel(nn.Module):
    def __init__(self, map_res, map_size):
        super().__init__()
        self.map_res = map_res
        self.map_size = map_size

    def forward(self, delta_pos, delta_rot, local_map, base_cart_scan, map_coords, sonar_mask, alpha, noise_decay_factor):
        cos_r, sin_r = torch.cos(delta_rot), torch.sin(delta_rot)
        
        x = map_coords[..., 0]
        y = map_coords[..., 1]
        
        x_shifted = x * cos_r - y * sin_r + delta_pos[0]
        y_shifted = x * sin_r + y * cos_r + delta_pos[1]
        
        pixel_x = (x_shifted / self.map_res) + (self.map_size // 2)
        pixel_y = (y_shifted / self.map_res) + (self.map_size // 2)
        
        norm_x = (pixel_x / (self.map_size - 1) * 2) - 1
        norm_y = (pixel_y / (self.map_size - 1) * 2) - 1
        
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
        
        shifted_map = F.grid_sample(
            local_map.view(1, 1, self.map_size, self.map_size),
            grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze()

        valid_scan = base_cart_scan > 0
        update_mask = sonar_mask & valid_scan
        
        one_tensor = torch.tensor(1.0, dtype=alpha.dtype, device=alpha.device)
        updated_map = torch.where(
            update_mask,
            (one_tensor - alpha) * shifted_map + alpha * base_cart_scan,
            shifted_map
        )
        
        mean_val = updated_map.mean()
        std_val = updated_map.std()
        cutoff = mean_val + noise_decay_factor * std_val
        
        final_map = torch.where(updated_map < cutoff, torch.zeros_like(updated_map), updated_map)
        return final_map


# ==========================================
# Robot Motion Model
# ==========================================
class RobotMotionModel:
    def __init__(self, dtype=torch.float32, device='cpu'):
        self.dtype = dtype
        self.device = device
        self.pos = torch.zeros(2, dtype=self.dtype, device=self.device)
        self.rot = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.vx, self.vy, self.vth = 0.0, 0.0, 0.0
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
    def __init__(self, opts, res, h, w, node_logger):
        with torch.no_grad():
            self.device = get_best_device()
            self.dtype = torch.float32
            self.logger = node_logger
            self.logger.info(f"SonarStateEstimator initializing on device: {self.device.type.upper()}")
            
            self.sonar_res = torch.tensor(res, dtype=self.dtype, device=self.device)
            self.step_counter = 0

            self.state = torch.zeros(3, dtype=self.dtype, device=self.device)
            self.motion_model = RobotMotionModel(dtype=self.dtype, device=self.device)

            self.H_img = h
            self.W_img = w
            self.map_res = float(opts['mapping']['res'])
            self.max_range = float(opts['mapping']['max_range'])
            self.alpha = torch.tensor(opts['mapping']['update_alpha'], dtype=self.dtype, device=self.device)
            self.is_warped = bool(opts['mapping'].get('is_warped', False))
            self.noise_decay_factor_tensor = torch.tensor(opts['mapping'].get('noise_decay_factor', 4.0), dtype=self.dtype, device=self.device)
            
            self.blur_kernel_size = opts['mapping'].get('blur_kernel_size', [1, 1])
            self.blur_sigma = float(opts['mapping'].get('blur_sigma', 0.0))
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
            
            solver_opts = opts.get('solver', {})
            self.iterations = int(solver_opts.get('iterations', 15)) 
            self.lr = float(solver_opts.get('lr', 1.0)) 
            self.max_grad_norm = float(solver_opts.get('max_grad_norm', 10.0))
            self.lm_damping = float(solver_opts.get('lm_damping', 1e-3)) 
            self.bound_factor = float(solver_opts.get('bound_factor', 1.75))
            self.fallback_bound = torch.tensor(solver_opts.get('fallback_bound', [0.25, 0.25, 0.08725]), dtype=self.dtype, device=self.device)

            self.local_map = torch.zeros((self.map_size, self.map_size), dtype=self.dtype, device=self.device)
            self.local_map_blurred = torch.zeros_like(self.local_map)
            self.last_cart_scan = None
            
            axis = (torch.arange(self.map_size, dtype=self.dtype, device=self.device) - self.map_size // 2) * self.map_res
            grid_y, grid_x = torch.meshgrid(axis, axis, indexing='ij')
            self.map_coords = torch.stack([grid_x, grid_y], dim=-1).to(self.device).contiguous()

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
            self.flat_indices = torch.nonzero(self.flat_mask).squeeze().to(torch.int32).contiguous()
            self.N_valid_tensor = torch.tensor([self.flat_mask.sum().item()], dtype=self.dtype, device=self.device)
            
            if self.is_warped:
                norm_y = 1.0 - 2.0 * (x_safe / self.max_range)
                norm_x = -(y_safe / self.half_width_phys)
                self.cart_proj_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).to(self.device)
            else:
                a_idx = (self.theta - a_min) / (a_max - a_min) * (self.W_img - 1)
                r_idx = self.r / self.sonar_res
                norm_a = (a_idx / (self.W_img - 1) * 2) - 1
                norm_r = (r_idx / (self.H_img - 1) * 2) - 1
                self.cart_proj_grid = torch.stack([norm_a, norm_r], dim=-1).unsqueeze(0).to(self.device)

            self._setup_tensorrt()

    def _setup_tensorrt(self):
        # Create a unique ID for the current sonar resolution and range
        shape_id = f"R{self.max_range:.1f}_H{self.H_img}_W{self.W_img}"
        
        # Bake the shape ID into the filenames so auto-ranging creates separate engines
        solver_onnx_path = f"/tmp/sonar_full_solver_{shape_id}.onnx"
        map_onnx_path = f"/tmp/sonar_map_kernel_{shape_id}.onnx"
        
        trt_providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': False, 
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': f'/tmp/trt_cache_{shape_id}' # Unique cache per range
            }),
            ('CUDAExecutionProvider', {'device_id': 0})
        ]

        self.trt_state_init = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.trt_base_cart_scan = torch.zeros((self.map_size, self.map_size), dtype=self.dtype, device=self.device)
        self.trt_J_flat = torch.zeros((3, self.flat_indices.shape[0]), dtype=self.dtype, device=self.device)
        self.trt_M_flat = torch.zeros(self.flat_indices.shape[0], dtype=self.dtype, device=self.device)
        self.trt_H_inv = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        self.trt_final_state = torch.zeros(3, dtype=self.dtype, device=self.device)

        self.trt_delta_pos = torch.zeros(2, dtype=self.dtype, device=self.device)
        self.trt_delta_rot = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.trt_alpha = torch.tensor([self.alpha.item()], dtype=self.dtype, device=self.device)
        self.trt_noise_decay = torch.tensor([self.noise_decay_factor_tensor.item()], dtype=self.dtype, device=self.device)
        self.trt_final_map = torch.zeros((self.map_size, self.map_size), dtype=self.dtype, device=self.device)

        # 1. Full Solver Engine
        if not os.path.exists(solver_onnx_path):
            self.logger.info(f"Exporting full solver loop to ONNX for shape {shape_id}...")
            dummy_state = torch.randn(3, dtype=self.dtype, device=self.device)
            dummy_scan = torch.rand((self.map_size, self.map_size), dtype=self.dtype, device=self.device)
            dummy_J = torch.randn((3, self.flat_indices.shape[0]), dtype=self.dtype, device=self.device)
            dummy_M = torch.randn(self.flat_indices.shape[0], dtype=self.dtype, device=self.device)
            dummy_H_inv = torch.eye(3, dtype=self.dtype, device=self.device)

            model = FullSolverKernel(
                self.map_res, self.map_size, self.iterations, self.lr, 
                self.max_grad_norm, self.N_valid_tensor.item()
            ).to(self.device)
            
            torch.onnx.export(
                model, 
                (dummy_state, dummy_scan, dummy_J, dummy_M, dummy_H_inv, self.map_coords, self.flat_indices),
                solver_onnx_path, opset_version=16,
                input_names=['state_init', 'base_cart_scan', 'J_flat', 'M_flat', 'H_inv', 'map_coords', 'flat_indices'],
                output_names=['final_state'], 
                do_constant_folding=False 
            )

        self.ort_solver_session = ort.InferenceSession(solver_onnx_path, providers=trt_providers)
        self.io_binding_solver = self.ort_solver_session.io_binding()
        
        self._bind_tensor(self.io_binding_solver, "state_init", self.trt_state_init)
        self._bind_tensor(self.io_binding_solver, "base_cart_scan", self.trt_base_cart_scan)
        self._bind_tensor(self.io_binding_solver, "J_flat", self.trt_J_flat)
        self._bind_tensor(self.io_binding_solver, "M_flat", self.trt_M_flat)
        self._bind_tensor(self.io_binding_solver, "H_inv", self.trt_H_inv)
        self._bind_tensor(self.io_binding_solver, "map_coords", self.map_coords)
        self._bind_tensor(self.io_binding_solver, "flat_indices", self.flat_indices)
        self._bind_tensor(self.io_binding_solver, "final_state", self.trt_final_state, is_input=False)

        # 2. Map Update Engine
        if not os.path.exists(map_onnx_path):
            self.logger.info(f"Exporting map update kernel to ONNX for shape {shape_id}...")
            dummy_pos = torch.randn(2, dtype=self.dtype, device=self.device)
            dummy_rot = torch.randn(1, dtype=self.dtype, device=self.device) 
            dummy_scan = torch.rand((self.map_size, self.map_size), dtype=self.dtype, device=self.device)
            
            map_model = MapUpdateKernel(self.map_res, self.map_size).to(self.device)
            torch.onnx.export(
                map_model,
                (dummy_pos, dummy_rot, self.local_map, dummy_scan, self.map_coords, self.sonar_mask, self.trt_alpha, self.trt_noise_decay),
                map_onnx_path, opset_version=16,
                input_names=['delta_pos', 'delta_rot', 'local_map', 'base_cart_scan', 'map_coords', 'sonar_mask', 'alpha', 'noise_decay_factor'],
                output_names=['final_map'], do_constant_folding=False
            )
            
        self.ort_map_session = ort.InferenceSession(map_onnx_path, providers=trt_providers)
        self.io_binding_map = self.ort_map_session.io_binding()
        
        self._bind_tensor(self.io_binding_map, "delta_pos", self.trt_delta_pos)
        self._bind_tensor(self.io_binding_map, "delta_rot", self.trt_delta_rot)
        self._bind_tensor(self.io_binding_map, "local_map", self.local_map)
        self._bind_tensor(self.io_binding_map, "base_cart_scan", self.trt_base_cart_scan)
        self._bind_tensor(self.io_binding_map, "map_coords", self.map_coords)
        self._bind_tensor(self.io_binding_map, "sonar_mask", self.sonar_mask)
        self._bind_tensor(self.io_binding_map, "alpha", self.trt_alpha)
        self._bind_tensor(self.io_binding_map, "noise_decay_factor", self.trt_noise_decay)
        self._bind_tensor(self.io_binding_map, "final_map", self.trt_final_map, is_input=False)

    def _bind_tensor(self, binding_obj, name, tensor, is_input=True):
        device_id = tensor.device.index if tensor.device.index is not None else 0
        if tensor.dtype == torch.float32: elem_type = np.float32
        elif tensor.dtype == torch.bool: elem_type = np.bool_
        elif tensor.dtype == torch.int32: elem_type = np.int32
        else: elem_type = np.int64
            
        bind_kwargs = {
            'name': name, 'device_type': 'cuda', 'device_id': device_id,
            'element_type': elem_type,
            'shape': tuple(tensor.shape), 'buffer_ptr': tensor.data_ptr()
        }
        if is_input: binding_obj.bind_input(**bind_kwargs)
        else: binding_obj.bind_output(**bind_kwargs)

    def preprocess_sonar(self, scan_img):
        scan = scan_img.clone().detach().to(device=self.device, dtype=self.dtype).squeeze()
        if scan.max() > 2.0: 
            scan /= 255.0
        scan[scan < scan.mean() + self.denoise_factor * scan.std()] = 0.0
        return scan

    def get_cartesian_projection(self, scan_img):
        cart = F.grid_sample(
            scan_img.view(1, 1, self.H_img, self.W_img), 
            self.cart_proj_grid, 
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze()
        return torch.where(self.sonar_mask, cart, 0.0)

    def solve(self, state_init):
        res = self.map_res
        M = self.local_map_blurred.detach()
        
        M_x = torch.zeros_like(M, dtype=self.dtype, device=self.device)
        M_y = torch.zeros_like(M, dtype=self.dtype, device=self.device)
        M_x[:, 1:-1] = (M[:, 2:] - M[:, :-2]) / (2 * res)
        M_y[1:-1, :] = (M[2:, :] - M[:-2, :]) / (2 * res)
        
        x_grid = self.map_coords[..., 0]
        y_grid = self.map_coords[..., 1]
        M_theta = -y_grid * M_x + x_grid * M_y

        M_x_valid = M_x.view(-1)[self.flat_indices]
        M_y_valid = M_y.view(-1)[self.flat_indices]
        M_theta_valid = M_theta.view(-1)[self.flat_indices]
        
        J_flat = torch.stack([M_x_valid, M_y_valid, M_theta_valid]).to(self.device)
        M_flat = M.view(-1)[self.flat_indices]

        H = (J_flat @ J_flat.T) / self.N_valid_tensor.item()
        H += torch.eye(3, dtype=self.dtype, device=self.device) * self.lm_damping
        H_inv = torch.linalg.inv(H)

        self.trt_state_init.copy_(state_init.flatten())
        self.trt_J_flat.copy_(J_flat)
        self.trt_M_flat.copy_(M_flat)
        self.trt_H_inv.copy_(H_inv)

        torch.cuda.synchronize()
        self.ort_solver_session.run_with_iobinding(self.io_binding_solver)
        torch.cuda.synchronize()

        return self.trt_final_state.clone()

    def odometry_step(self, scan_img, current_time):
        with torch.no_grad():           
            clean_scan = self.preprocess_sonar(scan_img)
            
            base_cart_scan = self.get_cartesian_projection(clean_scan)
            self.trt_base_cart_scan.copy_(base_cart_scan)

            footprint_pixels = base_cart_scan[self.sonar_mask]
            active_pixels_count = (footprint_pixels > self.min_intensity_threshold).sum().float()
            active_fraction = (active_pixels_count / self.N_valid_tensor.item()).item()

            if active_fraction < self.min_valid_fraction:
                self.logger.warn(f"Empty scan detected. Skipping optimization.", throttle_duration_sec=1.0)
                self.motion_model.last_time = current_time
                self.motion_model.vx, self.motion_model.vy, self.motion_model.vth = 0.0, 0.0, 0.0
                return

            if self.step_counter > 0:
                state_guess = self.motion_model.get_state_guess(current_time, default_guess=self.state)
                proposed_state = self.solve(state_guess)
                
                is_nan = torch.isnan(proposed_state).any().item()
                
                current_bound = self.fallback_bound
                if torch.norm(state_guess) > 1e-5:
                    dynamic_bound = torch.abs(state_guess) * self.bound_factor
                    current_bound = torch.max(dynamic_bound, self.fallback_bound)

                if is_nan or \
                   (torch.abs(proposed_state[0]) > current_bound[0]) or \
                   (torch.abs(proposed_state[1]) > current_bound[1]) or \
                   (torch.abs(proposed_state[2]) > current_bound[2]):
                    
                    self.logger.warn("Optimization Rejected. Divergence or bounds exceeded.", throttle_duration_sec=1.0)
                    self.state = state_guess 
                else:
                    self.state = proposed_state

                frame_pos, frame_rot = self.state[0:2], self.state[2]
                self.motion_model.update(frame_pos, frame_rot, current_time)
                
                if torch.norm(frame_pos) > 1e-4 or torch.abs(frame_rot) > 1e-5:
                    self.trt_delta_pos.copy_(frame_pos)
                    self.trt_delta_rot[0] = frame_rot
                    
                    torch.cuda.synchronize()
                    self.ort_map_session.run_with_iobinding(self.io_binding_map)
                    torch.cuda.synchronize()
                    
                    self.local_map.copy_(self.trt_final_map)

            else:
                self.local_map.copy_(base_cart_scan)
                self.motion_model.last_time = current_time

            if self.blur_kernel_size[0] > 1:
                self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(
                    self.local_map.unsqueeze(0).unsqueeze(0), self.blur_kernel_size, self.blur_sigma
                ).squeeze()
            else:
                self.local_map_blurred = self.local_map

            self.step_counter += 1
            self.last_cart_scan = base_cart_scan


# ==========================================
# ROS 2 Base Class
# ==========================================
class BaseSonarOdometryNode(Node):
    def __init__(self, node_name='base_sonar_odometry'):
        super().__init__(node_name)

        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('topic_out', '/odom')
        self.declare_parameter('path_topic', '/sonar_path')

        self.declare_parameter('scan_img_topic', '/sonar_scan_img')
        self.declare_parameter('map_img_topic', '/sonar_map_img')
        self.declare_parameter('peaks_topic', '/sonar_peaks')
        self.declare_parameter('contrast', 10.0)

        self.declare_parameter('enable_debug_viz', False)

        self.declare_parameter('mapping.update_alpha', 0.3)
        self.declare_parameter('mapping.is_warped', False)
        self.declare_parameter('mapping.noise_decay_factor', 3.0)
        self.declare_parameter('mapping.denoise_factor', 2.5)
        self.declare_parameter('mapping.min_intensity_threshold', 0.0008)
        self.declare_parameter('mapping.min_valid_fraction', 0.006)

        self.declare_parameter('solver.iterations', 15)
        self.declare_parameter('solver.lr', 1.0)
        self.declare_parameter('solver.max_grad_norm', 10.0)
        self.declare_parameter('solver.lm_damping', 1e-3)
        self.declare_parameter('solver.bound_factor', 1.75)
        
        # Ignored by TRT, but declared for backward compatibility in launch files
        self.declare_parameter('solver.grad_tol', 1e-6)
        self.declare_parameter('solver.loss_tol', 1e-9)

        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.contrast = self.get_parameter('contrast').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('topic_out').value, 10)
        self.path_pub = self.create_publisher(Path, self.get_parameter('path_topic').value, 10)
        
        if self.enable_debug_viz:
            self.peaks_pub = self.create_publisher(PointCloud2, self.get_parameter('peaks_topic').value, 10)
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
            elif param.name == 'enable_debug_viz':
                self.enable_debug_viz = param.value
            if self.estimator is not None:
                if param.name == 'mapping.update_alpha':
                    self.estimator.alpha = torch.tensor(param.value, device=self.estimator.device)
                elif param.name == 'solver.lr':
                    self.estimator.lr = param.value
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
                'max_grad_norm': self.get_parameter('solver.max_grad_norm').value, 
                'lm_damping': self.get_parameter('solver.lm_damping').value,
                'bound_factor': self.get_parameter('solver.bound_factor').value,
                'fallback_bound': [0.25, 0.25, 0.08725]
            }
        }
        
        self.estimator = SonarStateEstimator(opts, res, h, w, self.get_logger())
        self.current_range = max_range
        self.current_shape = (h, w)
        
        self.estimator.step_counter = 0
        self.estimator.state.zero_()
        self.estimator.motion_model.reset()
        self.estimator.local_map.zero_()

    def process_sonar_scan(self, scan_tensor, max_range_m, h, w, fov_degrees, stamp):
        if self.estimator is None or self.current_range != max_range_m or self.current_shape != (h, w):
            self.initialize_estimator(h, w, max_range_m, fov_degrees)

        current_time = stamp.sec + (stamp.nanosec * 1e-9) if stamp else 0.0
        
        scan_tensor = scan_tensor.to(self.estimator.device)
        self.estimator.odometry_step(scan_tensor, current_time)

        pos_x = self.estimator.motion_model.pos[0].item()
        pos_y = -self.estimator.motion_model.pos[1].item()
        rot_theta = -self.estimator.motion_model.rot.item()
        
        vx = self.estimator.motion_model.vx
        vy = -self.estimator.motion_model.vy
        vth = -self.estimator.motion_model.vth

        self.publish_odometry(stamp, pos_x, pos_y, rot_theta, vx, vy, vth)
        
        if self.enable_debug_viz:
            self.publish_debug_images(stamp)
            self.publish_peaks(stamp)

    def publish_peaks(self, stamp):
        if not hasattr(self.estimator, 'local_map') or self.estimator.local_map is None:
            return

        try:
            map_tensor = self.estimator.local_map
            threshold = self.get_parameter('mapping.min_intensity_threshold').value
            
            mask = map_tensor > threshold
            if not mask.any():
                return
            
            valid_coords = self.estimator.map_coords[mask].cpu().numpy()
            valid_intensities = map_tensor[mask].cpu().numpy()
            
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
            if stamp: 
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
            cloud_msg.row_step = cloud_msg.point_step * num_points
            cloud_msg.is_dense = True
            cloud_msg.data = points_np.tobytes()

            self.peaks_pub.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish local map point cloud: {e}")

    def publish_debug_images(self, stamp):
        if not hasattr(self.estimator, 'local_map') or getattr(self.estimator, 'last_cart_scan', None) is None:
            return

        try:
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

            scan_array = self.estimator.last_cart_scan.detach().cpu().numpy()
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