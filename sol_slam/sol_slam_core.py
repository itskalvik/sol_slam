import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import onnxruntime as ort
from dataclasses import dataclass, field
from typing import List

@dataclass
class SonarFrame:
    """Universal data contract for SOL-SLAM hardware frontends."""
    scan_tensor: torch.Tensor
    max_range: float
    fov_degrees: float
    timestamp: float
    height: int
    width: int

@dataclass
class SOLSLAMConfig:
    """Tuning parameters for the SLAM engine."""
    update_alpha: float = 0.3
    is_warped: bool = False
    noise_decay_factor: float = 3.0
    denoise_factor: float = 2.5
    min_intensity_threshold: float = 0.0008
    min_valid_fraction: float = 0.006
    iterations: int = 25
    lr: float = 1.0
    max_grad_norm: float = 10.0
    lm_damping: float = 1e-3
    bound_factor: float = 1.75
    fallback_bound: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.08725])
    blur_kernel_size: List[int] = field(default_factory=lambda: [1, 1])
    blur_sigma: float = 0.0

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

class FullSolverKernel(nn.Module):
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
            
            is_better = loss < best_loss
            best_loss = torch.where(is_better, loss, best_loss)
            best_state = torch.where(is_better, state, best_state)
            
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
            grid, mode='bilinear', padding_mode='zeros', align_corners=True
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
        
        return torch.where(updated_map < cutoff, torch.zeros_like(updated_map), updated_map)

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
        if self.last_time is None: return default_guess.clone().to(self.device)
        dt = current_time - self.last_time
        if dt > 1e-4:
            state_guess[0], state_guess[1], state_guess[2] = self.vx * dt, self.vy * dt, self.vth * dt
        else:
            state_guess = default_guess.clone().to(self.device)
        return state_guess

    def update(self, frame_pos, frame_rot, current_time):
        dt = current_time - self.last_time if self.last_time else 0
        if dt > 1e-4:
            self.vx, self.vy, self.vth = frame_pos[0].item() / dt, frame_pos[1].item() / dt, frame_rot.item() / dt
        self.last_time = current_time
        rot_m = get_rotation_matrix(self.rot, dtype=self.dtype, device=self.device)
        self.pos += rot_m @ frame_pos
        self.rot += frame_rot

class SOLSLAMCore:
    def __init__(self, config: SOLSLAMConfig, res: float, h: int, w: int, enable_tensorrt: bool = False, max_range: float = 12.0, fov_degrees: float = 130.0):
        with torch.no_grad():
            self.device = get_best_device()
            self.dtype = torch.float32
            self.enable_tensorrt = enable_tensorrt and self.device.type == 'cuda'
            self.config = config
            
            self.sonar_res = torch.tensor(res, dtype=self.dtype, device=self.device)
            self.step_counter = 0
            self.state = torch.zeros(3, dtype=self.dtype, device=self.device)
            self.motion_model = RobotMotionModel(dtype=self.dtype, device=self.device)

            self.H_img, self.W_img = h, w
            self.map_res = float(max_range / h)
            self.max_range = float(max_range)
            self.alpha = torch.tensor(self.config.update_alpha, dtype=self.dtype, device=self.device)
            self.noise_decay_factor_tensor = torch.tensor(self.config.noise_decay_factor, dtype=self.dtype, device=self.device)

            self.fov_rad = torch.tensor(np.radians(fov_degrees), dtype=self.dtype, device=self.device)
            self.half_width_phys = self.max_range * torch.sin(self.fov_rad / 2.0)
            
            angle_bins = torch.arange(w, dtype=self.dtype, device=self.device)
            self.beam_angles = (angle_bins / float(w - 1)) * self.fov_rad - (self.fov_rad / 2.0)
            self.map_size = int(self.max_range / self.map_res) * 2 + 1
            
            self.fallback_bound = torch.tensor(self.config.fallback_bound, dtype=self.dtype, device=self.device)

            self.local_map = torch.zeros((self.map_size, self.map_size), dtype=self.dtype, device=self.device)
            self.local_map_blurred = torch.zeros_like(self.local_map)
            self.last_cart_scan = None
            
            # Persistent buffers
            self.M_x_buf = torch.zeros_like(self.local_map)
            self.M_y_buf = torch.zeros_like(self.local_map)
            
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
            
            if self.config.is_warped:
                norm_y = 1.0 - 2.0 * (x_safe / self.max_range)
                norm_x = -(y_safe / self.half_width_phys)
                self.cart_proj_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).to(self.device)
            else:
                a_idx = (self.theta - a_min) / (a_max - a_min) * (self.W_img - 1)
                r_idx = self.r / self.sonar_res
                norm_a = (a_idx / (self.W_img - 1) * 2) - 1
                norm_r = (r_idx / (self.H_img - 1) * 2) - 1
                self.cart_proj_grid = torch.stack([norm_a, norm_r], dim=-1).unsqueeze(0).to(self.device)

            self.solver_kernel = FullSolverKernel(self.map_res, self.map_size, self.config.iterations, self.config.lr, self.config.max_grad_norm, self.N_valid_tensor.item()).to(self.device)
            self.map_kernel = MapUpdateKernel(self.map_res, self.map_size).to(self.device)

            if self.enable_tensorrt:
                self._setup_tensorrt()
            else:
                # Fallback to torch.compile for PyTorch 2.x performance gains
                if hasattr(torch, 'compile'):
                    self.solver_kernel = torch.compile(self.solver_kernel, mode="reduce-overhead")
                    self.map_kernel = torch.compile(self.map_kernel, mode="reduce-overhead")

    def _setup_tensorrt(self):
        shape_id = f"R{self.max_range:.1f}_H{self.H_img}_W{self.W_img}"
        solver_onnx_path = f"/tmp/sol_slam_solver_{shape_id}.onnx"
        map_onnx_path = f"/tmp/sol_slam_map_{shape_id}.onnx"
        
        trt_providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0, 'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': False, 'trt_engine_cache_enable': True,
                'trt_engine_cache_path': f'/tmp/trt_cache_{shape_id}'
            }), ('CUDAExecutionProvider', {'device_id': 0})
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

        if not os.path.exists(solver_onnx_path):
            torch.onnx.export(
                self.solver_kernel, 
                (self.trt_state_init, self.trt_base_cart_scan, self.trt_J_flat, self.trt_M_flat, self.trt_H_inv, self.map_coords, self.flat_indices),
                solver_onnx_path, opset_version=16,
                input_names=['state_init', 'base_cart_scan', 'J_flat', 'M_flat', 'H_inv', 'map_coords', 'flat_indices'],
                output_names=['final_state'], do_constant_folding=False 
            )
        self.ort_solver_session = ort.InferenceSession(solver_onnx_path, providers=trt_providers)
        self.io_binding_solver = self.ort_solver_session.io_binding()
        for name, tensor in zip(['state_init', 'base_cart_scan', 'J_flat', 'M_flat', 'H_inv', 'map_coords', 'flat_indices'],
                                [self.trt_state_init, self.trt_base_cart_scan, self.trt_J_flat, self.trt_M_flat, self.trt_H_inv, self.map_coords, self.flat_indices]):
            self._bind_tensor(self.io_binding_solver, name, tensor)
        self._bind_tensor(self.io_binding_solver, "final_state", self.trt_final_state, is_input=False)

        if not os.path.exists(map_onnx_path):
            torch.onnx.export(
                self.map_kernel,
                (self.trt_delta_pos, self.trt_delta_rot, self.local_map, self.trt_base_cart_scan, self.map_coords, self.sonar_mask, self.trt_alpha, self.trt_noise_decay),
                map_onnx_path, opset_version=16,
                input_names=['delta_pos', 'delta_rot', 'local_map', 'base_cart_scan', 'map_coords', 'sonar_mask', 'alpha', 'noise_decay_factor'],
                output_names=['final_map'], do_constant_folding=False
            )
        self.ort_map_session = ort.InferenceSession(map_onnx_path, providers=trt_providers)
        self.io_binding_map = self.ort_map_session.io_binding()
        for name, tensor in zip(['delta_pos', 'delta_rot', 'local_map', 'base_cart_scan', 'map_coords', 'sonar_mask', 'alpha', 'noise_decay_factor'],
                                [self.trt_delta_pos, self.trt_delta_rot, self.local_map, self.trt_base_cart_scan, self.map_coords, self.sonar_mask, self.trt_alpha, self.trt_noise_decay]):
            self._bind_tensor(self.io_binding_map, name, tensor)
        self._bind_tensor(self.io_binding_map, "final_map", self.trt_final_map, is_input=False)

    def _bind_tensor(self, binding_obj, name, tensor, is_input=True):
        device_id = tensor.device.index if tensor.device.index is not None else 0
        elem_type = np.float32 if tensor.dtype == torch.float32 else np.int32 if tensor.dtype == torch.int32 else np.bool_
        bind_kwargs = {'name': name, 'device_type': 'cuda', 'device_id': device_id, 'element_type': elem_type, 'shape': tuple(tensor.shape), 'buffer_ptr': tensor.data_ptr()}
        if is_input: binding_obj.bind_input(**bind_kwargs)
        else: binding_obj.bind_output(**bind_kwargs)

    def preprocess_sonar(self, scan_img):
        scan = scan_img.clone().detach().to(device=self.device, dtype=self.dtype).squeeze()
        if scan.max() > 2.0: scan /= 255.0
        scan[scan < scan.mean() + self.config.denoise_factor * scan.std()] = 0.0
        return scan

    def get_cartesian_projection(self, scan_img):
        cart = F.grid_sample(scan_img.view(1, 1, self.H_img, self.W_img), self.cart_proj_grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        return torch.where(self.sonar_mask, cart, 0.0)

    def solve(self, state_init, base_cart_scan):
        res = self.map_res
        M = self.local_map_blurred.detach()
        
        self.M_x_buf.zero_()
        self.M_y_buf.zero_()
        self.M_x_buf[:, 1:-1] = (M[:, 2:] - M[:, :-2]) / (2 * res)
        self.M_y_buf[1:-1, :] = (M[2:, :] - M[:-2, :]) / (2 * res)
        
        M_theta = -self.map_coords[..., 1] * self.M_x_buf + self.map_coords[..., 0] * self.M_y_buf

        J_flat = torch.stack([self.M_x_buf.view(-1)[self.flat_indices], self.M_y_buf.view(-1)[self.flat_indices], M_theta.view(-1)[self.flat_indices]]).to(self.device)
        M_flat = M.view(-1)[self.flat_indices]

        H = (J_flat @ J_flat.T) / self.N_valid_tensor.item()
        H += torch.eye(3, dtype=self.dtype, device=self.device) * self.config.lm_damping
        H_inv = torch.linalg.inv(H)

        if self.enable_tensorrt:
            self.trt_state_init.copy_(state_init.flatten())
            self.trt_base_cart_scan.copy_(base_cart_scan)
            self.trt_J_flat.copy_(J_flat)
            self.trt_M_flat.copy_(M_flat)
            self.trt_H_inv.copy_(H_inv)
            torch.cuda.synchronize()
            self.ort_solver_session.run_with_iobinding(self.io_binding_solver)
            torch.cuda.synchronize()
            return self.trt_final_state.clone()
        else:
            # Clone the output to detach it from the compiled graph's static memory
            return self.solver_kernel(state_init.flatten(), base_cart_scan, J_flat, M_flat, H_inv, self.map_coords, self.flat_indices).clone()

    def process_frame(self, frame: SonarFrame):
        with torch.no_grad():
            # Signal the start of a new step for CUDA Graphs
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()

            clean_scan = self.preprocess_sonar(frame.scan_tensor)
            base_cart_scan = self.get_cartesian_projection(clean_scan)

            active_pixels_count = (base_cart_scan[self.sonar_mask] > self.config.min_intensity_threshold).sum().float()
            if (active_pixels_count / self.N_valid_tensor.item()).item() < self.config.min_valid_fraction:
                self.motion_model.last_time = frame.timestamp
                self.motion_model.vx, self.motion_model.vy, self.motion_model.vth = 0.0, 0.0, 0.0
                return False 

            if self.step_counter > 0:
                state_guess = self.motion_model.get_state_guess(frame.timestamp, default_guess=self.state)
                proposed_state = self.solve(state_guess, base_cart_scan)
                
                if torch.isnan(proposed_state).any() or (torch.abs(proposed_state) > torch.max(torch.abs(state_guess) * self.config.bound_factor, self.fallback_bound)).any():
                    self.state = state_guess 
                else:
                    self.state = proposed_state

                f_pos, f_rot = self.state[0:2], self.state[2]
                self.motion_model.update(f_pos, f_rot, frame.timestamp)
                
                if torch.norm(f_pos) > 1e-4 or torch.abs(f_rot) > 1e-5:
                    if self.enable_tensorrt:
                        self.trt_delta_pos.copy_(f_pos)
                        self.trt_delta_rot[0] = f_rot
                        self.trt_base_cart_scan.copy_(base_cart_scan)
                        torch.cuda.synchronize()
                        self.ort_map_session.run_with_iobinding(self.io_binding_map)
                        torch.cuda.synchronize()
                        self.local_map.copy_(self.trt_final_map)
                    else:
                        # Clone the updated map to prevent cross-generation overwrites
                        self.local_map = self.map_kernel(f_pos, f_rot.view(1), self.local_map, base_cart_scan, self.map_coords, self.sonar_mask, self.alpha, self.noise_decay_factor_tensor).clone()
            else:
                self.local_map.copy_(base_cart_scan)
                self.motion_model.last_time = frame.timestamp

            if self.config.blur_kernel_size[0] > 1:
                self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(self.local_map.unsqueeze(0).unsqueeze(0), self.config.blur_kernel_size, self.config.blur_sigma).squeeze()
            else:
                self.local_map_blurred = self.local_map

            self.step_counter += 1
            self.last_cart_scan = base_cart_scan
            return True