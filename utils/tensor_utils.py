#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
import math

def validate_cuda_tensors(gaussians, operation_name="unknown"):
    """Validate CUDA tensors for memory corruption and invalid values."""
    try:
        with torch.no_grad():
            # Check primary tensors
            xyz = gaussians.get_xyz
            if torch.isnan(xyz).any() or torch.isinf(xyz).any():
                print(f"[ERROR] Invalid values in xyz tensor during {operation_name}")
                return False
            
            opacity = gaussians.get_opacity
            if torch.isnan(opacity).any() or torch.isinf(opacity).any():
                print(f"[ERROR] Invalid values in opacity tensor during {operation_name}")
                return False
            
            # Check auxiliary tensors if they exist
            if hasattr(gaussians, 'xyz_gradient_accum') and gaussians.xyz_gradient_accum is not None:
                if torch.isnan(gaussians.xyz_gradient_accum).any():
                    print(f"[ERROR] NaN in xyz_gradient_accum during {operation_name}")
                    return False
            
            # Check tensor devices
            if xyz.device.type != 'cuda':
                print(f"[ERROR] Tensor not on CUDA during {operation_name}")
                return False
            
            # Memory check
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                if memory_allocated > memory_reserved * 0.95:
                    print(f"[WARNING] CUDA memory almost full during {operation_name}: {memory_allocated/1024**3:.2f}GB")
            
            return True
            
    except Exception as e:
        print(f"[ERROR] CUDA validation failed during {operation_name}: {e}")
        return False

def safe_tensor_operation(func, *args, operation_name="tensor_op", **kwargs):
    """Safely execute tensor operations with error handling and recovery."""
    try:
        # Clear any pending CUDA errors
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        result = func(*args, **kwargs)
        
        # Check for CUDA errors after operation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return result
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "illegal memory access" in str(e):
            print(f"[ERROR] CUDA error in {operation_name}: {e}")
            print(f"[RECOVERY] Attempting to recover...")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Try operation once more with reduced complexity
            try:
                print(f"[RECOVERY] Retrying {operation_name}...")
                result = func(*args, **kwargs)
                print(f"[RECOVERY] Success on retry for {operation_name}")
                return result
            except Exception as e2:
                print(f"[ERROR] Recovery failed for {operation_name}: {e2}")
                raise e  # Re-raise original error
        else:
            raise e

def beta_from_half_life(half_life):
    """
    Compute EMA beta coefficient from half-life.
    
    β = 2^(-1/h) where h is the half-life in iterations.
    
    Args:
        half_life: Number of iterations for the EMA to decay by half
    
    Returns:
        beta: EMA coefficient for exponential moving average
    """
    return math.pow(2.0, -1.0 / half_life)

def cap_gradient_ema(gaussians, max_grad_per_channel=0.01):
    """
    Cap gradient EMA per channel to avoid exploding uncertainty.
    
    Args:
        gaussians: GaussianModel instance
        max_grad_per_channel: Maximum allowed gradient EMA per coordinate
    """
    if hasattr(gaussians, 'xyz_gradient_accum') and gaussians.xyz_gradient_accum is not None:
        # Get current gradient EMA
        grad_ema = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)
        
        # Cap each channel (x, y, z) independently
        grad_ema_capped = torch.clamp(grad_ema, max=max_grad_per_channel)
        
        # Update the accumulated gradients to maintain the capped EMA
        # xyz_gradient_accum = grad_ema_capped * (denom + 1e-8)
        gaussians.xyz_gradient_accum = grad_ema_capped * (gaussians.denom + 1e-8)
        
        # Count how many gradients were capped for logging
        num_capped = (grad_ema > max_grad_per_channel).sum().item()
        
        return num_capped
    
    return 0

def batch_update_gaussian_properties(gaussians, operation_masks, property_updates=None):
    """
    Efficient batched update of Gaussian properties using vectorized operations.
    
    Args:
        gaussians: GaussianModel instance
        operation_masks: Dictionary of boolean masks for different operations
        property_updates: Dictionary of property updates to apply
    """
    with torch.no_grad():
        # Batch opacity updates
        if 'opacity_boost' in operation_masks and property_updates and 'opacity_boost_factor' in property_updates:
            boost_mask = operation_masks['opacity_boost']
            if boost_mask.any():
                current_opacity = gaussians.get_opacity[boost_mask]
                boosted_opacity = torch.clamp(current_opacity * property_updates['opacity_boost_factor'], max=1.0)
                gaussians._opacity[boost_mask] = gaussians.inverse_opacity_activation(boosted_opacity)
        
        # Batch scaling updates
        if 'scaling_reduction' in operation_masks and property_updates and 'scaling_reduction_factor' in property_updates:
            reduction_mask = operation_masks['scaling_reduction']
            if reduction_mask.any():
                current_scaling = gaussians.get_scaling[reduction_mask]
                reduced_scaling = current_scaling * property_updates['scaling_reduction_factor']
                gaussians._scaling[reduction_mask] = gaussians.scaling_inverse_activation(reduced_scaling)

def vectorized_mask_combinations(masks_dict, combination_rules):
    """
    Efficiently combine multiple boolean masks using vectorized operations.
    
    Args:
        masks_dict: Dictionary of named boolean masks
        combination_rules: List of tuples (result_name, operation, mask_names)
            operation can be 'and', 'or', 'not', 'xor'
    
    Returns:
        Dictionary of combined masks
    """
    results = {}
    
    for result_name, operation, mask_names in combination_rules:
        if operation == 'and':
            result = masks_dict[mask_names[0]].clone()
            for name in mask_names[1:]:
                result &= masks_dict[name]
        elif operation == 'or':
            result = masks_dict[mask_names[0]].clone()
            for name in mask_names[1:]:
                result |= masks_dict[name]
        elif operation == 'not':
            result = ~masks_dict[mask_names[0]]
        elif operation == 'xor':
            result = masks_dict[mask_names[0]].clone()
            for name in mask_names[1:]:
                result ^= masks_dict[name]
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        results[result_name] = result
    
    return results