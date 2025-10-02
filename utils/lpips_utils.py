#
# Copyright (C) 2023, Inria  
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

try:
    import lpips
    LPIPS_AVAILABLE = True
    
    # Initialize LPIPS model (lazy loading)
    _lpips_model = None
    
    def get_lpips_model(device='cuda'):
        """Get or initialize LPIPS model."""
        global _lpips_model
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net='alex').to(device)
            _lpips_model.eval()
        return _lpips_model
    
    def compute_lpips(img1, img2, device='cuda'):
        """
        Compute LPIPS between two images.
        
        Args:
            img1, img2: Tensors of shape [C, H, W] with values in [0, 1]
            device: Device to run computation on
            
        Returns:
            LPIPS value as float
        """
        try:
            model = get_lpips_model(device)
            
            # Ensure images are in correct format
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)  # Add batch dimension
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)  # Add batch dimension
            
            # Ensure values are in [-1, 1] range (LPIPS expects this)
            img1 = img1 * 2.0 - 1.0
            img2 = img2 * 2.0 - 1.0
            
            with torch.no_grad():
                lpips_value = model(img1, img2)
            
            return lpips_value.item()
        except Exception as e:
            print(f"[LPIPS] Error computing LPIPS: {e}")
            return 0.0

except ImportError:
    LPIPS_AVAILABLE = False
    
    def compute_lpips(img1, img2, device='cuda'):
        """Dummy LPIPS function when lpips is not available."""
        return 0.0
    
    def get_lpips_model(device='cuda'):
        """Dummy LPIPS model getter when lpips is not available."""
        return None