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

class KNNCache:
    """Intelligent cache for KNN results with smart invalidation based on scene changes."""
    
    def __init__(self, max_age=3, change_threshold=0.05, point_count_threshold=0.1):
        # Cached KNN results
        self.cached_indices = None
        self.cached_distances = None
        self.cached_positions = None
        self.cached_k = None
        
        # Cache lifecycle management
        self.age = 0
        self.max_age = max_age  # Reduced from 5 to 3 cycles
        self.change_threshold = change_threshold  # Reduced from 0.1 to 0.05 for more sensitivity
        
        # Smart invalidation triggers
        self.cached_point_count = None
        self.point_count_threshold = point_count_threshold  # 10% point count change
        self.cached_visibility_pattern = None
        self.last_camera_positions = None
        self.camera_coverage_threshold = 0.2  # 20% camera coverage change
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.invalidation_reasons = []
    
    def get(self, xyz, k, visibility_count=None, camera_positions=None):
        """Get cached KNN results if valid with intelligent invalidation."""
        if (self.cached_indices is None or self.cached_positions is None or 
            self.cached_k != k):
            self.cache_misses += 1
            return None, None
            
        # Check for cache invalidation triggers
        invalidation_reason = self._check_invalidation_triggers(
            xyz, visibility_count, camera_positions
        )
        
        if invalidation_reason:
            self.invalidation_reasons.append(invalidation_reason)
            self.cache_misses += 1
            print(f"[KNN_CACHE] Invalidated: {invalidation_reason}")
            return None, None
        
        # Cache is still valid
        self.cache_hits += 1
        self.increment_age()
        return self.cached_distances, self.cached_indices
    
    def _check_invalidation_triggers(self, xyz, visibility_count=None, camera_positions=None):
        """Check various triggers that should invalidate the cache."""
        current_point_count = xyz.shape[0]
        
        # Trigger 1: Age limit exceeded
        if self.age >= self.max_age:
            return f"age_limit_exceeded ({self.age}/{self.max_age})"
        
        # Trigger 2: Point count changed significantly
        if self.cached_point_count is not None:
            point_count_change = abs(current_point_count - self.cached_point_count) / self.cached_point_count
            if point_count_change > self.point_count_threshold:
                return f"point_count_change ({point_count_change:.1%} > {self.point_count_threshold:.1%})"
        
        # Trigger 3: Positions changed significantly
        if xyz.shape == self.cached_positions.shape:
            with torch.no_grad():
                position_change = torch.mean(torch.norm(xyz - self.cached_positions, dim=1))
                relative_change = position_change / (torch.mean(torch.norm(xyz, dim=1)) + 1e-8)
                
                if relative_change > self.change_threshold:
                    return f"position_change ({relative_change:.3f} > {self.change_threshold:.3f})"
        else:
            return f"shape_mismatch (cached: {self.cached_positions.shape}, current: {xyz.shape})"
        
        # Trigger 4: Camera coverage changed significantly
        if camera_positions is not None and self.last_camera_positions is not None:
            if len(camera_positions) != len(self.last_camera_positions):
                return f"camera_count_change ({len(self.last_camera_positions)} -> {len(camera_positions)})"
                
            # Check if camera positions shifted significantly
            with torch.no_grad():
                if isinstance(camera_positions, (list, tuple)):
                    # Convert list of positions to tensor
                    current_cams = torch.stack([torch.tensor(pos, device=xyz.device) 
                                              for pos in camera_positions[-10:]])  # Last 10 cameras
                    cached_cams = torch.stack([torch.tensor(pos, device=xyz.device) 
                                             for pos in self.last_camera_positions[-10:]])
                else:
                    current_cams = camera_positions[-10:]  # Last 10 cameras
                    cached_cams = self.last_camera_positions[-10:]
                
                if current_cams.shape == cached_cams.shape:
                    camera_change = torch.mean(torch.norm(current_cams - cached_cams, dim=-1))
                    # Normalize by average camera distance from origin
                    avg_cam_dist = torch.mean(torch.norm(current_cams, dim=-1)) + 1e-8
                    relative_camera_change = camera_change / avg_cam_dist
                    
                    if relative_camera_change > self.camera_coverage_threshold:
                        return f"camera_coverage_change ({relative_camera_change:.3f} > {self.camera_coverage_threshold:.3f})"
        
        # Trigger 5: Visibility pattern changed significantly (optional)
        if visibility_count is not None and self.cached_visibility_pattern is not None:
            if visibility_count.shape == self.cached_visibility_pattern.shape:
                with torch.no_grad():
                    # Check if visibility distribution changed
                    vis_change = torch.mean(torch.abs(visibility_count - self.cached_visibility_pattern).float())
                    avg_visibility = torch.mean(visibility_count.float()) + 1e-8
                    relative_vis_change = vis_change / avg_visibility
                    
                    if relative_vis_change > 0.3:  # 30% visibility change
                        return f"visibility_pattern_change ({relative_vis_change:.2f} > 0.30)"
        
        # No invalidation triggers activated
        return None
    
    def update(self, xyz, k, distances, indices, visibility_count=None, camera_positions=None):
        """Update cache with new results and invalidation context."""
        self.cached_positions = xyz.clone()
        self.cached_k = k
        self.cached_distances = distances
        self.cached_indices = indices
        self.age = 0
        
        # Update invalidation context
        self.cached_point_count = xyz.shape[0]
        
        if visibility_count is not None:
            self.cached_visibility_pattern = visibility_count.clone()
        
        if camera_positions is not None:
            # Store recent camera positions for coverage tracking
            if isinstance(camera_positions, (list, tuple)):
                self.last_camera_positions = list(camera_positions[-10:])  # Keep last 10
            else:
                self.last_camera_positions = camera_positions[-10:].clone()
        
        print(f"[KNN_CACHE] Updated cache for {xyz.shape[0]:,} points, k={k}")
    
    def increment_age(self):
        """Increment cache age."""
        self.age += 1
    
    def clear(self):
        """Clear the cache completely."""
        self.cached_indices = None
        self.cached_distances = None
        self.cached_positions = None
        self.cached_k = None
        self.age = 0
        self.cached_point_count = None
        self.cached_visibility_pattern = None
        self.last_camera_positions = None
        print("[KNN_CACHE] Cache cleared")
    
    def get_performance_stats(self):
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        # Count invalidation reasons
        reason_counts = {}
        for reason in self.invalidation_reasons:
            reason_type = reason.split('(')[0]  # Get reason type without details
            reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'current_age': self.age,
            'max_age': self.max_age,
            'invalidation_reasons': reason_counts,
            'is_active': self.cached_indices is not None
        }
    
    def print_performance_stats(self):
        """Print detailed cache performance statistics."""
        stats = self.get_performance_stats()
        print(f"\n[KNN_CACHE] Performance Statistics:")
        print(f"  Cache Hits: {stats['cache_hits']:,}")
        print(f"  Cache Misses: {stats['cache_misses']:,}")
        print(f"  Hit Rate: {stats['hit_rate']:.1%}")
        print(f"  Current Age: {stats['current_age']}/{stats['max_age']}")
        print(f"  Active: {stats['is_active']}")
        
        if stats['invalidation_reasons']:
            print(f"  Invalidation Reasons:")
            for reason, count in stats['invalidation_reasons'].items():
                print(f"    {reason}: {count}")
        print()

# Global KNN cache instance (for backward compatibility)
_knn_cache = KNNCache()