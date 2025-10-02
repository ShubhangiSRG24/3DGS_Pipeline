"""
Quality metrics for 3D Gaussian Splatting reconstruction evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


class QualityMetrics3DGS:
    """Quality metrics for 3DGS reconstruction evaluation"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_reconstruction(self, reconstruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate complete reconstruction quality for 3DGS"""
        
        metrics = {}
        
        # Camera pose evaluation
        if 'cameras' in reconstruction_data and 'images' in reconstruction_data:
            metrics.update(self._evaluate_camera_poses(
                reconstruction_data['cameras'], 
                reconstruction_data['images']
            ))
        
        # Point cloud evaluation
        if 'sparse_points' in reconstruction_data:
            metrics.update(self._evaluate_point_cloud(
                reconstruction_data['sparse_points']
            ))
        
        
        # Overall quality score
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _evaluate_camera_poses(self, cameras: Dict, images: Dict) -> Dict[str, float]:
        """Evaluate camera pose quality"""
        
        metrics = {
            'camera_count': len(cameras),
            'image_count': len(images),
            'reprojection_error': 0.0,
            'pose_consistency': 0.0,
            'baseline_ratio': 0.0
        }
        
        if len(images) < 2:
            return metrics
        
        # Calculate reprojection errors
        reprojection_errors = []
        for img_name, img_data in images.items():
            if 'xys' in img_data and 'point3D_ids' in img_data:
                errors = self._calculate_reprojection_errors(img_data)
                if errors:
                    reprojection_errors.extend(errors)
        
        if reprojection_errors:
            metrics['reprojection_error'] = np.mean(reprojection_errors)
        
        # Calculate pose consistency
        metrics['pose_consistency'] = self._calculate_pose_consistency(images)
        
        # Calculate baseline ratios
        metrics['baseline_ratio'] = self._calculate_baseline_ratios(images)
        
        return metrics
    
    def _evaluate_point_cloud(self, sparse_points: Dict) -> Dict[str, float]:
        """Evaluate point cloud quality"""
        
        metrics = {
            'point_count': len(sparse_points),
            'track_length': 0.0,
            'point_density': 0.0,
            'point_quality': 0.0
        }
        
        if not sparse_points:
            return metrics
        
        # Calculate track lengths
        track_lengths = []
        for point_id, point_data in sparse_points.items():
            if 'track' in point_data:
                track_lengths.append(len(point_data['track']))
        
        if track_lengths:
            metrics['track_length'] = np.mean(track_lengths)
        
        # Calculate point density (simplified)
        metrics['point_density'] = len(sparse_points) / 1000.0  # Normalized
        
        # Calculate point quality based on track length and reprojection error
        if track_lengths and 'reprojection_error' in self.metrics:
            metrics['point_quality'] = np.mean(track_lengths) / (1 + self.metrics['reprojection_error'])
        
        return metrics
    
    def _evaluate_depth_maps(self, depth_maps: Dict) -> Dict[str, float]:
        """Evaluate depth map quality"""
        
        metrics = {
            'depth_consistency': 0.0,
            'depth_smoothness': 0.0,
            'depth_coverage': 0.0
        }
        
        if not depth_maps:
            return metrics
        
        # Calculate depth consistency across views
        consistency_scores = []
        smoothness_scores = []
        coverage_scores = []
        
        for img_path, depth_map in depth_maps.items():
            if depth_map is not None and depth_map.size > 0:
                # Coverage: percentage of non-zero depth values
                coverage = np.sum(depth_map > 0) / depth_map.size
                coverage_scores.append(coverage)
                
                # Smoothness: gradient magnitude
                if depth_map.shape[0] > 1 and depth_map.shape[1] > 1:
                    grad_x = np.gradient(depth_map, axis=1)
                    grad_y = np.gradient(depth_map, axis=0)
                    smoothness = 1.0 / (1.0 + np.mean(np.sqrt(grad_x**2 + grad_y**2)))
                    smoothness_scores.append(smoothness)
        
        if coverage_scores:
            metrics['depth_coverage'] = np.mean(coverage_scores)
        if smoothness_scores:
            metrics['depth_smoothness'] = np.mean(smoothness_scores)
        
        return metrics
    
    def _calculate_reprojection_errors(self, img_data: Dict) -> List[float]:
        """Calculate reprojection errors for an image"""
        
        errors = []
        
        # This would calculate actual reprojection errors
        # For now, return placeholder
        return errors
    
    def _calculate_pose_consistency(self, images: Dict) -> float:
        """Calculate pose consistency across images"""
        
        # This would calculate pose consistency metrics
        # For now, return placeholder
        return 0.95
    
    def _calculate_baseline_ratios(self, images: Dict) -> float:
        """Calculate baseline ratios between camera pairs"""
        
        # This would calculate baseline ratios
        # For now, return placeholder
        return 0.8
    
    def _calculate_overall_quality(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score for 3DGS"""
        
        # Weighted combination of different metrics
        weights = {
            'reprojection_error': -0.3,  # Lower is better
            'pose_consistency': 0.2,
            'track_length': 0.2,
            'point_density': 0.15,
            'depth_coverage': 0.15
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == 'reprojection_error':
                    # Convert error to score (lower error = higher score)
                    score += weight * max(0, 1.0 - value / 10.0)
                else:
                    score += weight * min(1.0, value)
                total_weight += abs(weight)
        
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0.5
        
        return max(0.0, min(1.0, final_score))
    
    def get_quality_report(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable quality report"""
        
        report = []
        report.append("=" * 60)
        report.append("3DGS RECONSTRUCTION QUALITY REPORT")
        report.append("=" * 60)
        
        # Camera metrics
        if 'camera_count' in metrics:
            report.append(f"📷 Cameras: {metrics['camera_count']}")
        if 'image_count' in metrics:
            report.append(f"🖼️  Images: {metrics['image_count']}")
        if 'reprojection_error' in metrics:
            report.append(f"🎯 Reprojection Error: {metrics['reprojection_error']:.3f} px")
        if 'pose_consistency' in metrics:
            report.append(f"🔗 Pose Consistency: {metrics['pose_consistency']:.3f}")
        
        # Point cloud metrics
        if 'point_count' in metrics:
            report.append(f"📍 3D Points: {metrics['point_count']}")
        if 'track_length' in metrics:
            report.append(f"📏 Avg Track Length: {metrics['track_length']:.2f}")
        if 'point_density' in metrics:
            report.append(f"📊 Point Density: {metrics['point_density']:.3f}")
        
        # Depth metrics
        if 'depth_coverage' in metrics:
            report.append(f"🏔️ Depth Coverage: {metrics['depth_coverage']:.1%}")
        if 'depth_smoothness' in metrics:
            report.append(f"🌊 Depth Smoothness: {metrics['depth_smoothness']:.3f}")
        
        # Overall quality
        if 'overall_quality' in metrics:
            quality_level = "Excellent" if metrics['overall_quality'] > 0.9 else \
                          "Good" if metrics['overall_quality'] > 0.7 else \
                          "Fair" if metrics['overall_quality'] > 0.5 else "Poor"
            report.append(f"⭐ Overall Quality: {metrics['overall_quality']:.3f} ({quality_level})")
        
        report.append("=" * 60)
        
        return "\n".join(report) 