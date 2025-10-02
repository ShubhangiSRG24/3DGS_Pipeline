#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
from scene.knn_cache import _knn_cache

def batched_knn_distances(xyz, k, chunk_size=4096, use_sampling=False, sample_ratio=0.5):
    """
    Compute k-nearest neighbor distances using batched/chunked approach for large datasets.
    
    Args:
        xyz: [N, 3] point positions
        k: Number of nearest neighbors
        chunk_size: Size of chunks for batched processing
        use_sampling: Whether to use sampling for very large datasets
        sample_ratio: Ratio of points to sample when using sampling
    
    Returns:
        knn_distances: [N, k] or [N_sampled, k] distances to k nearest neighbors
        knn_indices: [N, k] or [N_sampled, k] indices of k nearest neighbors
        valid_mask: [N] boolean mask indicating which points were processed
    """
    device = xyz.device
    N = xyz.shape[0]
    
    # Check cache first for self-KNN (xyz vs xyz case)
    cached_distances, cached_indices = _knn_cache.get(xyz, k)
    if cached_distances is not None:
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
        return cached_distances, cached_indices, valid_mask
    
    # For very large datasets, use sampling
    if use_sampling and N > 50000:
        n_samples = int(N * sample_ratio)
        sample_indices = torch.randperm(N, device=device)[:n_samples]
        sample_xyz = xyz[sample_indices]
        
        # Compute KNN for samples against full dataset
        knn_distances, knn_indices = batched_knn_core(sample_xyz, xyz, k, chunk_size)
        
        # Create valid mask for sampled points
        valid_mask = torch.zeros(N, dtype=torch.bool, device=device)
        valid_mask[sample_indices] = True
        
        return knn_distances, knn_indices, valid_mask
    
    else:
        # Process all points
        knn_distances, knn_indices = batched_knn_core(xyz, xyz, k, chunk_size)
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
        
        # Update cache for self-KNN
        _knn_cache.update(xyz, k, knn_distances, knn_indices)
        
        return knn_distances, knn_indices, valid_mask

def batched_knn_core(query_points, reference_points, k, chunk_size=4096):
    """
    Core batched KNN computation using chunked torch.cdist.
    
    Args:
        query_points: [M, 3] points to find neighbors for
        reference_points: [N, 3] reference point cloud
        k: Number of nearest neighbors
        chunk_size: Size of processing chunks
    
    Returns:
        knn_distances: [M, k] distances to k nearest neighbors
        knn_indices: [M, k] indices of k nearest neighbors in reference_points
    """
    device = query_points.device
    M = query_points.shape[0]
    N = reference_points.shape[0]
    
    # Initialize output tensors
    knn_distances = torch.zeros(M, k, device=device)
    knn_indices = torch.zeros(M, k, dtype=torch.long, device=device)
    
    # Process in chunks to manage memory
    for start_idx in range(0, M, chunk_size):
        end_idx = min(start_idx + chunk_size, M)
        chunk_queries = query_points[start_idx:end_idx]
        
        # For very large reference sets, also chunk the reference points
        if N > 20000:
            # Use chunked reference processing
            chunk_knn_dists, chunk_knn_indices = chunked_reference_knn(
                chunk_queries, reference_points, k, chunk_size
            )
        else:
            # Standard approach for smaller reference sets
            distances = torch.cdist(chunk_queries, reference_points)
            
            # Set diagonal to infinity for self-distance exclusion if same point set
            if torch.equal(query_points, reference_points):
                for i, global_i in enumerate(range(start_idx, end_idx)):
                    distances[i, global_i] = float('inf')
            
            chunk_knn_dists, chunk_knn_indices = torch.topk(
                distances, k, dim=1, largest=False
            )
        
        knn_distances[start_idx:end_idx] = chunk_knn_dists
        knn_indices[start_idx:end_idx] = chunk_knn_indices
    
    return knn_distances, knn_indices

def chunked_reference_knn(query_chunk, reference_points, k, ref_chunk_size=4096):
    """
    Find KNN for query chunk against large reference set using reference chunking.
    
    Args:
        query_chunk: [M, 3] query points
        reference_points: [N, 3] reference points (large)
        k: Number of nearest neighbors
        ref_chunk_size: Size of reference chunks
    
    Returns:
        knn_distances: [M, k] final k nearest distances
        knn_indices: [M, k] final k nearest indices
    """
    device = query_chunk.device
    M = query_chunk.shape[0]
    N = reference_points.shape[0]
    
    # Initialize with very large distances
    best_distances = torch.full((M, k), float('inf'), device=device)
    best_indices = torch.zeros(M, k, dtype=torch.long, device=device)
    
    # Process reference points in chunks
    for ref_start in range(0, N, ref_chunk_size):
        ref_end = min(ref_start + ref_chunk_size, N)
        ref_chunk = reference_points[ref_start:ref_end]
        
        # Compute distances for this reference chunk
        chunk_distances = torch.cdist(query_chunk, ref_chunk)
        
        # Get top-k from this chunk
        chunk_k = min(k, ref_chunk.shape[0])
        chunk_knn_dists, chunk_knn_indices = torch.topk(
            chunk_distances, chunk_k, dim=1, largest=False
        )
        
        # Adjust indices to global reference indexing
        chunk_knn_indices += ref_start
        
        # Merge with current best
        # Combine current best with chunk results
        combined_dists = torch.cat([best_distances, chunk_knn_dists], dim=1)
        combined_indices = torch.cat([best_indices, chunk_knn_indices], dim=1)
        
        # Keep only top-k overall
        final_k = min(k, combined_dists.shape[1])
        best_distances, topk_positions = torch.topk(
            combined_dists, final_k, dim=1, largest=False
        )
        
        # Select corresponding indices
        best_indices = torch.gather(combined_indices, 1, topk_positions)
    
    return best_distances, best_indices