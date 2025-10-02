"""
Direct COLMAP binary execution without pycolmap to avoid CUDA conflicts
"""

import os
import subprocess
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
import cv2
from collections import namedtuple

logger = logging.getLogger(__name__)

# COLMAP data structures
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def filter_matches_with_magsac(features: Dict[str, Any], matches: Dict[Tuple[str, str], Any]) -> Dict[Tuple[str, str], Any]:
    """Filter matches using cv2.USAC_MAGSAC geometric verification with optimized parameters"""
    logger.info("Filtering matches with cv2 USAC_MAGSAC (EXTREME parameters: threshold=0.8, confidence=0.9999, maxIters=5000)...")
    
    filtered_matches = {}
    
    for (img1_path, img2_path), match_data in tqdm(matches.items(), desc="MAGSAC filtering"):
        try:
            # Get keypoints for both images
            kpts1 = features[img1_path]['keypoints']
            kpts2 = features[img2_path]['keypoints']
            
            # Get matched keypoint indices
            matches0 = match_data['matches0']
            matches1 = match_data['matches1']
            
            # Filter out invalid matches (-1)
            valid_mask = (matches0 >= 0) & (matches1 >= 0)
            if not valid_mask.any():
                continue
                
            valid_matches0 = matches0[valid_mask]
            valid_matches1 = matches1[valid_mask]
            
            # Get matched keypoints
            matched_kpts1 = kpts1[valid_matches0]
            matched_kpts2 = kpts2[valid_matches1]
            
            if len(matched_kpts1) < 8:  # Need at least 8 points for fundamental matrix
                continue
            
            # Run MAGSAC with balanced parameters
            F_matrix, inlier_mask = cv2.findFundamentalMat(
                matched_kpts1.astype(np.float32),
                matched_kpts2.astype(np.float32),
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=2.0,  # Reasonable threshold (was 0.8, too strict)
                confidence=0.999,           # Standard confidence (was 0.9999, too high)
                maxIters=1000              # Efficient iterations (was 5000, too many)
            )
            
            if F_matrix is None or inlier_mask is None:
                continue
            
            # Keep only inlier matches
            inlier_mask = inlier_mask.ravel().astype(bool)
            if inlier_mask.sum() < 8:  # Need minimum matches (was 15, too high)
                continue
            
            # Update match data with filtered matches
            filtered_matches0 = valid_matches0[inlier_mask]
            filtered_matches1 = valid_matches1[inlier_mask]
            
            # Create new match data
            filtered_matches[(img1_path, img2_path)] = {
                'matches0': filtered_matches0,
                'matches1': filtered_matches1,
                'mscores0': match_data['mscores0'][valid_mask][inlier_mask] if 'mscores0' in match_data else np.ones(len(filtered_matches0)),
                'mscores1': match_data['mscores1'][valid_mask][inlier_mask] if 'mscores1' in match_data else np.ones(len(filtered_matches1))
            }
            
            logger.debug(f"Match {Path(img1_path).name} - {Path(img2_path).name}: {len(valid_matches0)} -> {len(filtered_matches0)} matches")
            
        except Exception as e:
            #logger.debug(f"MAGSAC filtering failed for {Path(img1_path).name} - {Path(img2_path).name}: {e}")
            continue
    
    retention_rate = len(filtered_matches) / len(matches) * 100 if matches else 0
    logger.info(f"MAGSAC filtering: {len(matches)} -> {len(filtered_matches)} pairs ({retention_rate:.1f}% retained)")
    return filtered_matches


def create_colmap_database(features: Dict[str, Any], matches: Dict[Tuple[str, str], Any], 
                          database_path: Path) -> Dict[str, int]:
    """Create COLMAP database with features and matches"""
    logger.info("Creating COLMAP database...")
    
    # Remove existing database
    if database_path.exists():
        database_path.unlink()
    
    # Create database connection
    conn = sqlite3.connect(database_path)
    
    # Create COLMAP tables
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL);
            
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL);
            
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB);
            
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB);
            
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB,
            qvec BLOB,
            tvec BLOB);
    ''')
    
    # Add camera (assume single camera)
    first_feature = next(iter(features.values()))
    height, width = first_feature['image_shape']
    focal_length = 1.2 * max(width, height)  # hloc heuristic
    
    camera_params = np.array([focal_length, width / 2.0, height / 2.0], dtype=np.float64)
    params_blob = camera_params.tobytes()
    
    conn.execute(
        "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
        (1, 0, width, height, params_blob, 1)  # model 0 = SIMPLE_PINHOLE
    )
    
    # Add images and get image IDs
    image_ids = {}
    for i, img_path in enumerate(features.keys(), 1):
        image_name = Path(img_path).name
        conn.execute(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            (i, image_name, 1)
        )
        image_ids[image_name] = i
    
    # Add keypoints
    logger.info("Adding keypoints to database...")
    for img_path, feat_data in tqdm(features.items(), desc="Adding keypoints"):
        image_name = Path(img_path).name
        image_id = image_ids[image_name]
        keypoints = feat_data['keypoints'] + 0.5  # COLMAP origin
        keypoints = keypoints.astype(np.float32)
        keypoints_blob = keypoints.tobytes()
        
        conn.execute(
            "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_blob)
        )
    
    # Add matches
    logger.info("Adding matches to database...")
    for (img1_path, img2_path), match_data in tqdm(matches.items(), desc="Adding matches"):
        name0 = Path(img1_path).name
        name1 = Path(img2_path).name
        
        if name0 in image_ids and name1 in image_ids:
            id0, id1 = image_ids[name0], image_ids[name1]
            
            # Create pair ID
            if id0 > id1:
                id0, id1 = id1, id0
            pair_id = id0 * 2147483647 + id1
            
            # Extract matches
            matches_array = np.column_stack([
                match_data['matches0'],
                match_data['matches1']
            ]).astype(np.uint32)
            
            matches_blob = matches_array.tobytes()
            
            conn.execute(
                "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, matches_array.shape[0], matches_array.shape[1], matches_blob)
            )
            
            # Also add to two_view_geometries (required for COLMAP incremental mapping)
            # Use dummy geometry for now - COLMAP will verify
            dummy_F = np.eye(3, dtype=np.float64).tobytes()
            dummy_E = np.eye(3, dtype=np.float64).tobytes()
            dummy_H = np.eye(3, dtype=np.float64).tobytes()
            dummy_qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64).tobytes()
            dummy_tvec = np.array([0.0, 0.0, 0.0], dtype=np.float64).tobytes()
            
            conn.execute(
                "INSERT OR REPLACE INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id, matches_array.shape[0], matches_array.shape[1], matches_blob, 2, dummy_F, dummy_E, dummy_H, dummy_qvec, dummy_tvec)
            )
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database created with {len(image_ids)} images and {len(matches)} match pairs")
    
    # Verify database contents
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM matches")
    match_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM keypoints")
    keypoint_count = cursor.fetchone()[0]
    
    conn.close()
    
    logger.info(f"Database verification: {image_count} images, {match_count} matches, {keypoint_count} keypoints")
    
    return image_ids


def run_colmap_binary(database_path: Path, image_dir: Path, output_path: Path) -> bool:
    """Run COLMAP using binary executable"""
    
    sparse_path = output_path / "sparse"
    sparse_path.mkdir(exist_ok=True)
    
    # Step 2: Incremental mapping (hloc-style minimal options)
    logger.info("Running COLMAP incremental mapping...")
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_path),
        "--Mapper.num_threads", "16",
        # CPU 최적화 옵션들 (PBA 미지원 버전)
        "--Mapper.ba_local_max_num_iterations", "20",  # 25->20 (안전한 가속)
        "--Mapper.ba_global_max_num_iterations", "35",  # 50->35 (안전한 가속)
        "--Mapper.max_num_models", "1",  # 단일 모델
        "--Mapper.max_model_overlap", "15",  # 20->15 (적당한 가속)
        "--Mapper.min_num_matches", "8",   # Reduced from 15 to 8
        "--Mapper.ba_global_images_ratio", "1.2",  # 메모리 효율성
        "--Mapper.ba_global_points_ratio", "1.2"  # 메모리 효율성
    ]
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        logger.info(f"COLMAP mapper return code: {result.returncode}")
        logger.info(f"COLMAP stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"COLMAP stderr: {result.stderr}")
            
        if result.returncode == 0:
            logger.info("COLMAP incremental mapping completed successfully")
            
            # Convert binary to text format for easier reading
            model_converter_cmd = [
                "colmap", "model_converter",
                "--input_path", str(sparse_path / "0"),
                "--output_path", str(sparse_path / "0"),
                "--output_type", "TXT"
            ]
            
            try:
                logger.info("Converting COLMAP binary to text format...")
                converter_result = subprocess.run(model_converter_cmd, capture_output=True, text=True, timeout=120)
                if converter_result.returncode == 0:
                    logger.info("Binary to text conversion successful")
                else:
                    logger.warning(f"Binary to text conversion failed: {converter_result.stderr}")
            except Exception as e:
                logger.warning(f"Model converter error (non-critical): {e}")
            
            return True
        else:
            logger.error(f"COLMAP mapping failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("COLMAP mapping timed out")
        return False
    except Exception as e:
        logger.error(f"COLMAP mapping error: {e}")
        return False


def read_cameras_binary(path_to_model_file: Path) -> Dict[int, Dict]:
    """Read cameras.bin file with robust error handling"""
    cameras = {}
    
    if not path_to_model_file.exists():
        logger.error(f"Camera binary file not found: {path_to_model_file}")
        return cameras
        
    # Check file size
    file_size = path_to_model_file.stat().st_size
    if file_size < 8:
        logger.error(f"Camera binary file too small ({file_size} bytes)")
        return cameras
    
    try:
        with open(path_to_model_file, "rb") as fid:
            # Read number of cameras with error handling
            try:
                num_cameras_data = fid.read(8)
                if len(num_cameras_data) < 8:
                    logger.error(f"Could not read camera count header")
                    return cameras
                num_cameras = struct.unpack("<Q", num_cameras_data)[0]
            except struct.error as e:
                logger.error(f"Error reading camera count: {e}")
                return cameras
            
            # Bounds check
            if num_cameras > 10000:
                logger.warning(f"Unusually large number of cameras: {num_cameras}, truncating to 10000")
                num_cameras = 10000
                
            if num_cameras == 0:
                logger.warning("No cameras found in binary file")
                return cameras
                
            logger.info(f"Reading {num_cameras} cameras from binary file")
            
            for _ in range(num_cameras):
                # Read camera properties with proper error handling
                try:
                    camera_properties = struct.unpack("<IiQQ", fid.read(24))  # Use I for unsigned int
                    camera_id = camera_properties[0]
                    model_id = camera_properties[1]
                    width = int(camera_properties[2])
                    height = int(camera_properties[3])
                    
                    num_params = struct.unpack("<Q", fid.read(8))[0]
                    
                    # Bounds check for params
                    if num_params > 20:  # Reasonable limit for camera parameters
                        logger.warning(f"Unusually large number of camera parameters: {num_params}, truncating to 20")
                        num_params = 20
                    
                    params = struct.unpack(f"<{num_params}d", fid.read(8 * num_params))
                    
                    cameras[camera_id] = {
                        'model_id': model_id,
                        'model': 'PINHOLE',  # Simplified
                        'width': width,
                        'height': height,
                        'params': list(params)
                    }
                except (struct.error, OverflowError, ValueError) as e:
                    logger.warning(f"Error reading camera data: {e}, skipping camera")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading cameras.bin file: {e}")
        return {}
        
    return cameras


def read_images_binary(path_to_model_file: Path) -> Dict[int, Dict]:
    """Read images.bin file with robust error handling"""
    images = {}
    
    if not path_to_model_file.exists():
        logger.error(f"Images binary file not found: {path_to_model_file}")
        return images
        
    # Check file size
    file_size = path_to_model_file.stat().st_size
    if file_size < 8:
        logger.error(f"Images binary file too small ({file_size} bytes)")
        return images
    
    try:
        with open(path_to_model_file, "rb") as fid:
            # Read number of images with error handling
            try:
                num_reg_images_data = fid.read(8)
                if len(num_reg_images_data) < 8:
                    logger.error(f"Could not read image count header")
                    return images
                num_reg_images = struct.unpack("<Q", num_reg_images_data)[0]
            except struct.error as e:
                logger.error(f"Error reading image count: {e}")
                return images
            
            # Add bounds checking to prevent overflow
            if num_reg_images > 100000:  # Reasonable upper limit
                logger.warning(f"Unusually large number of registered images: {num_reg_images}, truncating to 100000")
                num_reg_images = 100000
            
            if num_reg_images == 0:
                logger.warning("No registered images found in binary file")
                return images
                
            logger.info(f"Reading {num_reg_images} images from binary file")
            
            for i in range(num_reg_images):
                try:
                    # Record current position for potential backtracking
                    current_pos = fid.tell()
                    
                    # Check available bytes more efficiently
                    end_pos = fid.seek(0, 2)  # Seek to end
                    fid.seek(current_pos)  # Return to current position
                    remaining_bytes = end_pos - current_pos
                    
                    if remaining_bytes < 64:
                        logger.warning(f"Insufficient bytes remaining ({remaining_bytes}) for image header, skipping")
                        break
                    
                    # Try multiple COLMAP format variations
                    format_attempts = [
                        ("<iidddddi", 64),    # Standard: int, int, 4*double, int (64 bytes)
                        ("<Idddddi", 56),     # Alternative: uint, 4*double, int (56 bytes) 
                        ("<iidddd", 48),      # Minimal: int, int, 4*double (48 bytes)
                        ("<Idddd", 40),       # Minimal alt: uint, 4*double (40 bytes)
                        ("<iiddddi", 60),     # Variant: int, int, 4*double, int (60 bytes)
                    ]
                    
                    image_properties_parsed = False
                    
                    for format_str, expected_bytes in format_attempts:
                        try:
                            fid.seek(current_pos)  # Reset position
                            if remaining_bytes < expected_bytes:
                                continue
                                
                            binary_image_properties = struct.unpack(format_str, fid.read(expected_bytes))
                            
                            # Parse based on format
                            if format_str.startswith("<ii"):  # Formats starting with two ints
                                image_id = binary_image_properties[0]
                                # Skip second int (might be padding)
                                qvec = np.array(binary_image_properties[2:6]) if len(binary_image_properties) >= 6 else np.array(binary_image_properties[1:5])
                                tvec = np.array(binary_image_properties[6:9]) if len(binary_image_properties) >= 9 else np.array(binary_image_properties[5:8])
                                camera_id = binary_image_properties[-1] if len(binary_image_properties) > 8 else 1
                            elif format_str.startswith("<I"):  # Formats starting with uint
                                image_id = binary_image_properties[0]
                                qvec = np.array(binary_image_properties[1:5])
                                tvec = np.array(binary_image_properties[5:8]) if len(binary_image_properties) >= 8 else np.array([0.0, 0.0, 0.0])
                                camera_id = binary_image_properties[-1] if len(binary_image_properties) > 7 else 1
                            else:
                                continue
                            
                            # Validate quaternion (should be normalized)
                            qvec_norm = np.linalg.norm(qvec)
                            if qvec_norm < 1e-8 or qvec_norm > 2.0:  # Reasonable bounds
                                continue
                                
                            # Validate image_id
                            if image_id <= 0 or image_id > 1000000:
                                continue
                            
                            image_properties_parsed = True
                            logger.debug(f"Successfully parsed image {image_id} with format {format_str}")
                            break
                            
                        except (struct.error, IndexError, ValueError):
                            continue
                    
                    if not image_properties_parsed:
                        logger.debug(f"Could not parse image properties with any known format, skipping image at position {current_pos}")
                        # Try to skip to next image by reading past this entry
                        try:
                            # Skip ahead with boundary checking
                            next_pos = min(current_pos + 64, end_pos)
                            fid.seek(next_pos)
                        except:
                            break
                        continue
                        
                except Exception as e:
                    logger.warning(f"Unexpected error reading image: {e}, skipping")
                    continue
                
                try:
                    # Read image name
                    image_name = ""
                    current_char = struct.unpack("<c", fid.read(1))[0]
                    while current_char != b"\x00":
                        image_name += current_char.decode("utf-8")
                        current_char = struct.unpack("<c", fid.read(1))[0]
                    
                    # Read 2D points with overflow protection
                    num_points2D = struct.unpack("<Q", fid.read(8))[0]
                    
                    # Add bounds checking to prevent overflow
                    if num_points2D > 1000000:  # Reasonable upper limit
                        logger.warning(f"Unusually large number of 2D points ({num_points2D}) for image {image_name}, truncating to 1000000")
                        num_points2D = 1000000
                    
                    if num_points2D > 0:
                        try:
                            # Use int64 to prevent overflow in struct calculations
                            total_elements = int(num_points2D) * 3
                            total_bytes = int(num_points2D) * 24
                            
                            x_y_id_s = struct.unpack(f"<{total_elements}d", fid.read(total_bytes))
                            xys = np.column_stack([np.array(x_y_id_s[0::3]), np.array(x_y_id_s[1::3])])
                            point3D_ids = np.array(x_y_id_s[2::3], dtype=np.int64)
                        except (struct.error, OverflowError, MemoryError) as e:
                            logger.warning(f"Error reading 2D points for image {image_name}: {e}, skipping")
                            xys = np.array([]).reshape(0, 2)
                            point3D_ids = np.array([], dtype=np.int64)
                    else:
                        xys = np.array([]).reshape(0, 2)
                        point3D_ids = np.array([], dtype=np.int64)
                    
                    images[image_id] = {
                        'qvec': qvec,
                        'tvec': tvec,
                        'camera_id': camera_id,
                        'name': image_name,
                        'xys': xys,
                        'point3D_ids': point3D_ids
                    }
                    
                except (struct.error, UnicodeDecodeError, ValueError) as e:
                    logger.warning(f"Error reading image data for image_id {image_id}: {e}, skipping")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading images.bin file: {e}")
        return {}
        
    return images


def read_points3d_binary(path_to_model_file: Path) -> Dict[int, Dict]:
    """Read points3D.bin file with robust error handling"""
    points3D = {}
    
    if not path_to_model_file.exists():
        logger.error(f"Points3D binary file not found: {path_to_model_file}")
        return points3D
        
    # Check file size
    file_size = path_to_model_file.stat().st_size
    if file_size < 8:
        logger.error(f"Points3D binary file too small ({file_size} bytes)")
        return points3D
    
    try:
        with open(path_to_model_file, "rb") as fid:
            # Read number of points with error handling
            try:
                num_points_data = fid.read(8)
                if len(num_points_data) < 8:
                    logger.error(f"Could not read points count header")
                    return points3D
                num_points = struct.unpack("<Q", num_points_data)[0]
            except struct.error as e:
                logger.error(f"Error reading points count: {e}")
                return points3D
            
            # Add bounds checking to prevent overflow
            if num_points > 10000000:  # Reasonable upper limit
                logger.warning(f"Unusually large number of 3D points: {num_points}, truncating to 10000000")
                num_points = 10000000
                
            if num_points == 0:
                logger.warning("No 3D points found in binary file")
                return points3D
                
            logger.info(f"Reading {num_points} 3D points from binary file")
            
            for _ in range(num_points):
                try:
                    # Read point properties
                    point_data = fid.read(43)
                    if len(point_data) < 43:
                        logger.warning(f"Insufficient data for 3D point, skipping")
                        break
                        
                    binary_point_line_properties = struct.unpack("<QdddBBBd", point_data)
                    point3D_id = binary_point_line_properties[0]
                    xyz = np.array(binary_point_line_properties[1:4])
                    rgb = np.array(binary_point_line_properties[4:7], dtype=np.int64)
                    error = binary_point_line_properties[7]
                    
                    # Read track length
                    track_length_data = fid.read(8)
                    if len(track_length_data) < 8:
                        logger.warning(f"Cannot read track length for point {point3D_id}, skipping")
                        continue
                        
                    track_length = struct.unpack("<Q", track_length_data)[0]
                    
                    # Add bounds checking to prevent overflow
                    if track_length > 100000:  # Reasonable upper limit for track length
                        logger.warning(f"Unusually large track length ({track_length}) for point {point3D_id}, truncating to 100000")
                        track_length = 100000
                    
                    if track_length > 0:
                        try:
                            # Use int64 to prevent overflow in struct calculations
                            total_elements = int(track_length) * 2
                            total_bytes = int(track_length) * 8
                            
                            track_data = fid.read(total_bytes)
                            if len(track_data) < total_bytes:
                                logger.warning(f"Insufficient track data for point {point3D_id}, using empty track")
                                image_ids = np.array([], dtype=np.int64)
                                point2D_idxs = np.array([], dtype=np.int64)
                            else:
                                track_elems = struct.unpack(f"<{total_elements}i", track_data)
                                image_ids = np.array(track_elems[0::2], dtype=np.int64)
                                point2D_idxs = np.array(track_elems[1::2], dtype=np.int64)
                        except (struct.error, OverflowError, MemoryError) as e:
                            logger.warning(f"Error reading track for point {point3D_id}: {e}, using empty track")
                            image_ids = np.array([], dtype=np.int64)
                            point2D_idxs = np.array([], dtype=np.int64)
                    else:
                        image_ids = np.array([], dtype=np.int64)
                        point2D_idxs = np.array([], dtype=np.int64)
                    
                    points3D[point3D_id] = {
                        'xyz': xyz,
                        'rgb': rgb,
                        'error': error,
                        'track': list(zip(image_ids, point2D_idxs))
                    }
                    
                except (struct.error, ValueError) as e:
                    logger.warning(f"Error reading 3D point: {e}, skipping")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading points3D.bin file: {e}")
        
    return points3D


def read_colmap_text_results(sparse_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Read COLMAP text results (fallback when binary reading fails)"""
    
    # Look for reconstruction directory
    reconstruction_dirs = [d for d in sparse_path.iterdir() if d.is_dir()]
    if not reconstruction_dirs:
        logger.warning("No reconstruction found")
        return {}, {}, {}
    
    recon_dir = reconstruction_dirs[0]
    logger.info(f"Reading COLMAP text results from {recon_dir}")
    
    cameras_file = recon_dir / "cameras.txt"
    images_file = recon_dir / "images.txt"
    points_file = recon_dir / "points3D.txt"
    
    cameras, images, points3d = {}, {}, {}
    
    # Read cameras.txt
    if cameras_file.exists():
        try:
            with open(cameras_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 5:
                            camera_id = int(parts[0])
                            model = parts[1]
                            width = int(parts[2])
                            height = int(parts[3])
                            params = [float(p) for p in parts[4:]]
                            cameras[camera_id] = {
                                'model': model,
                                'width': width,
                                'height': height,
                                'params': params
                            }
            logger.info(f"Read {len(cameras)} cameras from text file")
        except Exception as e:
            logger.warning(f"Error reading cameras.txt: {e}")
    
    # Read images.txt
    if images_file.exists():
        try:
            with open(images_file, 'r') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 10:
                            image_id = int(parts[0])
                            qvec = [float(parts[j]) for j in range(1, 5)]
                            tvec = [float(parts[j]) for j in range(5, 8)]
                            camera_id = int(parts[8])
                            name = parts[9]
                            
                            # Next line contains 2D points (optional)
                            xys, point3D_ids = [], []
                            if i + 1 < len(lines):
                                points_line = lines[i + 1].strip()
                                if points_line and not points_line.startswith('#'):
                                    point_parts = points_line.split()
                                    for j in range(0, len(point_parts), 3):
                                        if j + 2 < len(point_parts):
                                            x, y = float(point_parts[j]), float(point_parts[j + 1])
                                            point3d_id = int(point_parts[j + 2]) if point_parts[j + 2] != '-1' else -1
                                            xys.append([x, y])
                                            point3D_ids.append(point3d_id)
                            
                            images[image_id] = {
                                'qvec': qvec,
                                'tvec': tvec,
                                'camera_id': camera_id,
                                'name': name,
                                'xys': np.array(xys).reshape(-1, 2) if xys else np.array([]).reshape(0, 2),
                                'point3D_ids': np.array(point3D_ids, dtype=np.int64) if point3D_ids else np.array([], dtype=np.int64)
                            }
                            i += 2  # Skip the points line
                        else:
                            i += 1
                    else:
                        i += 1
            logger.info(f"Read {len(images)} images from text file")
        except Exception as e:
            logger.warning(f"Error reading images.txt: {e}")
    
    # Read points3D.txt
    if points_file.exists():
        try:
            with open(points_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 7:
                            point_id = int(parts[0])
                            xyz = [float(parts[j]) for j in range(1, 4)]
                            rgb = [int(parts[j]) for j in range(4, 7)]
                            error = float(parts[7]) if len(parts) > 7 else 0.0
                            
                            # Track information (image_id, point2d_idx pairs)
                            track = []
                            for j in range(8, len(parts), 2):
                                if j + 1 < len(parts):
                                    img_id = int(parts[j])
                                    point2d_idx = int(parts[j + 1])
                                    track.append((img_id, point2d_idx))
                            
                            points3d[point_id] = {
                                'xyz': xyz,
                                'rgb': rgb,
                                'error': error,
                                'track': track
                            }
            logger.info(f"Read {len(points3d)} 3D points from text file")
        except Exception as e:
            logger.warning(f"Error reading points3D.txt: {e}")
    
    return points3d, cameras, images


def read_colmap_binary_results(sparse_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Read COLMAP binary results and return proper dictionaries"""
    
    # Look for reconstruction directory
    reconstruction_dirs = [d for d in sparse_path.iterdir() if d.is_dir()]
    if not reconstruction_dirs:
        logger.warning("No reconstruction found")
        return {}, {}, {}
    
    # Use first (or largest) reconstruction
    recon_dir = reconstruction_dirs[0]
    
    logger.info(f"Reading COLMAP results from {recon_dir}")
    
    # Check for required output files
    cameras_file = recon_dir / "cameras.bin"
    images_file = recon_dir / "images.bin"  
    points_file = recon_dir / "points3D.bin"
    
    if cameras_file.exists() and images_file.exists() and points_file.exists():
        # Get file sizes for verification
        cameras_size = cameras_file.stat().st_size
        images_size = images_file.stat().st_size
        points_size = points_file.stat().st_size
        
        logger.info(f"COLMAP reconstruction successful:")
        logger.info(f"  - cameras.bin: {cameras_size} bytes")
        logger.info(f"  - images.bin: {images_size} bytes") 
        logger.info(f"  - points3D.bin: {points_size} bytes")
        
        # Read actual COLMAP binary files with comprehensive error handling
        try:
            cameras = read_cameras_binary(cameras_file)
            logger.debug(f"Successfully read {len(cameras)} cameras")
        except Exception as e:
            logger.warning(f"Error reading cameras.bin: {e}")
            cameras = {}
            
        try:
            images = read_images_binary(images_file)
            logger.debug(f"Successfully read {len(images)} images")
        except Exception as e:
            logger.warning(f"Error reading images.bin: {e}")
            images = {}
            
        try:
            points3d = read_points3d_binary(points_file)
            logger.debug(f"Successfully read {len(points3d)} points")
        except Exception as e:
            logger.warning(f"Error reading points3D.bin: {e}")
            points3d = {}
        
        # Log comprehensive reconstruction statistics like hloc
        num_cameras = len(cameras)
        num_images = len(images)
        num_points = len(points3d)
        
        # If binary reading mostly failed, try text format
        if num_cameras == 0 and num_images == 0 and num_points > 0:
            logger.warning("Binary reading failed for cameras and images, trying text format...")
            text_points3d, text_cameras, text_images = read_colmap_text_results(sparse_path)
            
            if text_cameras or text_images:
                logger.info("Successfully read cameras and images from text format")
                cameras = text_cameras
                images = text_images
                # Keep points3d from binary if it worked
                num_cameras = len(cameras)
                num_images = len(images)
        
        if num_cameras > 0 or num_images > 0 or num_points > 0:
            # Calculate statistics
            registered_images = sum(1 for img_data in images.values() if len(img_data.get('point3D_ids', [])) > 0)
            observations = sum(len(point_data.get('track', [])) for point_data in points3d.values())
            mean_track_length = observations / num_points if num_points > 0 else 0.0
            mean_observations_per_image = observations / registered_images if registered_images > 0 else 0.0
            
            # Calculate reprojection errors if available
            errors = [point_data.get('error', 0.0) for point_data in points3d.values()]
            mean_reprojection_error = np.mean(errors) if errors else 0.0
            
            logger.info("=" * 50)
            logger.info("RECONSTRUCTION STATISTICS")
            logger.info("=" * 50)
            logger.info(f"Cameras: {num_cameras}")
            logger.info(f"Images: {num_images}")
            logger.info(f"Registered images: {registered_images}")
            logger.info(f"Points: {num_points}")
            logger.info(f"Observations: {observations}")
            logger.info(f"Mean track length: {mean_track_length:.2f}")
            logger.info(f"Mean observations per image: {mean_observations_per_image:.2f}")
            logger.info(f"Mean reprojection error: {mean_reprojection_error:.4f} px")
            logger.info("=" * 50)
        else:
            logger.warning("No reconstruction data could be read from binary files")
        
        return points3d, cameras, images
    else:
        logger.warning("COLMAP binary reconstruction files not found or could not be read")
        # Try text format as fallback
        logger.info("Attempting to read text format files...")
        text_points3d, text_cameras, text_images = read_colmap_text_results(sparse_path)
        
        if text_cameras or text_images or text_points3d:
            logger.info("Successfully read COLMAP text format results")
            return text_points3d, text_cameras, text_images
        else:
            logger.error("No COLMAP reconstruction files found (binary or text)")
            return {}, {}, {}


def colmap_binary_reconstruction(features: Dict[str, Any], matches: Dict[Tuple[str, str], Any],
                                output_path: Path, image_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Run complete COLMAP reconstruction using binary executable"""
    
    database_path = output_path / "database.db"
    
    # First filter matches with cv2 MAGSAC
    filtered_matches = filter_matches_with_magsac(features, matches)
    
    if not filtered_matches:
        logger.error("No matches passed MAGSAC filtering")
        return {}, {}, {}
    
    # Create database with filtered matches
    image_ids = create_colmap_database(features, filtered_matches, database_path)
    
    # Run COLMAP binary
    success = run_colmap_binary(database_path, image_dir, output_path)
    
    if success:
        # Read results
        points3d, cameras, images = read_colmap_binary_results(output_path / "sparse")
        
        # Convert images dictionary from image_id keys to image_path keys for pipeline compatibility
        images_by_path = {}
        logger.info(f"Converting {len(images)} images from ID-based to path-based keys")
        logger.debug(f"Original images keys (sample): {list(images.keys())[:3]}")
        logger.debug(f"Features keys (sample): {list(features.keys())[:3]}")
        
        conversion_success = 0
        for image_id, img_data in images.items():
            image_name = img_data.get('name', f'unknown_image_{image_id}')
            
            # Find matching image path from original features (they should have matching basenames)
            matching_path = None
            for img_path in features.keys():
                if Path(img_path).name == image_name:
                    matching_path = img_path
                    conversion_success += 1
                    break
            
            # Use matching path if found, otherwise use name as fallback
            key = matching_path if matching_path else image_name
            
            # Ensure all required fields exist
            img_data_copy = img_data.copy()
            if 'name' not in img_data_copy:
                img_data_copy['name'] = image_name
            # Ensure camera_id is preserved from COLMAP results
            if 'camera_id' not in img_data_copy and 'camera_id' in img_data:
                img_data_copy['camera_id'] = img_data['camera_id']
            
            images_by_path[key] = img_data_copy
            
            if not matching_path:
                logger.warning(f"Could not match image {image_name} to original feature path, using name as key")
        
        logger.info(f"Successfully converted {conversion_success}/{len(images)} images to path-based keys")
        if conversion_success < len(images):
            logger.warning(f"Some images could not be matched to original paths")
            
        return points3d, cameras, images_by_path
    else:
        logger.error("COLMAP reconstruction failed")
        return {}, {}, {}