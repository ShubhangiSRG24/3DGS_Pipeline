# -*- coding: utf-8 -*-
"""
Combined script: PLY rotation + Concave hull generation
Combines PLYRotAutoScrpt.py and concavehull_cam.py functionality

@author: Boyack
"""

import json
import plyfile
import numpy as np
# import tkinter as tk
# from tkinter import filedialog as fd
from scipy.spatial.transform import Rotation as R
# import matplotlib.pyplot as plt
from concave_hull import concave_hull, concave_hull_indexes
import argparse


class CombinedProcessor():
    def __init__(self, input_file, camera_file, output_file):
        self.in_file = input_file
        self.cm_file = camera_file
        self.ot_file = output_file
        print(f"Input PLY: {self.in_file}")
        print(f"Camera JSON: {self.cm_file}")
        print(f"Output PLY: {self.ot_file}")

        # Run the processing pipeline
        self.process()

    def process(self):
        # Step 1: Load files
        self.load_ply_file()
        self.load_json_file()

        # Step 2: Calculate rotation alignment
        self.calculate_alignment()

        # Step 3: Rotate PLY file
        self.rotate_ply()

        # Step 4: Rotate cameras
        self.rotate_cameras()

        # Step 5: Generate concave hull
        self.generate_concave_hull()

    def load_ply_file(self):
        """Load PLY file data"""
        self.filename = self.in_file
        self.plydata = plyfile.PlyData.read(self.filename)['vertex']
        self.prop_names = [val.name for val in self.plydata.properties]
        self.pc_points_initial = np.c_[self.plydata['x'], self.plydata['y'], self.plydata['z']]
        self.pc_norms_initial = np.c_[self.plydata['nx'], self.plydata['ny'], self.plydata['nz']]
        self.pc_scale_initial = np.c_[self.plydata['scale_0'], self.plydata['scale_1'], self.plydata['scale_2']]
        self.pc_quarts_initial = np.c_[self.plydata['rot_1'], self.plydata['rot_2'], self.plydata['rot_3'], self.plydata['rot_0']]
        self.pc_points_current = np.arange(len(self.pc_points_initial))

    def load_json_file(self):
        """Load camera JSON file"""
        self.filenameJSON = self.cm_file
        with open(self.filenameJSON, 'r') as loc_cam:
            self.cam_locs = json.load(loc_cam)

    def calculate_alignment(self):
        """Calculate rotation alignment based on camera positions"""
        self.cam_pnts_xyz = np.array([cam_num['position'] for cam_num in self.cam_locs])
        G = self.cam_pnts_xyz.sum(axis=0) / self.cam_pnts_xyz.shape[0]

        # Run SVD
        u, s, vh = np.linalg.svd(self.cam_pnts_xyz - G)

        # Unitary normal vector
        self.u_norm_flr = vh[2, :]
        self.u_norm_wll = self.cam_pnts_xyz[0] - G

        # Floor is negative y direction, wall is positive x
        self.flr_vec = [0, -1, 0]
        self.wll_vec = [1, 0, 0]

        # Generate rotation matrix to align floor and wall
        self.algn_rot_mat, _ = R.align_vectors(
            np.c_[self.flr_vec, self.wll_vec].T,
            np.c_[self.u_norm_flr, self.u_norm_wll].T,
            weights=[10000, 1]
        )

    def rotate_ply(self):
        """Apply rotation to PLY point cloud"""
        self.pc_points_new = self.algn_rot_mat.apply(self.pc_points_initial)
        self.pc_norms_new = self.algn_rot_mat.apply(self.pc_norms_initial)
        self.pc_quarts_new = self.algn_rot_mat.as_quat()

        # Update PLY data
        for points in self.pc_points_current:
            self.plydata['x'][points] = self.pc_points_new[points][0]
            self.plydata['y'][points] = self.pc_points_new[points][1]
            self.plydata['z'][points] = self.pc_points_new[points][2]
            self.plydata['nx'][points] = self.pc_norms_new[points][0]
            self.plydata['ny'][points] = self.pc_norms_new[points][1]
            self.plydata['nz'][points] = self.pc_norms_new[points][2]
            qt_vls = self.quat_multiply(self.pc_quarts_initial[points], self.pc_quarts_new)
            self.plydata['rot_0'][points] = qt_vls[0]
            self.plydata['rot_1'][points] = qt_vls[1]
            self.plydata['rot_2'][points] = qt_vls[2]
            self.plydata['rot_3'][points] = qt_vls[3]

        # Save rotated PLY
        dtypeval = [(val.name, val.val_dtype) for val in self.plydata.properties]
        self.new_array = np.array(
            [tuple([self.plydata[key][deals] for key in self.prop_names]) for deals in self.pc_points_current],
            dtype=dtypeval
        )
        el = plyfile.PlyElement.describe(self.new_array, 'vertex')
        plyfile.PlyData([el]).write(self.ot_file)
        print(f"Saved rotated PLY to {self.ot_file}")

    def rotate_cameras(self):
        """Apply rotation to camera positions and orientations"""
        self.rotated_cams = []
        cam_pnts_xyz_new = self.algn_rot_mat.apply(self.cam_pnts_xyz)
        cam_rot_new = [self.algn_rot_mat.apply(np.array(cam_num['rotation']).T) for cam_num in self.cam_locs]

        # Center cameras around mean
        mean_cam_pnt = np.mean(cam_pnts_xyz_new, axis=0)
        for points in range(len(cam_pnts_xyz_new)):
            cam_pnts_xyz_new[points] -= np.array(mean_cam_pnt)

        # Store new camera data
        for ii, cam in enumerate(self.cam_locs):
            rotated_cam = cam.copy()
            rotated_cam['position'] = cam_pnts_xyz_new[ii].tolist()
            rotated_cam['rotation'] = cam_rot_new[ii].tolist()
            self.rotated_cams.append(rotated_cam)

        # Save rotated cameras
        rotated_json_path = self.cm_file.replace('.json', '_rotated.json')
        with open(rotated_json_path, 'w') as f:
            json.dump(self.rotated_cams, f, indent=2)
        print(f"Saved rotated cameras to {rotated_json_path}")

        # Store for concave hull generation
        self.cam_pnts_xyz_new = cam_pnts_xyz_new

    def generate_concave_hull(self):
        """Generate concave hull from rotated camera positions"""
        # Use only x and z coordinates (floor plane)
        ccv_hl_pnts = self.cam_pnts_xyz_new[:, 0:3:2]

        # Generate concave hull
        idxes = concave_hull_indexes(
            ccv_hl_pnts[:, :2],
            length_threshold=0.01,
        )

        # Visualize hull (commented out for pipeline usage)
        # for f, t in zip(idxes[:-1], idxes[1:]):
        #     seg = ccv_hl_pnts[[f, t]]
        #     plt.plot(seg[:, 0], seg[:, 1], "k-", alpha=1)
        # plt.title("Concave Hull of Camera Positions")
        # plt.xlabel("X")
        # plt.ylabel("Z")
        # plt.show()

        # Export hull data as JSON
        hull_data = {
            'hull_points_2d': ccv_hl_pnts[idxes].tolist()
        }

        output_filename = self.cm_file.replace('.json', '_hull.json')
        with open(output_filename, 'w') as f:
            json.dump(hull_data, f, indent=2)

        print(f"Hull data exported to: {output_filename}")
        print(f"Hull has {len(idxes)} vertices")

    def quat_multiply(self, quaternion0, quaternion1):
        """Multiply two quaternions"""
        x0, y0, z0, w0 = np.split(quaternion0, 4, axis=-1)
        x1, y1, z1, w1 = np.split(quaternion1, 4, axis=-1)
        return np.concatenate((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined script that rotates PLY file and generates concave hull from camera positions"
    )
    parser.add_argument(
        "-i", "--input_file_path",
        required=True,
        type=str,
        help="Path to the 3DGS PLY input file"
    )
    parser.add_argument(
        "-c", "--camera_file_path",
        required=True,
        type=str,
        help="Path to the camera JSON file"
    )
    parser.add_argument(
        "-o", "--output_file_path",
        required=True,
        type=str,
        help="Path for the rotated 3DGS PLY output file"
    )

    args = parser.parse_args()

    processor = CombinedProcessor(
        args.input_file_path,
        args.camera_file_path,
        args.output_file_path
    )

    print("Processing completed successfully!")
