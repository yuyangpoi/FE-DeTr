from metavision_core_ml.data.camera_poses import CameraPoseGenerator, generate_image_homography, get_flow


class My_CameraPoseGenerator(CameraPoseGenerator):
    """
    CameraPoseGenerator generates a series of continuous homographies
    with interpolation.

    Args:
        height (int): height of image
        width (int): width of image
        max_frames (int): maximum number of poses
        pause_probability (float): probability that the sequence contains a pause
        max_optical_flow_threshold (float): maximum optical flow between two consecutive frames
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames
    """
    def get_homography_ts_pose(self, height, width):
        """
        Returns next homography
        """
        rvec2 = self.rvecs[self.time]   # rotation vector
        tvec2 = self.tvecs[self.time]   # translation vector
        ts = self.times[self.time]
        H = generate_image_homography(rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv)

        self.time += 1
        return H, ts, rvec2, tvec2















