"""Eye model for gaze angle calculations."""

import numpy as np


class EyeModel3D:
    def __init__(self, eye_radius_px=100):
        self.eye_radius = float(eye_radius_px)

    def pupil_to_gaze_angles(self, pupil_x, pupil_y, reference_center):
        dx = float(pupil_x) - float(reference_center[0])
        dy = float(pupil_y) - float(reference_center[1])
        norm_x = np.clip(dx / self.eye_radius, -0.99, 0.99)
        norm_y = np.clip(dy / self.eye_radius, -0.99, 0.99)
        theta = np.arcsin(norm_x)
        phi = np.arcsin(-norm_y)
        return theta, phi

    def get_gaze_features(self, pupil_x, pupil_y, reference_center):
        theta, phi = self.pupil_to_gaze_angles(pupil_x, pupil_y, reference_center)
        features = [
            theta, phi,
            theta**2, phi**2, theta * phi,
            theta**3, phi**3,
            (theta**2) * phi,
            theta * (phi**2),
        ]
        return np.asarray(features, dtype=np.float64)
