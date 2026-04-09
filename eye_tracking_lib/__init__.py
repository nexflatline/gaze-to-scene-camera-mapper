"""
Eye Tracking Library

A library of modules for eye tracking, gaze calibration, and data processing.
"""

from .eye_model import EyeModel3D
from .geometry_utils import rotate_point
from .stabilization import (
    get_image_compensation,
    apply_compensation_to_pupil,
    auto_init_stabilization_from_loaded_calibration
)
from .pupil_detection import (
    load_dlc_csv,
    fit_ellipse_center,
    get_pupil_data_all,
    determine_best_eye
)
from .gaze_calibration import (
    compute_calibration,
    predict_gaze
)
from .data_persistence import (
    serialize_pipeline,
    deserialize_pipeline,
    encode_bgr_image_to_b64_png,
    decode_b64_png_to_bgr,
    save_calibration_data,
    load_calibration_data
)
from .data_export import (
    count_trackpoints,
    get_frame_gaze_data
)

__all__ = [
    'EyeModel3D',
    'rotate_point',
    'get_image_compensation',
    'apply_compensation_to_pupil',
    'auto_init_stabilization_from_loaded_calibration',
    'load_dlc_csv',
    'fit_ellipse_center',
    'get_pupil_data_all',
    'determine_best_eye',
    'compute_calibration',
    'predict_gaze',
    'serialize_pipeline',
    'deserialize_pipeline',
    'encode_bgr_image_to_b64_png',
    'decode_b64_png_to_bgr',
    'save_calibration_data',
    'load_calibration_data',
    'count_trackpoints',
    'get_frame_gaze_data',
]
