"""Data persistence functions for saving/loading calibration data."""

import json
import base64
import hashlib
import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def serialize_pipeline(pipe):
    """Serialize sklearn pipeline to dictionary."""
    scaler = pipe.steps[0][1]
    ridge = pipe.steps[1][1]
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "scaler_var": getattr(scaler, "var_", None).tolist() if getattr(scaler, "var_", None) is not None else None,
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
    }


def deserialize_pipeline(data, alpha):
    """Deserialize dictionary to sklearn pipeline."""
    scaler = StandardScaler()
    ridge = Ridge(alpha=alpha)
    pipe = make_pipeline(scaler, ridge)

    scaler.mean_ = np.asarray(data["scaler_mean"])
    scaler.scale_ = np.asarray(data["scaler_scale"])
    scaler.var_ = np.asarray(data["scaler_var"]) if data.get("scaler_var") else scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)

    ridge.coef_ = np.asarray(data["ridge_coef"])
    ridge.intercept_ = float(data["ridge_intercept"])

    return pipe


def encode_bgr_image_to_b64_png(img_bgr):
    """Encode BGR image to base64 PNG string."""
    if img_bgr is None:
        return None, None

    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return None, None

    b = buf.tobytes()
    b64 = base64.b64encode(b).decode("ascii")
    sha = hashlib.sha256(b).hexdigest()

    return b64, sha


def decode_b64_png_to_bgr(b64_str):
    """Decode base64 PNG string to BGR image."""
    if not b64_str:
        return None

    b = base64.b64decode(b64_str.encode("ascii"))
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return img


def save_calibration_data(ridge_alpha, reference_centers, gaze_regressors, 
                          stab_template, stab_ref_point, template_size):
    """
    Prepare calibration data dictionary for saving.
    
    Args:
        ridge_alpha: Ridge regression alpha
        reference_centers: Dict mapping side to (x, y)
        gaze_regressors: Dict mapping side to (model_x, model_y)
        stab_template: Stabilization template image
        stab_ref_point: Stabilization reference point
        template_size: Template size
    
    Returns:
        Dictionary ready for JSON serialization
    """
    data = {
        "ridge_alpha": ridge_alpha,
        "reference_centers": reference_centers,
        "models": {}
    }

    for side, models in gaze_regressors.items():
        if models:
            data["models"][side] = {
                "x": serialize_pipeline(models[0]),
                "y": serialize_pipeline(models[1])
            }

    stab_template_b64, stab_template_sha256 = encode_bgr_image_to_b64_png(stab_template)
    data["stabilization"] = {
        "enabled": (stab_template is not None and stab_ref_point is not None),
        "stab_ref_point": [float(stab_ref_point[0]), float(stab_ref_point[1])] if stab_ref_point else None,
        "template_size": int(template_size),
        "stab_template_b64_png": stab_template_b64,
        "stab_template_sha256": stab_template_sha256,
    }

    return data


def load_calibration_data(data):
    """
    Load calibration data from dictionary.
    
    Args:
        data: Dictionary loaded from JSON
    
    Returns:
        Tuple of (ridge_alpha, reference_centers, gaze_regressors, 
                 stab_template, stab_ref_point, template_size)
    """
    ridge_alpha = data.get("ridge_alpha", 0.1)
    
    reference_centers = {"left": (320.0, 240.0), "right": (320.0, 240.0)}
    if "reference_centers" in data:
        reference_centers["left"] = tuple(data["reference_centers"].get("left", [320, 240]))
        reference_centers["right"] = tuple(data["reference_centers"].get("right", [320, 240]))

    gaze_regressors = {"left": None, "right": None}
    if "models" in data:
        for side, mdata in data["models"].items():
            mx = deserialize_pipeline(mdata["x"], ridge_alpha)
            my = deserialize_pipeline(mdata["y"], ridge_alpha)
            gaze_regressors[side] = (mx, my)

    stab = data.get("stabilization", {})
    stab_template = None
    stab_ref_point = None
    template_size = 30
    
    if stab.get("enabled"):
        template_size = int(stab.get("template_size", 30))
        stab_template = decode_b64_png_to_bgr(stab.get("stab_template_b64_png"))

        rp = stab.get("stab_ref_point")
        if rp and len(rp) == 2:
            stab_ref_point = (float(rp[0]), float(rp[1]))

    return (ridge_alpha, reference_centers, gaze_regressors, 
            stab_template, stab_ref_point, template_size)
