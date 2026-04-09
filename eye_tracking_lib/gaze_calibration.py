"""Gaze calibration and prediction functions."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_calibration(calibration_pairs, eye_models, ridge_alpha=0.1):
    """
    Compute gaze calibration models for left and right eyes.
    
    Args:
        calibration_pairs: List of (pupil_x, pupil_y, scene_x, scene_y, side, frame_idx) tuples
        eye_models: Dict mapping "left"/"right" to EyeModel3D instances
        ridge_alpha: Ridge regression alpha parameter
    
    Returns:
        Tuple of (gaze_regressors dict, reference_centers dict)
        gaze_regressors maps side to (model_x, model_y) tuple
        reference_centers maps side to (x, y) tuple
    """
    if len(calibration_pairs) < 10:
        raise ValueError("Need at least 10 calibration points.")

    gaze_regressors = {"left": None, "right": None}
    reference_centers = {"left": (320.0, 240.0), "right": (320.0, 240.0)}

    for side in ["left", "right"]:
        pts = [p for p in calibration_pairs if p[4] == side]
        if len(pts) < 5: 
            continue

        arr = np.array(pts)
        pupil_coords = arr[:, 0:2].astype(float)
        scene_coords = arr[:, 2:4].astype(float)

        ref_center = np.mean(pupil_coords, axis=0)
        reference_centers[side] = tuple(ref_center)

        feats = []
        for i in range(len(pupil_coords)):
            f = eye_models[side].get_gaze_features(pupil_coords[i, 0], pupil_coords[i, 1], ref_center)
            feats.append(f)

        feats = np.array(feats)

        if np.isnan(feats).any():
            print(f"Warning: NaNs detected in features for {side}. Cleaning data.")
            mask = ~np.isnan(feats).any(axis=1)
            feats = feats[mask]
            scene_coords = scene_coords[mask]

            if len(feats) < 5:
                print(f"Not enough valid data for {side} after cleaning.")
                continue

        mx = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)).fit(feats, scene_coords[:, 0])
        my = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)).fit(feats, scene_coords[:, 1])

        gaze_regressors[side] = (mx, my)

    return gaze_regressors, reference_centers


def predict_gaze(pupil_x, pupil_y, side, frame_bgr, gaze_regressors, reference_centers, 
                 eye_models, apply_compensation_func, comp_enabled, stab_template,
                 stab_ref_point, stab_curr_point, comp_axes):
    """
    Predict gaze target from pupil position.
    
    Args:
        pupil_x, pupil_y: Raw pupil coordinates
        side: "left" or "right"
        frame_bgr: Current eye frame
        gaze_regressors: Dict mapping side to (model_x, model_y)
        reference_centers: Dict mapping side to (x, y)
        eye_models: Dict mapping side to EyeModel3D
        apply_compensation_func: Function to apply compensation
        comp_enabled, stab_template, stab_ref_point, stab_curr_point, comp_axes: Compensation params
    
    Returns:
        Tuple of ((gaze_x, gaze_y), (dx, dy, da)) or (None, stats) if prediction fails
    """
    models = gaze_regressors.get(side)
    if models is None: 
        return None, (0, 0, 0)

    (adj_px, adj_py), stats = apply_compensation_func(
        pupil_x, pupil_y, frame_bgr, comp_enabled, stab_template,
        stab_ref_point, stab_curr_point, comp_axes
    )

    if np.isnan(adj_px) or np.isnan(adj_py):
        return None, stats

    ref_center = reference_centers.get(side, (320, 240))
    features = eye_models[side].get_gaze_features(adj_px, adj_py, ref_center).reshape(1, -1)

    if np.isnan(features).any():
        return None, stats

    sx = float(models[0].predict(features)[0])
    sy = float(models[1].predict(features)[0])

    return (int(round(sx)), int(round(sy))), stats
