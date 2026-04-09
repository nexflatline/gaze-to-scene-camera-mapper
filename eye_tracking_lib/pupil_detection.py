"""Pupil detection and DLC data processing functions."""

import cv2
import numpy as np
import pandas as pd


def load_dlc_csv(path):
    """
    Load DLC CSV file and extract body parts.
    
    Args:
        path: Path to DLC CSV file
    
    Returns:
        Tuple of (dlc_data DataFrame, dlc_bodyparts list)
    """
    df = pd.read_csv(path, header=[1, 2])
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    dlc_data = df.apply(pd.to_numeric, errors="coerce")

    parts = set(col[:-2] for col in df.columns if col.endswith("_x"))
    dlc_bodyparts = sorted(list(parts))

    return dlc_data, dlc_bodyparts


def fit_ellipse_center(points):
    """
    Fit ellipse to points and return center and ellipse parameters.
    
    Args:
        points: Array of (x, y) points
    
    Returns:
        Tuple of (center, ellipse_params) where ellipse_params can be None
    """
    if len(points) < 5:
        return None, None
    
    try:
        ellipse = cv2.fitEllipse(points.astype(np.float32))
        center, axes, angle = ellipse
        if axes[0] > 0 and axes[1] > 0 and axes[0] < 10000 and axes[1] < 10000:
            return center, ellipse
        else:
            x_m, y_m = np.mean(points, axis=0)
            return (x_m, y_m), None
    except cv2.error:
        x_m, y_m = np.mean(points, axis=0)
        return (x_m, y_m), None


def get_pupil_data_all(frame_idx, dlc_data, dlc_bodyparts):
    """
    Get pupil data for all eyes with ellipse parameters.
    
    Args:
        frame_idx: Frame index
        dlc_data: DLC data DataFrame
        dlc_bodyparts: List of body part names
    
    Returns:
        Dict mapping side ("left"/"right") to (pupil_center, ellipse_params)
    """
    if dlc_data is None or frame_idx >= len(dlc_data): 
        return {}

    row = dlc_data.iloc[int(frame_idx)]
    res = {}

    for side in ["left", "right"]:
        parts = [p for p in dlc_bodyparts if "pupil" in p and side in p]
        pts = []
        for p in parts:
            x, y, l = row[f"{p}_x"], row[f"{p}_y"], row.get(f"{p}_likelihood", 1.0)
            if pd.notna(x) and l > 0.4: 
                pts.append([x, y])

        if len(pts) >= 5:
            pts_array = np.array(pts)
            ctr, ellipse_params = fit_ellipse_center(pts_array)
            if ctr: 
                res[side] = (ctr, ellipse_params)

    return res


def determine_best_eye(frame_idx, dlc_data, dlc_bodyparts):
    """
    Determine which eye has better tracking quality.
    
    Args:
        frame_idx: Frame index
        dlc_data: DLC data DataFrame
        dlc_bodyparts: List of body part names
    
    Returns:
        "left" or "right"
    """
    eyes = get_pupil_data_all(frame_idx, dlc_data, dlc_bodyparts)

    if "left" in eyes and "right" not in eyes:
        return "left"
    elif "right" in eyes and "left" not in eyes:
        return "right"
    elif "left" in eyes and "right" in eyes:
        left_params = eyes["left"][1]
        right_params = eyes["right"][1]

        # Prefer the eye with a valid ellipse fit
        if left_params is not None and right_params is None:
            return "left"
        elif right_params is not None and left_params is None:
            return "right"
        elif left_params is not None and right_params is not None:
            # Both have ellipses: choose larger area (more points fit the ellipse)
            left_area = left_params[1][0] * left_params[1][1]
            right_area = right_params[1][0] * right_params[1][1]
            return "left" if left_area >= right_area else "right"
        else:
            return "left"  # Neither has an ellipse; default to left
    else:
        return "left"  # Default
