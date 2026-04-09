"""Data export functions for CSV and video export."""

import numpy as np
import pandas as pd


def count_trackpoints(frame_idx, dlc_data, dlc_bodyparts, side):
    """
    Count the number of valid DLC trackpoints for a given eye side.
    
    Args:
        frame_idx: Frame index
        dlc_data: DLC data DataFrame
        dlc_bodyparts: List of body part names
        side: "left" or "right"
    
    Returns:
        Number of valid trackpoints (int)
    """
    if dlc_data is None or frame_idx >= len(dlc_data):
        return 0
    
    row = dlc_data.iloc[int(frame_idx)]
    parts = [p for p in dlc_bodyparts if "pupil" in p and side in p]
    count = 0
    
    for p in parts:
        x, y, l = row[f"{p}_x"], row[f"{p}_y"], row.get(f"{p}_likelihood", 1.0)
        if pd.notna(x) and l > 0.4:
            count += 1
    
    return count


def get_frame_gaze_data(frame_idx, eye_frame, dlc_data, dlc_bodyparts, 
                       comp_enabled, stab_template, stab_ref_point, stab_curr_point, comp_axes,
                       gaze_regressors, reference_centers, eye_models,
                       get_pupil_data_func, determine_best_eye_func, 
                       get_image_compensation_func, apply_compensation_func, predict_gaze_func):
    """
    Extract all gaze and pupil data for a single frame.
    
    Args:
        frame_idx: Frame index
        eye_frame: Current eye frame
        dlc_data, dlc_bodyparts: DLC data
        comp_enabled, stab_template, stab_ref_point, stab_curr_point, comp_axes: Stabilization params
        gaze_regressors, reference_centers, eye_models: Calibration data
        get_pupil_data_func: Function to get pupil data
        determine_best_eye_func: Function to determine best eye
        get_image_compensation_func: Function to get compensation
        apply_compensation_func: Function to apply compensation
        predict_gaze_func: Function to predict gaze
    
    Returns:
        Dictionary with all gaze and pupil data for the frame
    """
    # Initialize data structure
    data = {
        'best_eye': None,
        'best_eye_gaze_x': np.nan,
        'best_eye_gaze_y': np.nan,
        'left_eye_gaze_x': np.nan,
        'left_eye_gaze_y': np.nan,
        'right_eye_gaze_x': np.nan,
        'right_eye_gaze_y': np.nan,
        'left_eye_pupil_x_stab': np.nan,
        'left_eye_pupil_y_stab': np.nan,
        'right_eye_pupil_x_stab': np.nan,
        'right_eye_pupil_y_stab': np.nan,
        'left_eye_pupil_x_raw': np.nan,
        'left_eye_pupil_y_raw': np.nan,
        'right_eye_pupil_x_raw': np.nan,
        'right_eye_pupil_y_raw': np.nan,
        'left_eye_trackpoints': 0,
        'right_eye_trackpoints': 0,
        'stab_shift_dx': np.nan,
        'stab_shift_dy': np.nan,
    }
    
    # Get stabilization shift (only if stabilization is available)
    if eye_frame is not None and comp_enabled.get() and stab_template is not None:
        stab_stats, _ = get_image_compensation_func(
            eye_frame, stab_template, stab_ref_point, stab_curr_point, comp_axes
        )
        dx, dy, _ = stab_stats
        data['stab_shift_dx'] = dx
        data['stab_shift_dy'] = dy
    # If stabilization not available, leave shift as NaN (already initialized)
    
    # Get pupil data for both eyes
    eyes_data = get_pupil_data_func(frame_idx, dlc_data, dlc_bodyparts)
    
    # Count trackpoints for each eye
    data['left_eye_trackpoints'] = count_trackpoints(frame_idx, dlc_data, dlc_bodyparts, "left")
    data['right_eye_trackpoints'] = count_trackpoints(frame_idx, dlc_data, dlc_bodyparts, "right")
    
    # Process each eye
    for side in ["left", "right"]:
        if side not in eyes_data:
            continue
        
        pupil_raw, _ = eyes_data[side]
        pupil_x_raw, pupil_y_raw = pupil_raw
        
        # Store raw pupil coordinates
        data[f'{side}_eye_pupil_x_raw'] = pupil_x_raw
        data[f'{side}_eye_pupil_y_raw'] = pupil_y_raw
        
        # Get stabilized pupil coordinates (only if stabilization is available)
        if eye_frame is not None and comp_enabled.get() and stab_template is not None:
            pupil_stab, _ = apply_compensation_func(
                pupil_x_raw, pupil_y_raw, eye_frame, comp_enabled, stab_template,
                stab_ref_point, stab_curr_point, comp_axes
            )
            pupil_x_stab, pupil_y_stab = pupil_stab
            data[f'{side}_eye_pupil_x_stab'] = pupil_x_stab
            data[f'{side}_eye_pupil_y_stab'] = pupil_y_stab
        # If stabilization not available, leave stabilized columns as NaN (already initialized)
        
        # Get gaze prediction (uses stabilized data internally)
        if eye_frame is not None:
            gaze_pred, _ = predict_gaze_func(
                pupil_x_raw, pupil_y_raw, side, eye_frame, gaze_regressors, reference_centers,
                eye_models, apply_compensation_func, comp_enabled, stab_template,
                stab_ref_point, stab_curr_point, comp_axes
            )
            if gaze_pred:
                data[f'{side}_eye_gaze_x'] = gaze_pred[0]
                data[f'{side}_eye_gaze_y'] = gaze_pred[1]
    
    # Determine best eye
    best_eye = determine_best_eye_func(frame_idx, dlc_data, dlc_bodyparts)
    data['best_eye'] = 'L' if best_eye == 'left' else 'R'
    
    # Set best eye gaze coordinates
    if best_eye in eyes_data:
        if eye_frame is not None:
            pupil_raw, _ = eyes_data[best_eye]
            gaze_pred, _ = predict_gaze_func(
                pupil_raw[0], pupil_raw[1], best_eye, eye_frame, gaze_regressors, reference_centers,
                eye_models, apply_compensation_func, comp_enabled, stab_template,
                stab_ref_point, stab_curr_point, comp_axes
            )
            if gaze_pred:
                data['best_eye_gaze_x'] = gaze_pred[0]
                data['best_eye_gaze_y'] = gaze_pred[1]
    
    return data
