"""Image stabilization functions using template matching."""

import math
import cv2


def get_image_compensation(current_bgr, stab_template, stab_ref_point, stab_curr_point, comp_axes):
    """
    Compute image compensation using template matching.
    
    Args:
        current_bgr: Current frame (BGR image)
        stab_template: Template image for matching
        stab_ref_point: Reference point (x, y)
        stab_curr_point: Current tracked point (x, y) - can be None
        comp_axes: Dict with 'x', 'y' BooleanVars indicating which axes to compensate
    
    Returns:
        Tuple of ((dx, dy, da), (curr_cx, curr_cy)) where:
        - (dx, dy, da) is the compensation shift
        - (curr_cx, curr_cy) is the current matched point (or None)
    """
    if stab_template is None or stab_ref_point is None or current_bgr is None:
        return (0.0, 0.0, 0.0), None

    curr_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(stab_template, cv2.COLOR_BGR2GRAY)

    h, w = curr_gray.shape
    th, tw = tmpl_gray.shape

    margin = 100
    last_x, last_y = stab_curr_point if stab_curr_point else stab_ref_point

    search_x1 = max(0, int(last_x - margin))
    search_y1 = max(0, int(last_y - margin))
    search_x2 = min(w, int(last_x + margin))
    search_y2 = min(h, int(last_y + margin))

    search_region = curr_gray[search_y1:search_y2, search_x1:search_x2]

    if search_region.shape[0] < th or search_region.shape[1] < tw:
        search_region = curr_gray
        search_x1, search_y1 = 0, 0

    res = cv2.matchTemplate(search_region, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    tl_x, tl_y = max_loc
    curr_cx = tl_x + search_x1 + tw // 2
    curr_cy = tl_y + search_y1 + th // 2

    min_corr = 0.6
    valid_match = (max_val >= min_corr)

    max_step_px = 50.0
    move_dist = math.hypot(curr_cx - last_x, curr_cy - last_y)
    valid_move = (move_dist <= max_step_px)

    if valid_match and valid_move:
        new_curr_point = (curr_cx, curr_cy)
    else:
        new_curr_point = (last_x, last_y)
        curr_cx, curr_cy = last_x, last_y

    ref_cx, ref_cy = stab_ref_point
    dx = curr_cx - ref_cx
    dy = curr_cy - ref_cy

    if not comp_axes["x"].get(): dx = 0.0
    if not comp_axes["y"].get(): dy = 0.0

    return (dx, dy, 0.0), new_curr_point


def apply_compensation_to_pupil(pupil_x, pupil_y, frame_bgr, comp_enabled, stab_template, 
                                stab_ref_point, stab_curr_point, comp_axes):
    """
    Apply image compensation to pupil coordinates.
    
    Args:
        pupil_x, pupil_y: Raw pupil coordinates
        frame_bgr: Current frame
        comp_enabled: BooleanVar indicating if compensation is enabled
        stab_template: Template image
        stab_ref_point: Reference point
        stab_curr_point: Current tracked point
        comp_axes: Dict with compensation axes settings
    
    Returns:
        Tuple of ((final_x, final_y), (dx, dy, da)) - compensated coordinates and stats
    """
    if not comp_enabled.get() or stab_template is None:
        return (pupil_x, pupil_y), (0, 0, 0)

    stats, new_curr_point = get_image_compensation(frame_bgr, stab_template, stab_ref_point, 
                                                   stab_curr_point, comp_axes)
    dx, dy, da = stats

    final_x = pupil_x - dx
    final_y = pupil_y - dy

    return (final_x, final_y), stats


def auto_init_stabilization_from_loaded_calibration(raw_eye_frame, stab_template,
                                                      stab_ref_point, stab_curr_point):
    """
    Auto-locate stabilization landmark in a new video frame via full-frame template search.

    Args:
        raw_eye_frame: Current eye frame (BGR)
        stab_template: Template to search for (BGR patch)
        stab_ref_point: Previous reference point (unused; kept for API compatibility)
        stab_curr_point: Previous tracked point (unused; kept for API compatibility)

    Returns:
        Tuple of (success: bool, new_ref_point, new_curr_point)
    """
    if stab_template is None:
        return False, None, None

    if raw_eye_frame is None:
        return False, None, None

    frame_gray = cv2.cvtColor(raw_eye_frame, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(stab_template, cv2.COLOR_BGR2GRAY)

    fh, fw = frame_gray.shape
    th, tw = tmpl_gray.shape

    if fh < th or fw < tw:
        return False, None, None

    res = cv2.matchTemplate(frame_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.7:
        return False, None, None

    tl_x, tl_y = max_loc
    cx = tl_x + tw // 2
    cy = tl_y + th // 2

    new_ref_point = (cx, cy)
    new_curr_point = (cx, cy)

    return True, new_ref_point, new_curr_point
