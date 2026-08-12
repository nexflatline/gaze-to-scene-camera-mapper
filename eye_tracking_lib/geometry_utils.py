"""Geometry utility functions for coordinate transformations."""

import numpy as np


def rotate_point(pt, center, angle_rad):
    """Rotates a point (x,y) around center (cx,cy) by angle_rad."""
    x, y = pt
    cx, cy = center
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    tx, ty = x - cx, y - cy
    rx = tx * c - ty * s
    ry = tx * s + ty * c
    return rx + cx, ry + cy
