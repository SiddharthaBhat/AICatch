import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

LOGGER = logging.getLogger(__name__)

def classify_pixel_hsv(h: int, s: int, v: int) -> str:
    """Heuristic mapping of raw OpenCV HSV values to human-readable color categories.
    OpenCV HSV range: H is [0, 180], S is [0, 255], V is [0, 255].
    """
    # 1. Grayscale Check
    if v < 45:
        return "black"
    if s < 30 and v > 200:
        return "white"
    if s < 40:
        return "gray"
        
    # 2. Hue range classifications
    # Red has wrap-around at 0 and 180 (represented as 0-10 and 170-180 in OpenCV)
    if h < 10 or h >= 170:
        if v > 150 and s > 50 and s < 150:
            return "pink"
        return "red"
    if h < 22:
        # Brown is a low-value, low-saturation orange
        if v < 120 and s > 40:
            return "brown"
        return "orange"
    if h < 38:
        if v < 100:
            return "brown"
        return "yellow"
    if h < 75:
        return "green"
    if h < 95:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 155:
        return "purple"
    
    # Pink falls between purple and red
    return "pink"

def extract_body_colors(crop_bgr: np.ndarray) -> Dict[str, Any]:
    """Analyzes the upper and lower body of a person crop to return dominant and secondary colors.
    Excludes outer 15% padding context to minimize background noise.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "dominant_upper_color": "gray",
            "secondary_upper_color": "gray",
            "dominant_lower_color": "gray",
            "secondary_lower_color": "gray",
            "color_confidence": 0.0
        }
        
    h_img, w_img = crop_bgr.shape[:2]
    # Central crop box excluding 15% padding context
    x1 = max(0, int(0.15 * w_img))
    x2 = min(w_img, int(0.85 * w_img))
    y1 = max(0, int(0.15 * h_img))
    y2 = min(h_img, int(0.85 * h_img))
    
    # Ensure minimum width/height for division
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return {
            "dominant_upper_color": "gray",
            "secondary_upper_color": "gray",
            "dominant_lower_color": "gray",
            "secondary_lower_color": "gray",
            "color_confidence": 0.0
        }
        
    center_h = y2 - y1
    mid_y = y1 + int(center_h * 0.5)
    
    upper_body = crop_bgr[y1:mid_y, x1:x2]
    lower_body = crop_bgr[mid_y:y2, x1:x2]
    
    def get_dominant_colors(region_bgr) -> Tuple[str, str, float]:
        if region_bgr.size == 0:
            return "gray", "gray", 0.0
            
        # Downsample to speed up and filter out single-pixel noise
        resized = cv2.resize(region_bgr, (30, 30))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        counts = {}
        for row in hsv:
            for pixel in row:
                color = classify_pixel_hsv(pixel[0], pixel[1], pixel[2])
                counts[color] = counts.get(color, 0) + 1
                
        total = sum(counts.values())
        if total == 0:
            return "gray", "gray", 0.0
            
        sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_colors[0][0]
        secondary = sorted_colors[1][0] if len(sorted_colors) > 1 else dominant
        
        confidence = counts[dominant] / total
        return dominant, secondary, confidence

    dom_upper, sec_upper, conf_upper = get_dominant_colors(upper_body)
    dom_lower, sec_lower, conf_lower = get_dominant_colors(lower_body)
    
    return {
        "dominant_upper_color": dom_upper,
        "secondary_upper_color": sec_upper,
        "dominant_lower_color": dom_lower,
        "secondary_lower_color": sec_lower,
        "color_confidence": round(float((conf_upper + conf_lower) / 2.0), 2)
    }

def extract_vehicle_colors(crop_bgr: np.ndarray) -> Dict[str, Any]:
    """Analyzes a vehicle crop to return dominant and secondary colors."""
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "dominant_upper_color": "gray",
            "secondary_upper_color": "gray",
            "dominant_lower_color": None,
            "secondary_lower_color": None,
            "color_confidence": 0.0
        }
        
    h_img, w_img = crop_bgr.shape[:2]
    # Central crop box excluding 15% padding context
    x1 = max(0, int(0.15 * w_img))
    x2 = min(w_img, int(0.85 * w_img))
    y1 = max(0, int(0.15 * h_img))
    y2 = min(h_img, int(0.85 * h_img))
    
    region = crop_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return {
            "dominant_upper_color": "gray",
            "secondary_upper_color": "gray",
            "dominant_lower_color": None,
            "secondary_lower_color": None,
            "color_confidence": 0.0
        }
        
    # Downsample and convert
    resized = cv2.resize(region, (30, 30))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    counts = {}
    for row in hsv:
        for pixel in row:
            color = classify_pixel_hsv(pixel[0], pixel[1], pixel[2])
            counts[color] = counts.get(color, 0) + 1
            
    total = sum(counts.values())
    if total == 0:
        return {
            "dominant_upper_color": "gray",
            "secondary_upper_color": "gray",
            "dominant_lower_color": None,
            "secondary_lower_color": None,
            "color_confidence": 0.0
        }
        
    sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    dominant = sorted_colors[0][0]
    secondary = sorted_colors[1][0] if len(sorted_colors) > 1 else dominant
    confidence = counts[dominant] / total
    
    return {
        "dominant_upper_color": dominant,
        "secondary_upper_color": secondary,
        "dominant_lower_color": None,
        "secondary_lower_color": None,
        "color_confidence": round(float(confidence), 2)
    }

def evaluate_crop_quality(crop: np.ndarray, sharpness: float) -> Tuple[bool, float]:
    """Rejects extremely small, blurry, or unusable crops. Calculates a quality score in [0.0, 1.0]."""
    if crop is None or crop.size == 0:
        return False, 0.0
        
    h, w = crop.shape[:2]
    # Blurry, tiny or highly-skewed aspect ratio checks
    if w < 32 or h < 32:
        return False, 0.0
        
    aspect = w / h
    if aspect > 6.0 or aspect < 0.16:
        return False, 0.0
        
    if sharpness < 15.0:
        return False, 0.0
        
    # Calculate a composite quality score (0.0 to 1.0)
    # 50% resolution score (maxes out at 120x120 crop)
    res_score = min(1.0, (w * h) / (120 * 120)) * 0.5
    # 50% sharpness score (maxes out at Laplacian variance 200.0)
    sharp_score = min(1.0, sharpness / 200.0) * 0.5
    
    quality_score = float(res_score + sharp_score)
    return True, round(quality_score, 3)
