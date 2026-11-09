import cv2
import numpy as np

def extract_boundaries(edge_image: np.ndarray, original: np.ndarray) -> tuple:
    """
    Extracts topological object boundaries from binary edge maps using OpenCV contours.
    Faithful implementation of Cell 10 in quantumEdgeVision.ipynb.
    
    Operations:
    1. cv2.findContours with RETR_EXTERNAL and CHAIN_APPROX_SIMPLE
    2. Overlays extracted contours in vibrant green (0, 255, 0) onto 3-channel BGR original image
    
    Returns:
        (boundary_image_bgr, contours)
    """
    if edge_image is None or original is None:
        raise ValueError("Inputs to extract_boundaries cannot be None")
        
    contours, _ = cv2.findContours(
        edge_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Create 3-channel BGR representation of original grayscale image
    if len(original.shape) == 2:
        output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        output = original.copy()
        
    # Draw green contours with thickness 1 (suitable for 8x8 image resolution)
    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        1
    )
    
    return output, contours

def get_contour_metrics(contours: list) -> list:
    """
    Computes geometric properties (area, perimeter, bounding box) for each detected contour.
    Provides academic clarity on boundary extraction.
    """
    metrics = []
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, closed=True)
        x, y, w, h = cv2.boundingRect(c)
        metrics.append({
            "contour_id": idx + 1,
            "num_points": len(c),
            "area": float(area),
            "perimeter": round(float(perimeter), 2),
            "bounding_box": f"(X={x}, Y={y}, W={w}, H={h})"
        })
    return metrics
