import time
import numpy as np
import cv2
from modules.reversible_arithmetic import cached_reversible_subtraction, simulate_reversible_subtraction

def quantum_arithmetic_edge_detection(image: np.ndarray, use_caching: bool = True, progress_callback=None) -> tuple:
    """
    Computes horizontal and vertical image gradients using simulated reversible quantum subtraction.
    Faithful implementation of Cell 7 in quantumEdgeVision.ipynb.
    
    Gradient Formulation:
    - Horizontal gradient Gx(y, x) = |I(y, x-1) ⊖ I(y, x+1)|
    - Vertical gradient Gy(y, x)   = |I(y-1, x) ⊖ I(y+1, x)|
    where ⊖ represents reversible quantum subtraction.
    
    For an 8x8 image, interior pixels span y in [1, 6] and x in [1, 6] (6x6 = 36 pixels).
    Each pixel computes 2 differences -> exactly 72 reversible quantum operations.
    
    Returns:
        (gx, gy, magnitude, total_operations, execution_time)
    """
    height, width = image.shape
    gx = np.zeros((height, width), dtype=np.uint16)
    gy = np.zeros((height, width), dtype=np.uint16)
    
    start_time = time.time()
    total_operations = 0
    total_pixels = (height - 2) * (width - 2)
    processed_count = 0
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Horizontal pixel difference via reversible quantum subtraction
            left = int(image[y, x - 1])
            right = int(image[y, x + 1])
            if use_caching:
                horizontal = cached_reversible_subtraction(left, right, n=8)
            else:
                horizontal, _ = simulate_reversible_subtraction(left, right, n=8)
            gx[y, x] = horizontal
            
            # Vertical pixel difference via reversible quantum subtraction
            top = int(image[y - 1, x])
            bottom = int(image[y + 1, x])
            if use_caching:
                vertical = cached_reversible_subtraction(top, bottom, n=8)
            else:
                vertical, _ = simulate_reversible_subtraction(top, bottom, n=8)
            gy[y, x] = vertical
            
            total_operations += 2
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count / total_pixels)
                
    # Calculate gradient magnitude: Gx + Gy
    magnitude = gx + gy
    
    # Normalize the result to 0-255 using cv2.NORM_MINMAX
    normalized_magnitude = cv2.normalize(
        magnitude.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    execution_time = time.time() - start_time
    
    return (
        gx,
        gy,
        normalized_magnitude,
        total_operations,
        execution_time
    )

def apply_edge_threshold(magnitude: np.ndarray, threshold: int = 60) -> np.ndarray:
    """
    Applies binary thresholding to produce the final binary edge map.
    Faithful implementation of Cell 9 in quantumEdgeVision.ipynb:
        cv2.threshold(magnitude, THRESHOLD, 255, cv2.THRESH_BINARY)
    """
    _, edge_map = cv2.threshold(
        magnitude,
        threshold,
        255,
        cv2.THRESH_BINARY
    )
    return edge_map
