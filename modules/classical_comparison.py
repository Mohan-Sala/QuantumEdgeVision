import time
import numpy as np
import cv2
from modules.boundary_extraction import extract_boundaries

def classical_sobel_edge_detection(image: np.ndarray) -> tuple:
    """
    Computes classical edge gradients using the standard 3x3 Sobel operator.
    Faithful implementation of Cell 8 in quantumEdgeVision.ipynb.
    
    Returns:
        (gx, gy, magnitude, execution_time)
    """
    if image is None:
        raise ValueError("Cannot perform Sobel detection on a None image")
        
    start_time = time.time()
    
    # Horizontal Sobel derivative (3x3 kernel)
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )
    
    # Vertical Sobel derivative (3x3 kernel)
    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )
    
    # Gradient magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    
    # Normalize to 0-255 using cv2.NORM_MINMAX
    normalized_magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    execution_time = time.time() - start_time
    
    return gx, gy, normalized_magnitude, execution_time

def compare_quantum_and_classical(
    small_image: np.ndarray,
    quantum_edges: np.ndarray,
    quantum_contours: list,
    quantum_time: float,
    quantum_ops: int,
    threshold: int = 60
) -> dict:
    """
    Performs rigorous side-by-side comparison between Quantum-Assisted and Classical Sobel algorithms.
    Computes edge map correlation, Dice similarity, Jaccard Index (IoU), and execution metrics.
    """
    # Run classical Sobel
    classical_gx, classical_gy, classical_magnitude, classical_time = classical_sobel_edge_detection(small_image)
    
    # Threshold classical magnitude identically to quantum magnitude
    _, classical_edges = cv2.threshold(
        classical_magnitude,
        threshold,
        255,
        cv2.THRESH_BINARY
    )
    
    # Extract boundaries
    classical_boundary, classical_contours = extract_boundaries(
        classical_edges,
        small_image
    )
    
    # Compute quantitative similarity metrics
    q_binary = (quantum_edges > 0).astype(np.uint8)
    c_binary = (classical_edges > 0).astype(np.uint8)
    
    intersection = np.logical_and(q_binary, c_binary).sum()
    union = np.logical_or(q_binary, c_binary).sum()
    iou = float(intersection / union) if union > 0 else 1.0
    
    dice = float(2.0 * intersection / (q_binary.sum() + c_binary.sum())) if (q_binary.sum() + c_binary.sum()) > 0 else 1.0
    
    # Pixel difference map
    diff_map = np.abs(q_binary.astype(np.int16) - c_binary.astype(np.int16)) * 255
    diff_map = diff_map.astype(np.uint8)
    
    return {
        "classical_gx": classical_gx,
        "classical_gy": classical_gy,
        "classical_magnitude": classical_magnitude,
        "classical_edges": classical_edges,
        "classical_boundary": classical_boundary,
        "classical_contours": classical_contours,
        "classical_time": classical_time,
        "quantum_time": quantum_time,
        "quantum_ops": quantum_ops,
        "quantum_contour_count": len(quantum_contours),
        "classical_contour_count": len(classical_contours),
        "quantum_edge_pixels": int(q_binary.sum()),
        "classical_edge_pixels": int(c_binary.sum()),
        "iou": round(iou, 4),
        "dice_similarity": round(dice, 4),
        "diff_map": diff_map
    }
