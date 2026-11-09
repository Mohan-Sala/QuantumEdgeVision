import io
import unittest
import numpy as np
import cv2

from modules.preprocessing import load_image_from_upload, preprocess_image
from modules.neqr_encoding import create_neqr_circuit, simulate_and_verify_neqr
from modules.reversible_arithmetic import (
    create_reversible_subtractor,
    simulate_reversible_subtraction,
    get_subtractor_detailed_analysis,
    cached_reversible_subtraction
)
from modules.edge_detection import quantum_arithmetic_edge_detection, apply_edge_threshold
from modules.boundary_extraction import extract_boundaries, get_contour_metrics
from modules.classical_comparison import classical_sobel_edge_detection, compare_quantum_and_classical
from modules.circuit_analysis import compute_landauer_limit
from utils.visualization import (
    create_pixel_matrix_heatmap,
    upscale_for_display,
    create_gate_distribution_chart
)
from utils.helpers import get_image_download_bytes, get_circuit_image_bytes

class TestQuantumEdgeVisionPipeline(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 16x16 BGR test image containing a high-contrast square
        self.synthetic_bgr = np.zeros((16, 16, 3), dtype=np.uint8)
        self.synthetic_bgr[4:12, 4:12] = [220, 220, 220]

    def test_preprocessing(self):
        gray, small, matrix = preprocess_image(self.synthetic_bgr, target_size=(8, 8))
        self.assertEqual(gray.shape, (16, 16))
        self.assertEqual(small.shape, (8, 8))
        self.assertEqual(matrix.shape, (8, 8))
        self.assertEqual(matrix.dtype, np.uint8)

    def test_load_image_from_upload(self):
        success, encoded = cv2.imencode(".png", self.synthetic_bgr)
        self.assertTrue(success)
        buf = io.BytesIO(encoded.tobytes())
        loaded = load_image_from_upload(buf)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, self.synthetic_bgr.shape)

    def test_reversible_subtraction(self):
        cases = [(180, 75), (255, 0), (100, 100), (50, 120)]
        for a, b in cases:
            res, qc = simulate_reversible_subtraction(a, b, n=8)
            self.assertEqual(res, abs(a - b))
            self.assertEqual(qc.num_qubits, 18)

        # Test direct circuit construction
        sub_qc = create_reversible_subtractor(180, 75, n=8)
        self.assertEqual(sub_qc.num_qubits, 18)

        cached_res = cached_reversible_subtraction(180, 75, n=8)
        self.assertEqual(cached_res, 105)

        analysis = get_subtractor_detailed_analysis(200, 80, n=8)
        self.assertTrue(analysis["is_correct"])
        self.assertTrue(analysis["a_preserved"])
        self.assertEqual(analysis["simulated_diff"], 120)

    def test_neqr_encoding_and_verification(self):
        test_img = np.zeros((8, 8), dtype=np.uint8)
        test_img[2:6, 2:6] = 200
        
        qc = create_neqr_circuit(test_img)
        self.assertEqual(qc.num_qubits, 14)
        
        stats = simulate_and_verify_neqr(qc, test_img)
        self.assertTrue(stats["encoding_correct"], "NEQR statevector verification failed")
        self.assertEqual(stats["num_nonzero_states"], 64)
        self.assertAlmostEqual(stats["normalization"], 1.0, places=4)

    def test_edge_detection_and_thresholding(self):
        test_img = np.zeros((8, 8), dtype=np.uint8)
        test_img[2:6, 2:6] = 220
        
        gx, gy, mag, ops, el_time = quantum_arithmetic_edge_detection(test_img, use_caching=True)
        self.assertEqual(ops, 72)
        self.assertEqual(gx.shape, (8, 8))
        self.assertEqual(gy.shape, (8, 8))
        self.assertEqual(mag.shape, (8, 8))
        
        edges = apply_edge_threshold(mag, 60)
        self.assertEqual(edges.shape, (8, 8))
        self.assertTrue(np.any(edges > 0))

    def test_boundary_extraction(self):
        test_img = np.zeros((8, 8), dtype=np.uint8)
        test_img[2:6, 2:6] = 220
        _, _, mag, _, _ = quantum_arithmetic_edge_detection(test_img, use_caching=True)
        edges = apply_edge_threshold(mag, 60)
        
        boundary_img, contours = extract_boundaries(edges, test_img)
        self.assertEqual(boundary_img.shape, (8, 8, 3))
        self.assertGreater(len(contours), 0)

        metrics = get_contour_metrics(contours)
        self.assertEqual(len(metrics), len(contours))
        self.assertIn("contour_id", metrics[0])

    def test_classical_comparison(self):
        test_img = np.zeros((8, 8), dtype=np.uint8)
        test_img[2:6, 2:6] = 220
        
        # Test standalone classical sobel
        cgx, cgy, cmag, ctime = classical_sobel_edge_detection(test_img)
        self.assertEqual(cmag.shape, (8, 8))
        self.assertGreater(ctime, 0.0)

        gx, gy, mag, ops, el_time = quantum_arithmetic_edge_detection(test_img, use_caching=True)
        edges = apply_edge_threshold(mag, 60)
        _, contours = extract_boundaries(edges, test_img)
        
        comp = compare_quantum_and_classical(test_img, edges, contours, el_time, ops, threshold=60)
        self.assertIn("iou", comp)
        self.assertIn("dice_similarity", comp)
        self.assertIn("classical_time", comp)
        self.assertGreaterEqual(comp["iou"], 0.0)

    def test_thermodynamics(self):
        landauer = compute_landauer_limit(72, temp_kelvin=300.0)
        self.assertGreater(landauer["classical_thermodynamic_limit_joules"], 0.0)
        self.assertEqual(landauer["quantum_theoretical_limit_joules"], 0.0)
        self.assertEqual(landauer["total_bits_erased"], 72 * 8)

    def test_visualization_and_helpers(self):
        mat = np.arange(64, dtype=np.uint8).reshape((8, 8))
        fig_heat = create_pixel_matrix_heatmap(mat)
        self.assertIsNotNone(fig_heat)
        
        fig_gates = create_gate_distribution_chart({"cx": 10, "x": 5, "h": 6})
        self.assertIsNotNone(fig_gates)
        
        up = upscale_for_display(mat, (128, 128))
        self.assertEqual(up.shape, (128, 128))
        
        raw_bytes = get_image_download_bytes(mat)
        self.assertGreater(len(raw_bytes), 0)

        qc = create_reversible_subtractor(10, 5, n=4)
        circ_bytes = get_circuit_image_bytes(qc)
        self.assertGreater(len(circ_bytes), 0)

if __name__ == "__main__":
    unittest.main()
