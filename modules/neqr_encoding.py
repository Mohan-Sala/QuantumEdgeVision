import time
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

def create_neqr_circuit(image: np.ndarray) -> QuantumCircuit:
    """
    Constructs the Novel Enhanced Quantum Representation (NEQR) circuit for an 8x8 image.
    Faithful implementation of Cell 4 in quantumEdgeVision.ipynb.
    
    Quantum Register Architecture:
    - X Register: 3 qubits (|x2, x1, x0⟩ representing columns 0 to 7)
    - Y Register: 3 qubits (|y2, y1, y0⟩ representing rows 0 to 7)
    - Intensity Register: 8 qubits (|c7, ..., c0⟩ representing grayscale values 0 to 255)
    Total: 14 qubits.
    """
    height, width = image.shape
    if height != 8 or width != 8:
        raise ValueError(f"NEQR encoding requires an 8x8 image, got {height}x{width}")
        
    # Quantum registers
    x_reg = QuantumRegister(3, "X")
    y_reg = QuantumRegister(3, "Y")
    intensity_reg = QuantumRegister(8, "Intensity")
    qc = QuantumCircuit(x_reg, y_reg, intensity_reg, name="NEQR_Circuit")
    
    # Create equal superposition across all 64 spatial pixel positions
    qc.h(x_reg)
    qc.h(y_reg)
    
    controls = list(x_reg) + list(y_reg)
    
    # Encode each pixel intensity using multi-controlled X gates
    for y in range(8):
        for x in range(8):
            pixel_value = int(image[y, x])
            if pixel_value == 0:
                continue  # All intensity qubits are initialized to |0⟩, so 0 requires no gate
                
            # Qubit order is least-significant bit first
            for bit in range(8):
                if (pixel_value >> bit) & 1:
                    address_bits = (
                        [(x >> i) & 1 for i in range(3)]
                        + [(y >> i) & 1 for i in range(3)]
                    )
                    # Convert zero-controls into positive controls
                    for control, value in zip(controls, address_bits):
                        if value == 0:
                            qc.x(control)
                            
                    # Multi-controlled NOT to set the specific intensity bit
                    qc.mcx(controls, intensity_reg[bit])
                    
                    # Restore address qubits to preserve superposition state
                    for control, value in zip(controls, address_bits):
                        if value == 0:
                            qc.x(control)
                            
    return qc

def simulate_and_verify_neqr(circuit: QuantumCircuit, image: np.ndarray) -> dict:
    """
    Simulates the NEQR circuit using exact Qiskit Statevector simulation and verifies
    that every basis state faithfully encodes the corresponding pixel coordinate and intensity.
    Faithful implementation of Cell 13 in quantumEdgeVision.ipynb.
    """
    start_time = time.time()
    
    # Exact statevector simulation
    state = Statevector.from_instruction(circuit)
    state_data = state.data
    sim_time = time.time() - start_time
    
    # Identify nonzero computational basis states
    nonzero_indices = np.where(np.abs(state_data) > 1e-8)[0]
    
    # Verify each basis state against the image
    encoding_correct = True
    mismatches = []
    verified_pixels = []
    
    for index in nonzero_indices:
        # Extract quantum registers from computational basis index:
        # Bits 0..2: X coordinate (0-7)
        # Bits 3..5: Y coordinate (0-7)
        # Bits 6..13: Intensity (0-255)
        x = int(index & 7)
        y = int((index >> 3) & 7)
        intensity = int((index >> 6) & 255)
        
        expected = int(image[y, x])
        verified_pixels.append({
            "x": x,
            "y": y,
            "encoded_intensity": intensity,
            "expected_intensity": expected,
            "amplitude": complex(state_data[index]),
            "probability": float(np.abs(state_data[index])**2)
        })
        
        if intensity != expected:
            encoding_correct = False
            mismatches.append((x, y, intensity, expected))
            
    normalization = float(np.sum(np.abs(state_data)**2))
    
    stats = {
        "num_qubits": circuit.num_qubits,
        "total_gates": circuit.size(),
        "depth": circuit.depth(),
        "gate_counts": circuit.count_ops(),
        "num_nonzero_states": len(nonzero_indices),
        "expected_nonzero_states": 64,
        "normalization": round(normalization, 6),
        "encoding_correct": encoding_correct,
        "sim_time": sim_time,
        "mismatches": mismatches,
        "sample_verified": verified_pixels[:10]  # First 10 samples for inspection
    }
    return stats
