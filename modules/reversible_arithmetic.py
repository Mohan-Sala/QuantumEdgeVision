import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.quantum_info import Statevector

_SUBTRACTION_CACHE = {}

def create_reversible_subtractor(a: int, b: int, n: int = 8) -> QuantumCircuit:
    """
    Constructs an n-bit reversible quantum subtractor circuit using CDKMRippleCarryAdder.
    Computes A - B reversibly via Two's Complement: A + (~B) + 1.
    """
    mask = (1 << n) - 1
    a = int(a)
    b = int(b)
    if not (0 <= a <= mask and 0 <= b <= mask):
        raise ValueError(f"Inputs A={a}, B={b} must fit in the selected {n}-bit width (0 to {mask})")
        
    adder = CDKMRippleCarryAdder(n, kind="full")
    qc = QuantumCircuit(adder.num_qubits, name=f"RevSub_{a}_{b}")
    
    # Initialize Register A
    for i in range(n):
        if (a >> i) & 1:
            qc.x(1 + i)
            
    # Initialize one's complement of B: (~b) & mask
    b_complement = (~b) & mask
    for i in range(n):
        if (b_complement >> i) & 1:
            qc.x(1 + n + i)
            
    # Set carry-in to 1 (+1 for two's complement)
    qc.x(0)
    
    # Reversible addition: B = A + (~B) + 1 = A - B
    qc.compose(adder, inplace=True)
    return qc

def simulate_reversible_subtraction(a: int, b: int, n: int = 8, verify: bool = True) -> tuple:
    """
    Simulates the reversible quantum subtraction circuit via exact Statevector simulation.
    """
    mask = (1 << n) - 1
    a = int(a)
    b = int(b)
    larger = max(a, b)
    smaller = min(a, b)
    
    qc = create_reversible_subtractor(larger, smaller, n)
    state = Statevector.from_instruction(qc)
    
    probabilities = np.abs(state.data) ** 2
    basis_index = int(np.argmax(probabilities))
    
    result = (basis_index >> (1 + n)) & mask
    preserved_a = (basis_index >> 1) & mask
    
    if verify:
        if preserved_a != larger:
            raise RuntimeError(f"Reversible preservation error: A ({preserved_a}) != {larger}")
        if result != (larger - smaller):
            raise RuntimeError(f"Quantum subtraction error: Result ({result}) != {larger - smaller}")
            
    return result, qc

def cached_reversible_subtraction(a: int, b: int, n: int = 8) -> int:
    """
    Cached wrapper around simulate_reversible_subtraction for fast repeated edge sweeps.
    """
    key = (min(int(a), int(b)), max(int(a), int(b)), n)
    if key in _SUBTRACTION_CACHE:
        return _SUBTRACTION_CACHE[key]
        
    result, _ = simulate_reversible_subtraction(a, b, n=n, verify=True)
    _SUBTRACTION_CACHE[key] = result
    return result

def get_subtractor_detailed_analysis(a: int, b: int, n: int = 8) -> dict:
    """
    Executes quantum subtraction and returns register states, gate counts, and verification.
    """
    start_time = time.time()
    larger = max(int(a), int(b))
    smaller = min(int(a), int(b))
    expected = larger - smaller
    mask = (1 << n) - 1
    
    result, qc = simulate_reversible_subtraction(larger, smaller, n=n, verify=True)
    elapsed = time.time() - start_time
    
    bin_a = format(larger, f"0{n}b")
    bin_b = format(smaller, f"0{n}b")
    b_comp = (~smaller) & mask
    bin_b_comp = format(b_comp, f"0{n}b")
    bin_result = format(result, f"0{n}b")
    
    layout = [
        {"qubits": "q[0]", "role": "Carry-in", "initial": "1 (|1⟩)", "final": "Carry-out state"},
        {"qubits": f"q[1:{1+n}]", "role": "Register A (Larger Operand)", "initial": f"|{bin_a}⟩ ({larger})", "final": f"|{bin_a}⟩ (Preserved)"},
        {"qubits": f"q[{1+n}:{1+2*n}]", "role": "Register B (Subtrahend / Output)", "initial": f"|{bin_b_comp}⟩ (One's Comp)", "final": f"|{bin_result}⟩ (Difference = {result})"},
        {"qubits": f"q[{1+2*n}]", "role": "Carry-out / Overflow", "initial": "0 (|0⟩)", "final": "Generated carry bit"}
    ]
    
    return {
        "input_a": int(a),
        "input_b": int(b),
        "larger": larger,
        "smaller": smaller,
        "expected_diff": expected,
        "simulated_diff": result,
        "is_correct": (result == expected),
        "a_preserved": True,
        "bin_a": bin_a,
        "bin_b": bin_b,
        "bin_b_comp": bin_b_comp,
        "bin_result": bin_result,
        "num_qubits": qc.num_qubits,
        "total_gates": qc.size(),
        "depth": qc.depth(),
        "gate_counts": qc.count_ops(),
        "circuit": qc,
        "execution_time": elapsed,
        "layout": layout
    }
