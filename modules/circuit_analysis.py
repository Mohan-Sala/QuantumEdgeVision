import numpy as np

def compute_landauer_limit(operations_count: int, temp_kelvin: float = 300.0) -> dict:
    """
    Calculates the Landauer limit of energy dissipation for irreversible classical vs reversible computation.
    
    Landauer's Principle:
    Any logically irreversible operation that erases information dissipates at least:
        E_min = k_B * T * ln(2) Joules
    where k_B is Boltzmann's constant (1.380649e-23 J/K).
    
    In contrast, reversible quantum circuits preserve the full unitary state space,
    enabling theoretical zero-heat information processing.
    """
    kb = 1.380649e-23  # J/K
    energy_per_erasure = kb * temp_kelvin * np.log(2)
    # 8 bits erased per classical 8-bit subtraction
    bits_erased = operations_count * 8
    classical_min_energy = bits_erased * energy_per_erasure
    
    return {
        "temp_kelvin": temp_kelvin,
        "energy_per_bit_joules": energy_per_erasure,
        "total_bits_erased": bits_erased,
        "classical_thermodynamic_limit_joules": classical_min_energy,
        "quantum_theoretical_limit_joules": 0.0
    }
