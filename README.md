# QuantumEdgeVision: A Reversible Quantum Arithmetic Framework for Grayscale Image Edge Detection and Boundary Extraction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Qiskit 2.x](https://img.shields.io/badge/Qiskit-2.6.2-purple.svg)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

**QuantumEdgeVision** is an interactive, academic quantum computing simulation application. It translates classical 8×8 grayscale images into the quantum domain via the **Novel Enhanced Quantum Representation (NEQR)** and executes spatial gradient extraction using **18-qubit reversible quantum ripple-carry arithmetic (`CDKMRippleCarryAdder`)**. Extracted quantum gradients are subsequently binarized and mapped to closed object boundaries using topological contour extraction.

---

## 🌟 Key Features

1. **NEQR Quantum Image Representation (14 Qubits)**
   - Encodes spatial coordinates ($3$ qubits for column $X$, $3$ qubits for row $Y$) into an equal superposition using Hadamard transforms.
   - Entangles $8$ intensity qubits using multi-controlled Pauli-X gates ($MCX$) to represent grayscale values ($0$ to $255$).
   - Rigorously verified via exact statevector simulation ($64$ non-zero computational basis states, $\sum |c_i|^2 = 1.0$).

2. **Reversible Quantum Arithmetic (18 Qubits)**
   - Computes differences reversibly using two's complement arithmetic:
     $$A - B = A + (\sim B) + 1$$
   - Integrates Qiskit's `CDKMRippleCarryAdder(kind="full")` with carry-in set to $|1\rangle$.
   - Preserves Register $A$ throughout unitary computation, satisfying information reversibility.

3. **Quantum-Assisted Edge Detection**
   - Evaluates directional gradients across interior pixels:
     $$G_x(y, x) = |I(y, x-1) \ominus I(y, x+1)|$$
     $$G_y(y, x) = |I(y-1, x) \ominus I(y+1, x)|$$
   - Evaluates $72$ genuine reversible quantum subtraction circuits per 8×8 image decimation.
   - Computes gradient magnitude $G_x + G_y$ and applies dynamic thresholding.

4. **Topological Boundary Extraction**
   - Employs OpenCV contour topology (`cv2.findContours`, `cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE`).
   - Superimposes extracted green boundaries over original grayscale images.

5. **Rigorous Classical vs Quantum Benchmarking**
   - Compares simulated quantum arithmetic against classical $3\times3$ Sobel convolution.
   - Quantitative evaluation of execution times, contour counts, active edge pixels, Jaccard Index (IoU), and Dice similarity.
   - Clarifies classical CPU statevector simulation costs versus theoretical fault-tolerant quantum hardware scaling.

6. **Thermodynamic Landauer Analysis**
   - Evaluates theoretical minimum energy dissipation ($k_B T \ln 2$ per bit erasure) versus the zero-entropy limit of reversible unitary circuits.

---

## 🏗️ Project Architecture

```
QuantumEdgeVision/
│
├── app.py                      # Main Streamlit single-page interactive application
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation and setup guide
├── run.bat                     # Windows one-click batch launcher
├── quantumEdgeVision.ipynb     # Original reference Jupyter Notebook
├── test_pipeline.py            # Automated end-to-end unit test suite
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py        # Grayscale conversion & INTER_AREA 8×8 decimation
│   ├── neqr_encoding.py        # 14-qubit NEQR circuit synthesis & statevector verification
│   ├── reversible_arithmetic.py# 18-qubit CDKMRippleCarryAdder subtraction playground & cache
│   ├── edge_detection.py       # Directional gradient (Gx, Gy) & thresholding
│   ├── boundary_extraction.py  # Contour extraction & geometric metrics
│   ├── classical_comparison.py # Classical Sobel filter & quantitative similarity metrics
│   └── circuit_analysis.py     # Landauer thermodynamics calculations
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # Interactive Plotly heatmaps, charts, and upscaling
│   └── helpers.py              # Circuit export and byte conversion
│
└── assets/
    └── styles.css              # Dark quantum laboratory single-page theme
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10 or 3.11 installed on your system.
- Standard terminal (PowerShell, Command Prompt, or Bash).

### 1. Open the Workspace
```powershell
cd "d:\Quantum Edge Vision"
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite (Optional)
```powershell
python test_pipeline.py
```

### 4. Launch the Web Application
Run using the Python module syntax (recommended for all Windows setups):
```powershell
python -m streamlit run app.py
```
Or simply double-click **`run.bat`** in the project folder!

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🔬 Single-Page Interactive Simulation Flow

The application features a single-page interface with no sidebar navigation:

1. **User Image Upload:** Upload your image (PNG, JPG, or JPEG) and click **"Run Quantum Simulation & Generate Pipeline"**.
2. **Sequential Top-to-Bottom Generation:** The entire pipeline executes and renders cleanly down the page across 7 streamlined steps:
   - **Step 1: Image Preprocessing & 8×8 Decimation:** Converts to grayscale, decimates to an 8×8 pixel matrix with `INTER_AREA`, and displays an interactive Plotly heatmap with computational basis state annotations.
   - **Step 2: NEQR Quantum Image Encoding (14 Qubits):** Synthesizes the 14-qubit circuit, placing spatial registers into equal superposition via Hadamard gates and entangling 8 intensity qubits via multi-controlled X gates.
   - **Step 3: Reversible Quantum Arithmetic (18 Qubits):** Demonstrates two's complement subtraction ($A - B = A + (\sim B) + 1$) via `CDKMRippleCarryAdder`, preserving Register A and highlighting Landauer's theoretical zero-dissipation limit.
   - **Step 4: Quantum-Assisted Edge Detection:** Evaluates 72 reversible subtraction operations across 36 interior pixels to obtain horizontal ($G_x$) and vertical ($G_y$) gradients, with a live interactive threshold slider.
   - **Step 5: Boundary Extraction:** Traces closed outer contours using OpenCV and renders green outlines directly over the grayscale image.
   - **Step 6: Classical Sobel vs Quantum Comparison:** Benchmarks quantum reversible arithmetic against classical 3×3 Sobel convolution with IoU and Dice similarity metrics.
   - **Step 7: Circuit Complexity, Landauer Thermodynamics & Downloads:** Displays gate breakdowns, Landauer energy calculations, and direct download buttons for all images and reports.

---

## 📐 Quantum Mathematical Background

### NEQR State Formulation
For an $8\times 8$ grayscale image ($2^3 \times 2^3$), the image state $|\Psi\rangle$ is represented across 14 qubits:
$$|\Psi\rangle = \frac{1}{\sqrt{64}} \sum_{y=0}^{7} \sum_{x=0}^{7} |y\rangle_3 |x\rangle_3 |I(y, x)\rangle_8$$
where $|y\rangle_3 = |y_2 y_1 y_0\rangle$, $|x\rangle_3 = |x_2 x_1 x_0\rangle$, and $|I(y, x)\rangle_8 = |c_7 c_6 \dots c_0\rangle$.

### Reversible Subtraction via Two's Complement
Unitary quantum gates do not permit destructive erasure. To evaluate $A - B$ reversibly:
$$A - B = A + (\sim B) + 1$$
1. Load $A$ into Register $A$ ($q[1:9]$).
2. Load $\sim B$ (One's Complement) into Register $B$ ($q[9:17]$).
3. Initialize Carry-In $q[0] = |1\rangle$ (completing the $+1$ addition).
4. Apply the `CDKMRippleCarryAdder` full adder.
5. The output difference $|A - B|$ is obtained in Register $B$, while Register $A$ remains preserved.

---

## ⚖️ Academic Attribution & Disclaimer
This simulation executes **classical statevector simulations** of real Qiskit quantum circuits to demonstrate algorithmic correctness and entropy preservation. It is intended for academic research, education, and algorithmic prototyping in Quantum Image Processing (QIP).
