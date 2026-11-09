import io
import numpy as np
import cv2
from PIL import Image

def get_image_download_bytes(image: np.ndarray, format: str = "PNG", upscale: bool = True) -> bytes:
    """
    Converts a numpy image to downloadable bytes.
    Upscales small images (like 8x8) using INTER_NEAREST to 256x256 for clear viewing.
    """
    if image is None:
        return b""
    
    img_to_save = image.copy()
    if upscale and (img_to_save.shape[0] <= 16 or img_to_save.shape[1] <= 16):
        img_to_save = cv2.resize(img_to_save, (256, 256), interpolation=cv2.INTER_NEAREST)
        
    if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
        pil_img = Image.fromarray(cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(img_to_save.astype(np.uint8))
        
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()

def get_circuit_image_bytes(circuit, fold: int = 80) -> bytes:
    """Renders a Qiskit circuit to PNG bytes using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig = circuit.draw(output="mpl", fold=fold, idle_wires=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="#0e1726")
    plt.close(fig)
    return buf.getvalue()
