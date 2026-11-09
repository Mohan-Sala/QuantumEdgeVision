import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def create_pixel_matrix_heatmap(matrix: np.ndarray, title: str = "8×8 Pixel Intensity Matrix") -> go.Figure:
    """
    Creates an interactive Plotly heatmap for the 8x8 image pixel intensity matrix.
    Includes custom hover information showing coordinates, decimal value, and 8-bit binary representation.
    """
    height, width = matrix.shape
    x_coords = [f"X={x}" for x in range(width)]
    y_coords = [f"Y={y}" for y in range(height)]
    
    hover_text = []
    for y in range(height):
        row_text = []
        for x in range(width):
            val = int(matrix[y, x])
            bin_val = format(val, "08b")
            row_text.append(
                f"<b>Pixel Coordinates:</b> (X={x}, Y={y})<br>"
                f"<b>Intensity (Decimal):</b> {val}<br>"
                f"<b>Intensity (Hex):</b> 0x{val:02X}<br>"
                f"<b>Intensity (Binary):</b> |{bin_val}⟩<br>"
                f"<b>Normalized:</b> {val / 255.0:.3f}"
            )
        hover_text.append(row_text)
    
    colorscale = [
        [0.0, "#0a0e17"],
        [0.2, "#064e3b"],
        [0.4, "#047857"],
        [0.6, "#0d9488"],
        [0.8, "#06b6d4"],
        [1.0, "#34d399"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=x_coords,
        y=y_coords,
        text=hover_text,
        hoverinfo="text",
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="Intensity", font=dict(color="#94a3b8", size=12)),
            tickfont=dict(color="#cbd5e1"),
            thickness=14,
            len=0.9
        ),
        showscale=True
    ))
    
    annotations = []
    for y in range(height):
        for x in range(width):
            val = int(matrix[y, x])
            font_color = "#ffffff" if val < 160 else "#0f172a"
            annotations.append(dict(
                x=f"X={x}",
                y=f"Y={y}",
                text=str(val),
                showarrow=False,
                font=dict(color=font_color, size=11, family="Courier New, monospace")
            ))
            
    fig.update_layout(
        title=dict(text=title, font=dict(color="#38bdf8", size=16, family="Segoe UI, sans-serif")),
        annotations=annotations,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=420
    )
    fig.update_xaxes(tickfont=dict(color="#94a3b8"), side="bottom")
    fig.update_yaxes(tickfont=dict(color="#94a3b8"), autorange="reversed")
    return fig

def upscale_for_display(image: np.ndarray, target_size: tuple = (400, 400)) -> np.ndarray:
    """
    Upscales a low-resolution image (such as 8x8) using INTER_NEAREST interpolation
    to preserve crisp pixel boundaries for clear visual analysis.
    """
    if image is None:
        return None
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

def create_gate_distribution_chart(gate_counts: dict, title: str = "Quantum Gate Distribution") -> go.Figure:
    """
    Creates a modern horizontal bar chart of quantum gates used in the circuit.
    Includes robust fallback to ensure zero runtime crashes across all Plotly versions.
    """
    gates = list(gate_counts.keys())
    counts = list(gate_counts.values())
    
    try:
        colors = px.colors.qualitative.Prism[:len(gates)] if len(gates) > 0 else ["#38bdf8"]
        fig = go.Figure(data=go.Bar(
            x=counts,
            y=gates,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="#38bdf8", width=1.5)
            ),
            text=counts,
            textposition="auto",
            textfont=dict(color="#ffffff", size=12)
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color="#38bdf8", size=15)),
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            margin=dict(l=60, r=30, t=50, b=40),
            height=320
        )
        fig.update_xaxes(title_text="Gate Count", tickfont=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Gate Type", tickfont=dict(color="#cbd5e1"))
        return fig
    except Exception:
        fig = go.Figure(go.Bar(x=counts, y=gates, orientation="h"))
        fig.update_layout(title=title, template="plotly_dark", height=320)
        return fig
