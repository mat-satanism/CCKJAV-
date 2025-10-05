# planet_3d.py
import streamlit.components.v1 as components
from pathlib import Path

def show_planet_3d():
    """Відображає HTML з 3D-моделлю екзопланети."""
    html_path = Path(__file__).parent / "exoplanet_3d.html"
    if not html_path.exists():
        components.html(f"<div style='color:red'>❌ Файл {html_path} не знайдено.</div>", height=600)
        return

    components.html(
        html_path.read_text(encoding="utf-8"),
        height=900,  # можна змінювати
        scrolling=True
    )
