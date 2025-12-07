"""Output generation: PDFs, visualizations."""

from .pdf_generator import generate_pdf_report, generate_summary_pdf
from .visualizations import (
    create_confidence_gauge,
    create_component_confidence_bars,
    create_big_five_radar,
    create_dark_triad_bars,
    create_threat_matrix,
    create_mbti_chart,
    create_all_visualizations,
)

# Check if reportlab is available
try:
    import reportlab
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

__all__ = [
    'generate_pdf_report',
    'generate_summary_pdf',
    'REPORTLAB_AVAILABLE',
    'create_confidence_gauge',
    'create_component_confidence_bars',
    'create_big_five_radar',
    'create_dark_triad_bars',
    'create_threat_matrix',
    'create_mbti_chart',
    'create_all_visualizations',
]
