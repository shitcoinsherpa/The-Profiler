"""Core analysis logic for behavioral profiling."""

from .modular_executor import ModularAnalysisExecutor, StageResult, SubAnalysisResult, format_modular_results
from .confidence_scoring import (
    ConfidenceScore,
    AnalysisConfidence,
    calculate_analysis_confidence,
    format_confidence_for_display,
    add_confidence_to_result
)
from .signal_collapsing import collapse_analysis_outputs

__all__ = [
    'ModularAnalysisExecutor',
    'StageResult',
    'SubAnalysisResult',
    'format_modular_results',
    'ConfidenceScore',
    'AnalysisConfidence',
    'calculate_analysis_confidence',
    'format_confidence_for_display',
    'add_confidence_to_result',
    'collapse_analysis_outputs',
]
