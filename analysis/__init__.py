"""CV and audio analysis modules."""

from .blink_detector import (
    BlinkAnalysis,
    BlinkEvent,
    detect_blinks,
    format_blink_analysis,
    generate_trigger_response_map,
    annotate_transcript_with_blinks,
    get_blink_metrics_for_prompt
)
from .spectrogram_analyzer import (
    generate_spectrogram,
    SpectrogramResult,
    get_spectrogram_for_prompt
)

__all__ = [
    'BlinkAnalysis',
    'BlinkEvent',
    'detect_blinks',
    'format_blink_analysis',
    'generate_trigger_response_map',
    'annotate_transcript_with_blinks',
    'get_blink_metrics_for_prompt',
    'generate_spectrogram',
    'SpectrogramResult',
    'get_spectrogram_for_prompt',
]
