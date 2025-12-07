"""
Spectrogram Generation and Voice Stress Visualization.

Generates visual spectrograms from audio for multimodal analysis.
This allows the LLM to "see" voice stress patterns rather than inferring them.
"""

import logging
import base64
import io
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for required libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

SPECTROGRAM_AVAILABLE = NUMPY_AVAILABLE and LIBROSA_AVAILABLE and MATPLOTLIB_AVAILABLE


@dataclass
class SpectrogramResult:
    """Result of spectrogram generation."""
    available: bool
    image_base64: Optional[str] = None
    duration_seconds: float = 0.0
    sample_rate: int = 0
    error: Optional[str] = None


def generate_spectrogram(
    audio_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    n_mels: int = 128,
    hop_length: int = 512,
    highlight_stress_regions: bool = True
) -> SpectrogramResult:
    """
    Generate a mel spectrogram from audio file.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        output_path: Optional path to save the spectrogram image
        figsize: Figure size in inches (width, height)
        n_mels: Number of mel bands
        hop_length: Hop length for STFT
        highlight_stress_regions: If True, annotate potential stress regions

    Returns:
        SpectrogramResult with base64-encoded PNG image
    """
    if not SPECTROGRAM_AVAILABLE:
        missing = []
        if not NUMPY_AVAILABLE:
            missing.append("numpy")
        if not LIBROSA_AVAILABLE:
            missing.append("librosa")
        if not MATPLOTLIB_AVAILABLE:
            missing.append("matplotlib")
        return SpectrogramResult(
            available=False,
            error=f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}"
        )

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

        # Main spectrogram
        ax1 = axes[0]
        img = librosa.display.specshow(
            S_dB,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            hop_length=hop_length,
            ax=ax1,
            cmap='magma'
        )
        fig.colorbar(img, ax=ax1, format='%+2.0f dB', label='Power (dB)')
        ax1.set_title('Mel Spectrogram - Voice Stress Analysis', fontsize=12, fontweight='bold')
        ax1.set_xlabel('')

        # RMS energy plot (shows loudness/stress over time)
        ax2 = axes[1]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        ax2.plot(times, rms, color='#1f77b4', linewidth=1)
        ax2.fill_between(times, rms, alpha=0.3)
        ax2.set_ylabel('Energy (RMS)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_xlim([0, duration])

        # Highlight stress regions (high energy spikes)
        if highlight_stress_regions:
            mean_rms = np.mean(rms)
            std_rms = np.std(rms)
            threshold = mean_rms + 1.5 * std_rms

            stress_mask = rms > threshold
            stress_regions = []
            in_region = False
            region_start = 0

            for i, is_stress in enumerate(stress_mask):
                if is_stress and not in_region:
                    in_region = True
                    region_start = times[i]
                elif not is_stress and in_region:
                    in_region = False
                    stress_regions.append((region_start, times[i]))

            if in_region:
                stress_regions.append((region_start, times[-1]))

            # Highlight stress regions on spectrogram
            for start, end in stress_regions:
                ax1.axvspan(start, end, alpha=0.2, color='red')
                ax2.axvspan(start, end, alpha=0.2, color='red')

            # Add legend
            if stress_regions:
                ax1.text(
                    0.02, 0.98, f'Red: Elevated stress regions ({len(stress_regions)} detected)',
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Optionally save to file
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Spectrogram saved to {output_path}")

        plt.close(fig)

        return SpectrogramResult(
            available=True,
            image_base64=image_base64,
            duration_seconds=duration,
            sample_rate=sr
        )

    except Exception as e:
        logger.error(f"Spectrogram generation failed: {e}")
        return SpectrogramResult(
            available=False,
            error=str(e)
        )


def generate_spectrogram_from_base64(
    audio_base64: str,
    audio_format: str = 'wav',
    **kwargs
) -> SpectrogramResult:
    """
    Generate spectrogram from base64-encoded audio.

    Args:
        audio_base64: Base64-encoded audio data
        audio_format: Audio format (wav, mp3, etc.)
        **kwargs: Additional arguments passed to generate_spectrogram

    Returns:
        SpectrogramResult with base64-encoded PNG image
    """
    import tempfile
    import os

    if not SPECTROGRAM_AVAILABLE:
        return SpectrogramResult(
            available=False,
            error="Spectrogram dependencies not available"
        )

    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as f:
            f.write(base64.b64decode(audio_base64))
            temp_path = f.name

        # Generate spectrogram
        result = generate_spectrogram(temp_path, **kwargs)

        # Clean up
        os.unlink(temp_path)

        return result

    except Exception as e:
        logger.error(f"Spectrogram generation from base64 failed: {e}")
        return SpectrogramResult(
            available=False,
            error=str(e)
        )


# Prompt for multimodal spectrogram analysis
SPECTROGRAM_ANALYSIS_PROMPT = """Analyze this audio spectrogram for voice stress indicators.

You are viewing a mel spectrogram visualization of the subject's speech.
The top panel shows frequency content over time (brighter = louder).
The bottom panel shows RMS energy (loudness) over time.
Red highlighted regions indicate detected stress spikes.

VISUAL STRESS INDICATORS TO LOOK FOR:

1. PITCH CHANGES (Y-axis patterns):
   - Look for sudden vertical shifts in bright bands = pitch changes
   - Rising pitch patterns = possible stress or deception
   - Stable horizontal bands = calm, confident speech

2. ENERGY SPIKES (Bottom panel + brightness):
   - Sudden bright bursts = emphatic speech or stress
   - Correlate with red highlighted regions
   - Note the topics/timing of these spikes

3. MICRO-TREMORS:
   - Look for "fuzzy" or "wobbly" patterns in the frequency bands
   - Wavy horizontal lines instead of clean ones = voice tremor
   - Indicates autonomic stress response

4. PAUSES AND HESITATIONS:
   - Dark vertical bands = silence
   - Long silences before responses = cognitive load
   - Short gaps within sentences = hesitation markers

5. FORMANT PATTERNS:
   - The horizontal bright bands are formants (voice resonance)
   - Consistent formants = stable emotional state
   - Shifting formants = emotional arousal

Provide:
- STRESS REGIONS IDENTIFIED: [timestamp ranges]
- PITCH PATTERN ASSESSMENT: Stable/Variable/Erratic
- TREMOR INDICATORS: Present/Absent/Uncertain
- ENERGY SPIKE CORRELATION: [what topics corresponded to spikes]
- OVERALL VOCAL STRESS LEVEL: Low/Moderate/High/Extreme
- CONFIDENCE IN ASSESSMENT: Low/Medium/High

NOTE: This visual analysis grounds vocal stress assessment in actual acoustic data
rather than LLM inference from audio alone."""


def get_spectrogram_for_prompt(audio_path: str) -> Dict:
    """
    Generate spectrogram and format for LLM prompt injection.

    Returns dict with:
    - available: Whether generation succeeded
    - image_base64: Base64 PNG for multimodal model
    - prompt: Analysis prompt to use with image
    - duration: Audio duration
    """
    result = generate_spectrogram(audio_path)

    if result.available:
        return {
            'available': True,
            'image_base64': result.image_base64,
            'prompt': SPECTROGRAM_ANALYSIS_PROMPT,
            'duration_seconds': result.duration_seconds,
            'sample_rate': result.sample_rate
        }
    else:
        return {
            'available': False,
            'image_base64': None,
            'prompt': None,
            'error': result.error
        }
