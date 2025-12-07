"""
Model configuration for the behavioral profiling system.
Defines available models and their capabilities for each analysis stage.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an AI model."""
    id: str
    name: str
    provider: str
    supports_images: bool = True
    supports_audio: bool = False
    supports_video: bool = False
    cost_tier: str = "standard"  # "budget", "standard", "premium"
    description: str = ""


# Available models - Gemini only (native video support required)
VISION_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="google/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        supports_video=True,
        cost_tier="budget",
        description="Fast multimodal with audio/video support"
    ),
    ModelInfo(
        id="google/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        supports_video=True,
        cost_tier="standard",
        description="Powerful multimodal with audio/video support"
    ),
    ModelInfo(
        id="google/gemini-3-pro-preview",
        name="Gemini 3 Pro Preview",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        supports_video=True,
        cost_tier="premium",
        description="Google's flagship frontier model with 1M context"
    ),
]

# Audio-capable models (all Gemini models support audio)
AUDIO_MODELS: List[ModelInfo] = [
    model for model in VISION_MODELS if model.supports_audio
]

# Synthesis models (same as vision - all support native video)
SYNTHESIS_MODELS: List[ModelInfo] = VISION_MODELS.copy()


# ==================================================================================
# DEV META-ANALYSIS MODELS - TO REMOVE BEFORE PRODUCTION
# ==================================================================================
DEV_META_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="google/gemini-3-pro-preview",
        name="Gemini 3 Pro Preview",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        supports_video=True,
        cost_tier="premium",
        description="Recommended for meta-analysis (1M context)"
    ),
]

# Default meta-analysis model
DEFAULT_DEV_META_MODEL = "google/gemini-3-pro-preview"


@dataclass
class StageModelConfig:
    """Configuration for which model to use at each pipeline stage."""
    essence_model: str = "google/gemini-3-pro-preview"  # Visual analysis (native video)
    multimodal_model: str = "google/gemini-3-pro-preview"  # Multimodal analysis (video + audio)
    audio_model: str = "google/gemini-3-pro-preview"  # Audio-only analysis
    liwc_model: str = "google/gemini-3-pro-preview"  # LIWC linguistic analysis (audio)
    synthesis_model: str = "google/gemini-3-pro-preview"  # FBI synthesis (text)


# Default configuration
DEFAULT_MODEL_CONFIG = StageModelConfig()


def get_model_choices_for_stage(stage: str) -> List[tuple]:
    """
    Get available model choices for a specific pipeline stage.

    Args:
        stage: One of 'essence', 'multimodal', 'audio', 'liwc', 'synthesis'

    Returns:
        List of (display_name, model_id) tuples for Gradio dropdown
    """
    # All stages use the same Gemini models (native video support)
    models = VISION_MODELS
    return [(f"{m.name} ({m.provider}) - {m.cost_tier}", m.id) for m in models]


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """
    Get model info by ID.

    Args:
        model_id: The model identifier (e.g., 'google/gemini-2.5-flash')

    Returns:
        ModelInfo object or None if not found
    """
    for model in VISION_MODELS:
        if model.id == model_id:
            return model
    return None


def validate_model_for_stage(model_id: str, stage: str) -> tuple[bool, str]:
    """
    Validate that a model is appropriate for a given stage.

    Args:
        model_id: The model identifier
        stage: The pipeline stage

    Returns:
        Tuple of (is_valid, error_message)
    """
    model = get_model_info(model_id)
    if not model:
        return False, f"Unknown model: {model_id}"

    # All Gemini models support all stages (video, audio, images)
    return True, ""


def get_default_model_for_stage(stage: str) -> str:
    """Get the default model ID for a pipeline stage."""
    defaults = {
        "essence": DEFAULT_MODEL_CONFIG.essence_model,
        "multimodal": DEFAULT_MODEL_CONFIG.multimodal_model,
        "audio": DEFAULT_MODEL_CONFIG.audio_model,
        "liwc": DEFAULT_MODEL_CONFIG.liwc_model,
        "synthesis": DEFAULT_MODEL_CONFIG.synthesis_model,
    }
    return defaults.get(stage, "google/gemini-3-pro-preview")


# Export dropdown choices for UI
ESSENCE_MODEL_CHOICES = get_model_choices_for_stage("essence")
MULTIMODAL_MODEL_CHOICES = get_model_choices_for_stage("multimodal")
AUDIO_MODEL_CHOICES = get_model_choices_for_stage("audio")
LIWC_MODEL_CHOICES = get_model_choices_for_stage("liwc")
SYNTHESIS_MODEL_CHOICES = get_model_choices_for_stage("synthesis")
