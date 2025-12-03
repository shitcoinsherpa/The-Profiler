"""
Model configuration for the behavioral profiling system.
Defines available models and their capabilities for each analysis stage.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


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


# Available models organized by capability
VISION_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="openai/gpt-4.1",
        name="GPT-4.1",
        provider="OpenAI",
        supports_images=True,
        supports_audio=False,
        cost_tier="premium",
        description="Latest GPT-4 with strong vision capabilities"
    ),
    ModelInfo(
        id="openai/gpt-4o",
        name="GPT-4o",
        provider="OpenAI",
        supports_images=True,
        supports_audio=False,
        cost_tier="standard",
        description="Optimized GPT-4 with vision"
    ),
    ModelInfo(
        id="openai/gpt-4o-mini",
        name="GPT-4o Mini",
        provider="OpenAI",
        supports_images=True,
        supports_audio=False,
        cost_tier="budget",
        description="Fast and affordable GPT-4o variant"
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-4",
        name="Claude Sonnet 4",
        provider="Anthropic",
        supports_images=True,
        supports_audio=False,
        cost_tier="standard",
        description="Anthropic's balanced model with strong analysis"
    ),
    ModelInfo(
        id="anthropic/claude-opus-4",
        name="Claude Opus 4",
        provider="Anthropic",
        supports_images=True,
        supports_audio=False,
        cost_tier="premium",
        description="Anthropic's most capable model"
    ),
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
        id="google/gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        supports_video=True,
        cost_tier="budget",
        description="Previous gen multimodal, still capable"
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

# Audio-capable models (subset of above that support audio)
AUDIO_MODELS: List[ModelInfo] = [
    model for model in VISION_MODELS if model.supports_audio
]

# Text-only synthesis models (don't need vision for final synthesis)
SYNTHESIS_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="openai/gpt-4.1",
        name="GPT-4.1",
        provider="OpenAI",
        supports_images=True,
        cost_tier="premium",
        description="Latest GPT-4 for synthesis"
    ),
    ModelInfo(
        id="openai/gpt-4o",
        name="GPT-4o",
        provider="OpenAI",
        supports_images=True,
        cost_tier="standard",
        description="Optimized GPT-4"
    ),
    ModelInfo(
        id="openai/gpt-4o-mini",
        name="GPT-4o Mini",
        provider="OpenAI",
        supports_images=True,
        cost_tier="budget",
        description="Fast and affordable"
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-4",
        name="Claude Sonnet 4",
        provider="Anthropic",
        supports_images=True,
        cost_tier="standard",
        description="Strong analytical synthesis"
    ),
    ModelInfo(
        id="anthropic/claude-opus-4",
        name="Claude Opus 4",
        provider="Anthropic",
        supports_images=True,
        cost_tier="premium",
        description="Most capable synthesis"
    ),
    ModelInfo(
        id="google/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        cost_tier="standard",
        description="Google's flagship"
    ),
    ModelInfo(
        id="google/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        cost_tier="budget",
        description="Fast Google model"
    ),
    ModelInfo(
        id="google/gemini-3-pro-preview",
        name="Gemini 3 Pro Preview",
        provider="Google",
        supports_images=True,
        supports_audio=True,
        cost_tier="premium",
        description="Google's flagship frontier model"
    ),
]


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
        cost_tier="premium",
        description="Recommended for meta-analysis (1M context)"
    ),
    ModelInfo(
        id="anthropic/claude-opus-4",
        name="Claude Opus 4",
        provider="Anthropic",
        supports_images=True,
        cost_tier="premium",
        description="Alternative for meta-analysis"
    ),
    ModelInfo(
        id="openai/gpt-4.1",
        name="GPT-4.1",
        provider="OpenAI",
        supports_images=True,
        cost_tier="premium",
        description="Alternative for meta-analysis"
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
    if stage == "essence":
        # Vision models for Sam Christensen analysis
        models = VISION_MODELS
    elif stage in ("multimodal", "audio", "liwc"):
        # Must support audio for these stages
        models = AUDIO_MODELS
    elif stage == "synthesis":
        # Any model works for text synthesis
        models = SYNTHESIS_MODELS
    else:
        models = VISION_MODELS

    return [(f"{m.name} ({m.provider}) - {m.cost_tier}", m.id) for m in models]


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """
    Get model info by ID.

    Args:
        model_id: The model identifier (e.g., 'openai/gpt-4.1')

    Returns:
        ModelInfo object or None if not found
    """
    all_models = VISION_MODELS + SYNTHESIS_MODELS
    for model in all_models:
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

    if stage == "essence":
        if not model.supports_images:
            return False, f"{model.name} does not support image analysis"
    elif stage in ("multimodal", "audio", "liwc"):
        if not model.supports_audio:
            return False, f"{model.name} does not support audio analysis. Use a Gemini model."

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
    return defaults.get(stage, "openai/gpt-4.1")


# Export dropdown choices for UI
ESSENCE_MODEL_CHOICES = get_model_choices_for_stage("essence")
MULTIMODAL_MODEL_CHOICES = get_model_choices_for_stage("multimodal")
AUDIO_MODEL_CHOICES = get_model_choices_for_stage("audio")
LIWC_MODEL_CHOICES = get_model_choices_for_stage("liwc")
SYNTHESIS_MODEL_CHOICES = get_model_choices_for_stage("synthesis")
