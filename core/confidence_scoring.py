"""
Confidence scoring module for behavioral analysis assessments.
Extracts, calculates, and displays confidence scores for analysis results.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Individual confidence score for an assessment."""
    category: str
    score: float  # 0.0 to 1.0
    level: str  # "low", "moderate", "high", "very_high"
    reasoning: str = ""


@dataclass
class AnalysisConfidence:
    """Confidence scores for an entire analysis."""
    overall_confidence: float
    overall_level: str
    data_quality_score: float
    scores: List[ConfidenceScore] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# Confidence level thresholds
CONFIDENCE_LEVELS = {
    (0.0, 0.3): "low",
    (0.3, 0.6): "moderate",
    (0.6, 0.8): "high",
    (0.8, 1.0): "very_high"
}


def get_confidence_level(score: float) -> str:
    """Convert numeric score to confidence level."""
    for (low, high), level in CONFIDENCE_LEVELS.items():
        if low <= score < high:
            return level
    return "very_high" if score >= 0.8 else "low"


def extract_confidence_from_text(text: str) -> List[Tuple[str, float]]:
    """
    Extract confidence scores mentioned in analysis text.

    Looks for patterns like:
    - "confidence: 85%"
    - "confidence level: high (0.8)"
    - "certainty: 7/10"
    - "(confidence: moderate)"

    Args:
        text: Analysis text to parse

    Returns:
        List of (category, score) tuples
    """
    scores = []

    # Pattern: "confidence: XX%" or "confidence level: XX%"
    percent_pattern = r'(?:confidence|certainty|reliability)[\s:]+(\d{1,3})%'
    for match in re.finditer(percent_pattern, text, re.IGNORECASE):
        score = int(match.group(1)) / 100.0
        scores.append(("extracted_percent", min(score, 1.0)))

    # Pattern: "confidence: 0.XX" or score in parentheses
    decimal_pattern = r'(?:confidence|certainty)[\s:]+(?:\()?([01]?\.\d+)(?:\))?'
    for match in re.finditer(decimal_pattern, text, re.IGNORECASE):
        score = float(match.group(1))
        scores.append(("extracted_decimal", min(score, 1.0)))

    # Pattern: "X/10" or "X out of 10"
    fraction_pattern = r'(\d+)(?:\s*(?:/|out of)\s*)10'
    for match in re.finditer(fraction_pattern, text, re.IGNORECASE):
        score = int(match.group(1)) / 10.0
        scores.append(("extracted_fraction", min(score, 1.0)))

    # Pattern: confidence levels as text
    level_patterns = {
        r'\b(?:very\s+)?high\s+confidence\b': 0.85,
        r'\bhigh\s+confidence\b': 0.75,
        r'\bmoderate\s+confidence\b': 0.55,
        r'\blow\s+confidence\b': 0.35,
        r'\bvery\s+low\s+confidence\b': 0.2,
        r'\buncertain\b': 0.3,
        r'\bhighly\s+confident\b': 0.85,
        r'\bfairly\s+confident\b': 0.65,
        r'\bsomewhat\s+confident\b': 0.5,
    }

    for pattern, score in level_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            scores.append(("extracted_level", score))

    return scores


def calculate_data_quality_score(
    video_metadata: Dict,
    audio_metadata: Dict,
    transcription_available: bool = False
) -> Tuple[float, List[str]]:
    """
    Calculate a data quality score based on input data characteristics.

    Args:
        video_metadata: Video file metadata
        audio_metadata: Audio extraction metadata
        transcription_available: Whether transcription was successful

    Returns:
        Tuple of (quality_score, list of warnings)
    """
    score = 1.0
    warnings = []

    # Video quality factors
    duration = video_metadata.get('duration_seconds', 0)
    if duration < 15:
        score -= 0.15
        warnings.append("Very short video duration may limit analysis accuracy")
    elif duration < 30:
        score -= 0.05
        warnings.append("Short video duration may affect analysis completeness")

    # Check for native video mode (preferred) or fallback frame extraction
    native_video_used = video_metadata.get('native_video_processing', False) or video_metadata.get('native_video', False)
    if native_video_used:
        # Native video provides full temporal analysis - bonus
        score += 0.1
    else:
        # Fallback: check frame extraction
        frames_extracted = video_metadata.get('frames_extracted', 0)
        if frames_extracted < 3:
            score -= 0.2
            warnings.append("Few frames extracted - limited visual data")
        elif frames_extracted < 5:
            score -= 0.05

    # Resolution check
    resolution = video_metadata.get('resolution', (0, 0))
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        width, height = resolution[0], resolution[1]
        if width < 480 or height < 360:
            score -= 0.1
            warnings.append("Low video resolution may affect visual analysis")

    # Smart selection bonus
    if video_metadata.get('smart_selection_used', False):
        score += 0.05  # Bonus for intelligent frame selection

    # Audio quality factors
    if not audio_metadata:
        score -= 0.25
        warnings.append("No audio data - voice/speech analysis limited")
    else:
        audio_size = audio_metadata.get('size_kb', 0)
        if audio_size < 50:
            score -= 0.1
            warnings.append("Limited audio data may affect speech analysis")

    # Transcription availability
    if not transcription_available:
        score -= 0.1
        warnings.append("Transcription unavailable - linguistic analysis may be limited")

    return max(0.0, min(1.0, score)), warnings


def calculate_analysis_confidence(
    result: Dict,
    video_metadata: Dict = None,
    audio_metadata: Dict = None
) -> AnalysisConfidence:
    """
    Calculate confidence scores for a complete analysis result.

    Args:
        result: Complete profiling result dictionary
        video_metadata: Optional video metadata
        audio_metadata: Optional audio metadata

    Returns:
        AnalysisConfidence object with all scores
    """
    scores = []
    warnings = []

    # Get metadata from result if not provided
    if video_metadata is None:
        video_metadata = result.get('video_metadata', {})
    if audio_metadata is None:
        audio_metadata = result.get('audio_metadata', {})

    analyses = result.get('analyses', {})
    transcription = result.get('transcription', {})

    # Calculate data quality score
    transcription_available = transcription.get('available', False)
    data_quality, data_warnings = calculate_data_quality_score(
        video_metadata,
        audio_metadata,
        transcription_available
    )
    warnings.extend(data_warnings)

    scores.append(ConfidenceScore(
        category="Data Quality",
        score=data_quality,
        level=get_confidence_level(data_quality),
        reasoning="Based on video duration, resolution, frames, and audio availability"
    ))

    # Extract confidence from each analysis
    analysis_scores = []

    # Sam Christensen Essence Analysis
    essence_text = analyses.get('sam_christensen_essence', '')
    if essence_text and not essence_text.startswith('ERROR'):
        essence_scores = extract_confidence_from_text(essence_text)
        if essence_scores:
            avg_score = sum(s[1] for s in essence_scores) / len(essence_scores)
        else:
            # Default based on text length and content
            avg_score = min(0.7, 0.3 + len(essence_text) / 5000)

        scores.append(ConfidenceScore(
            category="Visual Essence Analysis",
            score=avg_score,
            level=get_confidence_level(avg_score),
            reasoning=f"Based on {len(essence_scores)} extracted confidence indicators"
        ))
        analysis_scores.append(avg_score)
    else:
        warnings.append("Visual essence analysis failed or unavailable")

    # Multimodal Analysis
    multimodal_text = analyses.get('multimodal_behavioral', '')
    if multimodal_text and not multimodal_text.startswith('ERROR'):
        mm_scores = extract_confidence_from_text(multimodal_text)
        if mm_scores:
            avg_score = sum(s[1] for s in mm_scores) / len(mm_scores)
        else:
            avg_score = min(0.7, 0.3 + len(multimodal_text) / 6000)

        scores.append(ConfidenceScore(
            category="Multimodal Analysis",
            score=avg_score,
            level=get_confidence_level(avg_score),
            reasoning=f"Based on {len(mm_scores)} extracted confidence indicators"
        ))
        analysis_scores.append(avg_score)
    else:
        warnings.append("Multimodal analysis failed or unavailable")

    # Audio Analysis
    audio_text = analyses.get('audio_voice_analysis', '')
    # Only fail if text is exactly the fallback message, not if it contains the word somewhere
    audio_available = (audio_text and
                       not audio_text.startswith('ERROR') and
                       audio_text.lower() != 'analysis unavailable' and
                       len(audio_text) > 100)  # Reasonable content check
    if audio_available:
        audio_scores = extract_confidence_from_text(audio_text)
        if audio_scores:
            avg_score = sum(s[1] for s in audio_scores) / len(audio_scores)
        else:
            avg_score = min(0.65, 0.25 + len(audio_text) / 5000)

        scores.append(ConfidenceScore(
            category="Audio/Voice Analysis",
            score=avg_score,
            level=get_confidence_level(avg_score),
            reasoning=f"Based on {len(audio_scores)} extracted confidence indicators"
        ))
        analysis_scores.append(avg_score)
    else:
        # Reduce confidence if audio unavailable
        scores.append(ConfidenceScore(
            category="Audio/Voice Analysis",
            score=0.0,
            level="low",
            reasoning="Audio analysis unavailable"
        ))

    # LIWC Analysis
    liwc_text = analyses.get('liwc_linguistic_analysis', '')
    liwc_available = (liwc_text and
                      not liwc_text.startswith('ERROR') and
                      liwc_text.lower() != 'liwc analysis unavailable' and
                      len(liwc_text) > 100)
    if liwc_available:
        liwc_scores = extract_confidence_from_text(liwc_text)
        # Base score on content length (successful analysis = more data)
        content_score = min(0.75, 0.35 + len(liwc_text) / 5000)

        if liwc_scores and len(liwc_scores) >= 3:
            # Multiple explicit indicators - trust them
            avg_score = sum(s[1] for s in liwc_scores) / len(liwc_scores)
        elif liwc_scores:
            # Few indicators - blend with content score (don't let one "low confidence" tank it)
            extracted_avg = sum(s[1] for s in liwc_scores) / len(liwc_scores)
            avg_score = (content_score * 0.6) + (extracted_avg * 0.4)
        else:
            # No explicit indicators - use content-based score
            avg_score = content_score

        scores.append(ConfidenceScore(
            category="Linguistic Analysis",
            score=avg_score,
            level=get_confidence_level(avg_score),
            reasoning=f"Based on {len(liwc_scores)} indicators + content depth ({len(liwc_text)} chars)"
        ))
        analysis_scores.append(avg_score)
    else:
        scores.append(ConfidenceScore(
            category="Linguistic Analysis",
            score=0.0,
            level="low",
            reasoning="Linguistic analysis unavailable"
        ))

    # FBI Synthesis
    fbi_text = analyses.get('fbi_behavioral_synthesis', '')
    if fbi_text and not fbi_text.startswith('ERROR'):
        fbi_scores = extract_confidence_from_text(fbi_text)
        if fbi_scores:
            avg_score = sum(s[1] for s in fbi_scores) / len(fbi_scores)
        else:
            avg_score = min(0.7, 0.35 + len(fbi_text) / 8000)

        scores.append(ConfidenceScore(
            category="FBI Behavioral Synthesis",
            score=avg_score,
            level=get_confidence_level(avg_score),
            reasoning=f"Based on {len(fbi_scores)} extracted confidence indicators"
        ))
        analysis_scores.append(avg_score)
    else:
        warnings.append("FBI synthesis failed or unavailable")

    # Calculate overall confidence
    if analysis_scores:
        # Weighted average: data quality (20%) + analysis scores (80%)
        analysis_avg = sum(analysis_scores) / len(analysis_scores)
        overall = data_quality * 0.2 + analysis_avg * 0.8
    else:
        overall = data_quality * 0.5  # Only data quality available
        warnings.append("No analysis scores available - confidence estimate limited")

    # Processing time factor (very fast may indicate cached/incomplete)
    processing_time = result.get('processing_time_seconds', 0)
    if processing_time < 5 and not result.get('retrieved_from_cache', False):
        warnings.append("Very fast processing time may indicate incomplete analysis")

    return AnalysisConfidence(
        overall_confidence=round(overall, 3),
        overall_level=get_confidence_level(overall),
        data_quality_score=round(data_quality, 3),
        scores=scores,
        warnings=warnings,
        metadata={
            'analysis_count': len(analysis_scores),
            'processing_time': processing_time,
            'from_cache': result.get('retrieved_from_cache', False)
        }
    )


def format_confidence_for_display(confidence: AnalysisConfidence) -> str:
    """
    Format confidence scores for UI display.

    Args:
        confidence: AnalysisConfidence object

    Returns:
        Formatted string for display
    """
    lines = []

    lines.append("=" * 60)
    lines.append("ANALYSIS CONFIDENCE ASSESSMENT")
    lines.append("=" * 60)
    lines.append("")

    # Overall confidence with visual indicator
    level_indicators = {
        "very_high": "████████████████████ VERY HIGH",
        "high": "███████████████░░░░░ HIGH",
        "moderate": "██████████░░░░░░░░░░ MODERATE",
        "low": "█████░░░░░░░░░░░░░░░ LOW"
    }

    indicator = level_indicators.get(confidence.overall_level, "░░░░░░░░░░░░░░░░░░░░")
    lines.append(f"Overall Confidence: {confidence.overall_confidence:.0%}")
    lines.append(f"[{indicator}]")
    lines.append("")

    # Individual scores
    lines.append("-" * 60)
    lines.append("COMPONENT SCORES")
    lines.append("-" * 60)

    for score in confidence.scores:
        bar_length = int(score.score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        lines.append(f"\n{score.category}:")
        lines.append(f"  [{bar}] {score.score:.0%} ({score.level})")
        if score.reasoning:
            lines.append(f"  └─ {score.reasoning}")

    # Warnings
    if confidence.warnings:
        lines.append("")
        lines.append("-" * 60)
        lines.append("CONFIDENCE FACTORS & WARNINGS")
        lines.append("-" * 60)
        for warning in confidence.warnings:
            lines.append(f"  ⚠️ {warning}")

    # Interpretation guide
    lines.append("")
    lines.append("-" * 60)
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 60)
    lines.append("  Very High (80-100%): Strong data, consistent indicators")
    lines.append("  High (60-80%): Good data, reliable analysis")
    lines.append("  Moderate (30-60%): Limited data or mixed signals")
    lines.append("  Low (0-30%): Insufficient data, use with caution")

    return "\n".join(lines)


def add_confidence_to_result(result: Dict) -> Dict:
    """
    Add confidence scores to a profiling result.

    Args:
        result: Complete profiling result dictionary

    Returns:
        Result dictionary with added confidence data
    """
    confidence = calculate_analysis_confidence(result)

    result['confidence'] = {
        'overall': confidence.overall_confidence,
        'level': confidence.overall_level,
        'data_quality': confidence.data_quality_score,
        'components': [
            {
                'category': s.category,
                'score': s.score,
                'level': s.level,
                'reasoning': s.reasoning
            }
            for s in confidence.scores
        ],
        'warnings': confidence.warnings,
        'metadata': confidence.metadata
    }

    return result
