"""
JSON schemas for structured output from AI models.
Used with OpenRouter's response_format parameter for consistent, parseable responses.
"""

from typing import Dict, Any, Optional


# Schema for Visual Essence Analysis (Sam Christensen method)
VISUAL_ESSENCE_SCHEMA = {
    "name": "visual_essence_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "subject_overview": {
                "type": "object",
                "properties": {
                    "primary_impression": {"type": "string"},
                    "energy_level": {"type": "string", "enum": ["low", "moderate", "high", "very_high"]},
                    "confidence_rating": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["primary_impression", "energy_level", "confidence_rating"],
                "additionalProperties": False
            },
            "physical_presence": {
                "type": "object",
                "properties": {
                    "posture": {"type": "string"},
                    "movement_patterns": {"type": "string"},
                    "spatial_awareness": {"type": "string"},
                    "physical_confidence": {"type": "string"}
                },
                "required": ["posture", "movement_patterns", "spatial_awareness", "physical_confidence"],
                "additionalProperties": False
            },
            "facial_expressions": {
                "type": "object",
                "properties": {
                    "dominant_expressions": {"type": "array", "items": {"type": "string"}},
                    "micro_expressions": {"type": "string"},
                    "eye_contact_patterns": {"type": "string"},
                    "emotional_authenticity": {"type": "string"}
                },
                "required": ["dominant_expressions", "micro_expressions", "eye_contact_patterns", "emotional_authenticity"],
                "additionalProperties": False
            },
            "personal_style": {
                "type": "object",
                "properties": {
                    "clothing_analysis": {"type": "string"},
                    "grooming": {"type": "string"},
                    "style_consistency": {"type": "string"},
                    "image_projection": {"type": "string"}
                },
                "required": ["clothing_analysis", "grooming", "style_consistency", "image_projection"],
                "additionalProperties": False
            },
            "behavioral_indicators": {
                "type": "object",
                "properties": {
                    "comfort_level": {"type": "string"},
                    "self_awareness": {"type": "string"},
                    "adaptability_cues": {"type": "string"},
                    "stress_indicators": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["comfort_level", "self_awareness", "adaptability_cues", "stress_indicators"],
                "additionalProperties": False
            },
            "essence_summary": {
                "type": "object",
                "properties": {
                    "core_traits": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "areas_of_concern": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "overall_assessment": {"type": "string"}
                },
                "required": ["core_traits", "strengths", "areas_of_concern", "overall_assessment"],
                "additionalProperties": False
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["subject_overview", "physical_presence", "facial_expressions", "personal_style", "behavioral_indicators", "essence_summary", "confidence_score"],
        "additionalProperties": False
    }
}


# Schema for Multimodal Behavioral Analysis
MULTIMODAL_BEHAVIORAL_SCHEMA = {
    "name": "multimodal_behavioral_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verbal_behavior": {
                "type": "object",
                "properties": {
                    "speech_patterns": {"type": "string"},
                    "vocabulary_complexity": {"type": "string"},
                    "communication_style": {"type": "string"},
                    "persuasion_techniques": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["speech_patterns", "vocabulary_complexity", "communication_style", "persuasion_techniques"],
                "additionalProperties": False
            },
            "nonverbal_behavior": {
                "type": "object",
                "properties": {
                    "gestures": {"type": "string"},
                    "facial_dynamics": {"type": "string"},
                    "body_language": {"type": "string"},
                    "congruence_assessment": {"type": "string"}
                },
                "required": ["gestures", "facial_dynamics", "body_language", "congruence_assessment"],
                "additionalProperties": False
            },
            "emotional_indicators": {
                "type": "object",
                "properties": {
                    "primary_emotions": {"type": "array", "items": {"type": "string"}},
                    "emotional_regulation": {"type": "string"},
                    "authenticity_markers": {"type": "string"},
                    "stress_responses": {"type": "string"}
                },
                "required": ["primary_emotions", "emotional_regulation", "authenticity_markers", "stress_responses"],
                "additionalProperties": False
            },
            "cognitive_patterns": {
                "type": "object",
                "properties": {
                    "thought_organization": {"type": "string"},
                    "problem_solving_approach": {"type": "string"},
                    "attention_patterns": {"type": "string"},
                    "memory_indicators": {"type": "string"}
                },
                "required": ["thought_organization", "problem_solving_approach", "attention_patterns", "memory_indicators"],
                "additionalProperties": False
            },
            "interpersonal_dynamics": {
                "type": "object",
                "properties": {
                    "rapport_building": {"type": "string"},
                    "dominance_submission": {"type": "string"},
                    "trustworthiness_cues": {"type": "string"},
                    "manipulation_indicators": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["rapport_building", "dominance_submission", "trustworthiness_cues", "manipulation_indicators"],
                "additionalProperties": False
            },
            "risk_assessment": {
                "type": "object",
                "properties": {
                    "deception_indicators": {"type": "array", "items": {"type": "string"}},
                    "aggression_potential": {"type": "string"},
                    "stability_assessment": {"type": "string"},
                    "risk_level": {"type": "string", "enum": ["low", "moderate", "elevated", "high"]}
                },
                "required": ["deception_indicators", "aggression_potential", "stability_assessment", "risk_level"],
                "additionalProperties": False
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["verbal_behavior", "nonverbal_behavior", "emotional_indicators", "cognitive_patterns", "interpersonal_dynamics", "risk_assessment", "confidence_score"],
        "additionalProperties": False
    }
}


# Schema for Audio/Voice Analysis
AUDIO_VOICE_SCHEMA = {
    "name": "audio_voice_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "vocal_characteristics": {
                "type": "object",
                "properties": {
                    "pitch_profile": {"type": "string"},
                    "volume_patterns": {"type": "string"},
                    "speech_rate": {"type": "string"},
                    "vocal_quality": {"type": "string"}
                },
                "required": ["pitch_profile", "volume_patterns", "speech_rate", "vocal_quality"],
                "additionalProperties": False
            },
            "paralinguistic_features": {
                "type": "object",
                "properties": {
                    "hesitations": {"type": "string"},
                    "filler_words": {"type": "array", "items": {"type": "string"}},
                    "breathing_patterns": {"type": "string"},
                    "emotional_leakage": {"type": "string"}
                },
                "required": ["hesitations", "filler_words", "breathing_patterns", "emotional_leakage"],
                "additionalProperties": False
            },
            "emotional_tone": {
                "type": "object",
                "properties": {
                    "dominant_emotion": {"type": "string"},
                    "emotional_variability": {"type": "string"},
                    "stress_indicators": {"type": "array", "items": {"type": "string"}},
                    "confidence_level": {"type": "string"}
                },
                "required": ["dominant_emotion", "emotional_variability", "stress_indicators", "confidence_level"],
                "additionalProperties": False
            },
            "truthfulness_indicators": {
                "type": "object",
                "properties": {
                    "consistency": {"type": "string"},
                    "spontaneity": {"type": "string"},
                    "vocal_stress": {"type": "string"},
                    "deception_markers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["consistency", "spontaneity", "vocal_stress", "deception_markers"],
                "additionalProperties": False
            },
            "personality_inferences": {
                "type": "object",
                "properties": {
                    "extraversion_indicators": {"type": "string"},
                    "dominance_level": {"type": "string"},
                    "warmth_indicators": {"type": "string"},
                    "overall_impression": {"type": "string"}
                },
                "required": ["extraversion_indicators", "dominance_level", "warmth_indicators", "overall_impression"],
                "additionalProperties": False
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["vocal_characteristics", "paralinguistic_features", "emotional_tone", "truthfulness_indicators", "personality_inferences", "confidence_score"],
        "additionalProperties": False
    }
}


# Schema for LIWC-Style Linguistic Analysis
LIWC_LINGUISTIC_SCHEMA = {
    "name": "liwc_linguistic_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "word_categories": {
                "type": "object",
                "properties": {
                    "self_references": {"type": "string"},
                    "social_words": {"type": "string"},
                    "positive_emotion": {"type": "string"},
                    "negative_emotion": {"type": "string"},
                    "cognitive_processes": {"type": "string"},
                    "certainty_markers": {"type": "string"}
                },
                "required": ["self_references", "social_words", "positive_emotion", "negative_emotion", "cognitive_processes", "certainty_markers"],
                "additionalProperties": False
            },
            "linguistic_style": {
                "type": "object",
                "properties": {
                    "formality_level": {"type": "string"},
                    "complexity": {"type": "string"},
                    "authenticity": {"type": "string"},
                    "analytical_thinking": {"type": "string"}
                },
                "required": ["formality_level", "complexity", "authenticity", "analytical_thinking"],
                "additionalProperties": False
            },
            "psychological_markers": {
                "type": "object",
                "properties": {
                    "anxiety_indicators": {"type": "string"},
                    "depression_markers": {"type": "string"},
                    "anger_indicators": {"type": "string"},
                    "power_dynamics": {"type": "string"}
                },
                "required": ["anxiety_indicators", "depression_markers", "anger_indicators", "power_dynamics"],
                "additionalProperties": False
            },
            "narrative_patterns": {
                "type": "object",
                "properties": {
                    "temporal_focus": {"type": "string"},
                    "causal_reasoning": {"type": "string"},
                    "certainty_level": {"type": "string"},
                    "personal_vs_impersonal": {"type": "string"}
                },
                "required": ["temporal_focus", "causal_reasoning", "certainty_level", "personal_vs_impersonal"],
                "additionalProperties": False
            },
            "deception_analysis": {
                "type": "object",
                "properties": {
                    "pronoun_patterns": {"type": "string"},
                    "detail_level": {"type": "string"},
                    "hedging_language": {"type": "string"},
                    "overall_assessment": {"type": "string"}
                },
                "required": ["pronoun_patterns", "detail_level", "hedging_language", "overall_assessment"],
                "additionalProperties": False
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["word_categories", "linguistic_style", "psychological_markers", "narrative_patterns", "deception_analysis", "confidence_score"],
        "additionalProperties": False
    }
}


# Schema for FBI Behavioral Synthesis
FBI_SYNTHESIS_SCHEMA = {
    "name": "fbi_behavioral_synthesis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "object",
                "properties": {
                    "overall_assessment": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "risk_level": {"type": "string", "enum": ["low", "low_moderate", "moderate", "moderate_high", "high"]},
                    "confidence_level": {"type": "string", "enum": ["low", "moderate", "high", "very_high"]}
                },
                "required": ["overall_assessment", "key_findings", "risk_level", "confidence_level"],
                "additionalProperties": False
            },
            "personality_profile": {
                "type": "object",
                "properties": {
                    "dominant_traits": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "personality_type": {"type": "string"},
                    "emotional_stability": {"type": "string"},
                    "interpersonal_style": {"type": "string"}
                },
                "required": ["dominant_traits", "personality_type", "emotional_stability", "interpersonal_style"],
                "additionalProperties": False
            },
            "behavioral_patterns": {
                "type": "object",
                "properties": {
                    "communication_style": {"type": "string"},
                    "decision_making": {"type": "string"},
                    "stress_response": {"type": "string"},
                    "adaptability": {"type": "string"}
                },
                "required": ["communication_style", "decision_making", "stress_response", "adaptability"],
                "additionalProperties": False
            },
            "risk_indicators": {
                "type": "object",
                "properties": {
                    "deception_likelihood": {"type": "string"},
                    "manipulation_tendency": {"type": "string"},
                    "violence_potential": {"type": "string"},
                    "substance_indicators": {"type": "string"}
                },
                "required": ["deception_likelihood", "manipulation_tendency", "violence_potential", "substance_indicators"],
                "additionalProperties": False
            },
            "recommendations": {
                "type": "object",
                "properties": {
                    "interview_approach": {"type": "string"},
                    "areas_for_investigation": {"type": "array", "items": {"type": "string"}},
                    "monitoring_needs": {"type": "string"},
                    "additional_assessments": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["interview_approach", "areas_for_investigation", "monitoring_needs", "additional_assessments"],
                "additionalProperties": False
            },
            "data_quality": {
                "type": "object",
                "properties": {
                    "video_quality": {"type": "string"},
                    "audio_quality": {"type": "string"},
                    "analysis_limitations": {"type": "array", "items": {"type": "string"}},
                    "confidence_factors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["video_quality", "audio_quality", "analysis_limitations", "confidence_factors"],
                "additionalProperties": False
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["executive_summary", "personality_profile", "behavioral_patterns", "risk_indicators", "recommendations", "data_quality", "confidence_score"],
        "additionalProperties": False
    }
}


# Mapping of analysis stages to their schemas
ANALYSIS_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "essence": VISUAL_ESSENCE_SCHEMA,
    "multimodal": MULTIMODAL_BEHAVIORAL_SCHEMA,
    "audio": AUDIO_VOICE_SCHEMA,
    "liwc": LIWC_LINGUISTIC_SCHEMA,
    "synthesis": FBI_SYNTHESIS_SCHEMA
}


def get_schema(stage: str) -> Optional[Dict[str, Any]]:
    """
    Get the JSON schema for a specific analysis stage.

    Args:
        stage: Analysis stage name (essence, multimodal, audio, liwc, synthesis)

    Returns:
        JSON schema dict or None if not found
    """
    return ANALYSIS_SCHEMAS.get(stage)


def build_response_format(stage: str) -> Optional[Dict[str, Any]]:
    """
    Build the response_format parameter for OpenRouter API.

    Args:
        stage: Analysis stage name

    Returns:
        Response format dict for API call, or None if schema not found
    """
    schema = get_schema(stage)
    if schema is None:
        return None

    return {
        "type": "json_schema",
        "json_schema": schema
    }


def format_structured_response(response_dict: Dict, stage: str) -> str:
    """
    Format a structured JSON response into readable text.

    Args:
        response_dict: Parsed JSON response
        stage: Analysis stage for formatting

    Returns:
        Formatted text string
    """
    import json

    def format_section(title: str, data: Any, indent: int = 0) -> str:
        """Recursively format a section."""
        lines = []
        prefix = "  " * indent

        if isinstance(data, dict):
            lines.append(f"{prefix}{title.upper().replace('_', ' ')}:")
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(format_section(key, value, indent + 1))
                else:
                    formatted_key = key.replace('_', ' ').title()
                    lines.append(f"{prefix}  {formatted_key}: {value}")
        elif isinstance(data, list):
            lines.append(f"{prefix}{title.upper().replace('_', ' ')}:")
            for item in data:
                if isinstance(item, dict):
                    lines.append(format_section("", item, indent + 1))
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{title}: {data}")

        return "\n".join(lines)

    output_lines = []

    # Add header based on stage
    stage_headers = {
        "essence": "SAM CHRISTENSEN VISUAL ESSENCE ANALYSIS",
        "multimodal": "COMPREHENSIVE MULTIMODAL BEHAVIORAL ANALYSIS",
        "audio": "AUDIO/VOICE ANALYSIS",
        "liwc": "LIWC-STYLE LINGUISTIC ANALYSIS",
        "synthesis": "FBI BEHAVIORAL SYNTHESIS PROFILE"
    }

    header = stage_headers.get(stage, "ANALYSIS RESULTS")
    output_lines.append("=" * 60)
    output_lines.append(header)
    output_lines.append("=" * 60)
    output_lines.append("")

    # Format each top-level section
    for key, value in response_dict.items():
        if key == "confidence_score":
            output_lines.append(f"\nOVERALL CONFIDENCE: {value:.0%}")
        else:
            output_lines.append(format_section(key, value))
            output_lines.append("")

    return "\n".join(output_lines)
