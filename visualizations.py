"""
Visualization module for FBI Behavioral Profiling System.
Creates Plotly charts from real analysis data - NO placeholders or mock data.
All visualizations parse actual results from the analysis pipeline.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Check if plotly is available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - visualizations will be disabled")

# FBI Theme Colors
FBI_COLORS = {
    'primary': '#4a9eff',      # Accent blue
    'secondary': '#2d6bbf',    # Darker blue
    'background': '#0a0e1a',   # Dark background
    'panel': '#1e2842',        # Panel background
    'text': '#e8eaf0',         # Light text
    'text_secondary': '#6b7280',  # Gray text
    'warning': '#ff9500',      # Orange warning
    'danger': '#ef4444',       # Red danger
    'success': '#22c55e',      # Green success
    'gold': '#ffd60a',         # Gold accent
}


def extract_big_five_scores(text: str) -> Optional[Dict[str, float]]:
    """
    Extract Big Five personality scores from analysis text.

    Looks for FBI synthesis format like:
    - "Openness to Experience: 75 | High | Evidence..."
    - "- Openness: 75/100"
    - "Openness (85%)"

    Returns None if scores cannot be reliably extracted.
    """
    if not text:
        return None

    # First, try to find the Big Five section specifically
    big_five_section = ""
    section_patterns = [
        r'Big Five Assessment[^\n]*\n([\s\S]*?)(?=\n[A-Z]{2,}|\nDARK TRIAD|\nMBTI|$)',
        r'PERSONALITY STRUCTURE[^\n]*\n([\s\S]*?)(?=\nDARK TRIAD|\nCOMMUNICATION|$)',
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            big_five_section = match.group(1)
            logger.debug(f"Found Big Five section: {big_five_section[:200]}...")
            break

    # Use section if found, otherwise search full text
    search_text = big_five_section if big_five_section else text

    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    scores = {}

    for trait in traits:
        # Multiple patterns to catch various output formats
        patterns = [
            # FBI format: "- Openness to Experience: 75 | High | ..."
            rf'-?\s*{trait}(?:\s+to\s+Experience)?(?:/Emotional Stability)?[\s:]+(\d{{1,3}})\s*\|',
            # "Trait: 75" or "Trait: 75/100"
            rf'{trait}(?:\s+to\s+Experience)?(?:/Emotional Stability)?[\s:]+(\d{{1,3}})(?:/100|\s*%|\b)',
            # "Trait (75%)" or "Trait (75)"
            rf'{trait}(?:\s+to\s+Experience)?\s*\((\d{{1,3}})%?\)',
            # "- Trait: [75]" with brackets
            rf'-?\s*{trait}[^:]*:\s*\[(\d{{1,3}})\]',
            # "Trait to Experience: 75" with any separator
            rf'{trait}(?:\s+to\s+Experience)?(?:/Emotional Stability)?\s*[:=]\s*(\d{{1,3}})',
            # Looser: "Trait" followed by number within 30 chars
            rf'{trait}[^0-9]{{0,30}}(\d{{1,3}})(?:\s*[%/]|\s*\||\s|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 0 <= score <= 100:
                        scores[trait] = score / 100.0  # Normalize to 0-1
                        logger.debug(f"Extracted {trait}: {score}")
                        break
                except (ValueError, IndexError):
                    continue

    logger.info(f"Big Five extraction: found {len(scores)}/5 traits: {list(scores.keys())}")

    # Return if we found at least 2 traits (more lenient)
    if len(scores) >= 2:
        # Fill missing with None (will be handled in visualization)
        for trait in traits:
            if trait not in scores:
                scores[trait] = None
        return scores

    return None


def extract_dark_triad_scores(text: str) -> Optional[Dict[str, float]]:
    """
    Extract Dark Triad scores from analysis text.

    Looks for FBI synthesis format like:
    - "NARCISSISM: 65 | High"
    - "- NARCISSISM: 65/100"
    - "Narcissism score: 65"

    Returns None if scores cannot be reliably extracted.
    """
    if not text:
        return None

    # First, try to find the Dark Triad section specifically
    dark_triad_section = ""
    section_patterns = [
        r'DARK TRIAD ASSESSMENT[^\n]*\n([\s\S]*?)(?=\nMESSIAH|\nMBTI|\nCOMMUNICATION|\nTHREAT|$)',
        r'Dark Triad[^\n]*\n([\s\S]*?)(?=\nMESSIAH|\nMBTI|\nCOMMUNICATION|$)',
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dark_triad_section = match.group(1)
            logger.debug(f"Found Dark Triad section: {dark_triad_section[:200]}...")
            break

    # Use section if found, otherwise search full text
    search_text = dark_triad_section if dark_triad_section else text

    traits = ['Narcissism', 'Machiavellianism', 'Psychopathy']
    scores = {}

    for trait in traits:
        patterns = [
            # FBI format: "- NARCISSISM: 65 | High"
            rf'-?\s*{trait}[\s:]+(\d{{1,3}})\s*\|',
            # "TRAIT: XX" or "TRAIT: XX/100"
            rf'{trait}[\s:]+(\d{{1,3}})(?:/100|\s*%|\b)',
            # "- TRAIT: [65]" with brackets
            rf'-?\s*{trait}[^:]*:\s*\[(\d{{1,3}})\]',
            # "TRAIT score: XX" or "TRAIT rating: XX"
            rf'{trait}\s+(?:score|rating|level)?[\s:]+(\d{{1,3}})',
            # Looser: trait followed by number within 40 chars
            rf'{trait}[^0-9]{{0,40}}(\d{{1,3}})(?:\s*[%/]|\s*\||\s|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 0 <= score <= 100:
                        scores[trait] = score / 100.0
                        logger.debug(f"Extracted {trait}: {score}")
                        break
                except (ValueError, IndexError):
                    continue

    logger.info(f"Dark Triad extraction: found {len(scores)}/3 traits: {list(scores.keys())}")

    # Return if we found at least 1 trait (very lenient for Dark Triad)
    if len(scores) >= 1:
        for trait in traits:
            if trait not in scores:
                scores[trait] = None
        return scores

    return None


def extract_threat_scores(text: str) -> Optional[Dict[str, float]]:
    """
    Extract threat assessment scores from analysis text.

    Looks for FBI threat matrix format like:
    - "Volatility risk (emotional instability): 45"
    - "Manipulation capacity: 70/100"
    - "Compliance likelihood: 80"
    """
    if not text:
        return None

    # First, try to find the Threat Assessment section specifically
    threat_section = ""
    section_patterns = [
        r'THREAT ASSESSMENT MATRIX[^\n]*\n([\s\S]*?)(?=\nVULNERABILITY|\nPREDICTIVE|\nOPERATIONAL|$)',
        r'Threat Assessment[^\n]*\n([\s\S]*?)(?=\nVULNERABILITY|\nPREDICTIVE|$)',
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            threat_section = match.group(1)
            logger.debug(f"Found Threat section: {threat_section[:200]}...")
            break

    # Use section if found, otherwise search full text
    search_text = threat_section if threat_section else text

    categories = [
        ('Volatility', ['volatility risk', 'volatility', 'emotional instability']),
        ('Manipulation', ['manipulation capacity', 'manipulation']),
        ('Compliance', ['compliance likelihood', 'compliance']),
        ('Stress Resilience', ['stress resilience', 'stress']),
        ('Ethical Boundaries', ['ethical boundaries', 'ethical']),
    ]

    scores = {}

    for name, keywords in categories:
        for keyword in keywords:
            patterns = [
                # "Keyword: 75" or "Keyword: 75/100"
                rf'{keyword}[^0-9]{{0,30}}(\d{{1,3}})(?:/100|\s*%|\s*\||\b)',
                # "- Keyword: 75"
                rf'-\s*{keyword}[^:]*:\s*(\d{{1,3}})',
            ]
            found = False
            for pattern in patterns:
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    try:
                        score = int(match.group(1))
                        if 0 <= score <= 100:
                            scores[name] = score / 100.0
                            logger.debug(f"Extracted {name}: {score}")
                            found = True
                            break
                    except (ValueError, IndexError):
                        continue
            if found:
                break

    logger.info(f"Threat extraction: found {len(scores)}/5 categories: {list(scores.keys())}")

    # Return if we found at least 1 category
    if len(scores) >= 1:
        return scores

    return None


def extract_mbti_type(text: str) -> Optional[str]:
    """
    Extract MBTI type from analysis text.

    Looks for 4-letter MBTI codes like ENTJ, ISFP, etc.
    """
    if not text:
        return None

    # Pattern for MBTI type (4 letters)
    pattern = r'\b([EI][NS][TF][JP])\b'
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    return None


def create_confidence_gauge(confidence_data: Dict) -> Optional[Any]:
    """
    Create a gauge chart for overall confidence score.

    Args:
        confidence_data: The confidence dictionary from result['confidence']

    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None

    if not confidence_data or 'overall' not in confidence_data:
        return None

    score = confidence_data['overall']
    level = confidence_data.get('level', 'unknown')

    # Determine color based on level
    if score >= 0.8:
        color = FBI_COLORS['success']
    elif score >= 0.6:
        color = FBI_COLORS['primary']
    elif score >= 0.3:
        color = FBI_COLORS['warning']
    else:
        color = FBI_COLORS['danger']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': '%', 'font': {'size': 40, 'color': FBI_COLORS['text']}},
        title={'text': "Overall Confidence", 'font': {'size': 16, 'color': FBI_COLORS['text']}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': FBI_COLORS['text_secondary'],
                'tickfont': {'color': FBI_COLORS['text_secondary']},
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': FBI_COLORS['panel'],
            'borderwidth': 2,
            'bordercolor': FBI_COLORS['text_secondary'],
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [30, 60], 'color': 'rgba(255, 149, 0, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(74, 158, 255, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(34, 197, 94, 0.2)'},
            ],
            'threshold': {
                'line': {'color': FBI_COLORS['gold'], 'width': 3},
                'thickness': 0.8,
                'value': score * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        height=250,
        margin=dict(l=30, r=30, t=50, b=30),
    )

    return fig


def create_component_confidence_bars(confidence_data: Dict) -> Optional[Any]:
    """
    Create horizontal bar chart for component confidence scores.

    Args:
        confidence_data: The confidence dictionary from result['confidence']

    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None

    if not confidence_data or 'components' not in confidence_data:
        return None

    components = confidence_data['components']
    if not components:
        return None

    categories = [c['category'] for c in components]
    scores = [c['score'] * 100 for c in components]

    # Color based on score
    colors = []
    for score in scores:
        if score >= 80:
            colors.append(FBI_COLORS['success'])
        elif score >= 60:
            colors.append(FBI_COLORS['primary'])
        elif score >= 30:
            colors.append(FBI_COLORS['warning'])
        else:
            colors.append(FBI_COLORS['danger'])

    fig = go.Figure(go.Bar(
        x=scores,
        y=categories,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.0f}%' for s in scores],
        textposition='inside',
        textfont={'color': FBI_COLORS['text'], 'size': 12},
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        xaxis={
            'range': [0, 100],
            'title': 'Confidence %',
            'gridcolor': FBI_COLORS['panel'],
            'tickfont': {'color': FBI_COLORS['text_secondary']},
        },
        yaxis={
            'tickfont': {'color': FBI_COLORS['text']},
        },
        height=250,
        margin=dict(l=150, r=30, t=30, b=50),
        showlegend=False,
    )

    return fig


def create_big_five_radar(analysis_text: str) -> Optional[Any]:
    """
    Create radar chart for Big Five personality scores.

    Args:
        analysis_text: The FBI synthesis or multimodal analysis text

    Returns:
        Plotly figure or None if scores cannot be extracted
    """
    if not PLOTLY_AVAILABLE:
        return None

    scores = extract_big_five_scores(analysis_text)
    if not scores:
        return None

    # Filter out None values and prepare data
    traits = []
    values = []
    for trait, score in scores.items():
        if score is not None:
            traits.append(trait)
            values.append(score * 100)

    if len(traits) < 2:
        logger.info(f"Big Five radar: insufficient traits ({len(traits)} < 2)")
        return None

    # Close the radar by repeating first value
    traits.append(traits[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=traits,
        fill='toself',
        fillcolor=f'rgba(74, 158, 255, 0.3)',
        line=dict(color=FBI_COLORS['primary'], width=2),
        marker=dict(size=8, color=FBI_COLORS['primary']),
        name='Big Five'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont={'color': FBI_COLORS['text_secondary'], 'size': 10},
                gridcolor=FBI_COLORS['panel'],
            ),
            angularaxis=dict(
                tickfont={'color': FBI_COLORS['text'], 'size': 11},
                gridcolor=FBI_COLORS['panel'],
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40),
        title={
            'text': 'Big Five Personality Profile',
            'font': {'size': 14, 'color': FBI_COLORS['text']},
            'x': 0.5,
        }
    )

    return fig


def create_dark_triad_bars(analysis_text: str) -> Optional[Any]:
    """
    Create horizontal bar chart for Dark Triad scores.

    Args:
        analysis_text: The FBI synthesis text

    Returns:
        Plotly figure or None if scores cannot be extracted
    """
    if not PLOTLY_AVAILABLE:
        return None

    scores = extract_dark_triad_scores(analysis_text)
    if not scores:
        return None

    # Prepare data (filter None values)
    traits = []
    values = []
    for trait, score in scores.items():
        if score is not None:
            traits.append(trait)
            values.append(score * 100)

    if len(traits) < 1:
        logger.info("Dark Triad bars: no traits extracted")
        return None

    # Color intensity based on score (higher = more red/dangerous)
    colors = []
    for value in values:
        if value >= 70:
            colors.append(FBI_COLORS['danger'])
        elif value >= 50:
            colors.append(FBI_COLORS['warning'])
        elif value >= 30:
            colors.append(FBI_COLORS['primary'])
        else:
            colors.append(FBI_COLORS['success'])

    fig = go.Figure(go.Bar(
        x=values,
        y=traits,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.0f}' for v in values],
        textposition='inside',
        textfont={'color': FBI_COLORS['text'], 'size': 14, 'family': 'Arial Black'},
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        xaxis={
            'range': [0, 100],
            'title': 'Score (0-100)',
            'gridcolor': FBI_COLORS['panel'],
            'tickfont': {'color': FBI_COLORS['text_secondary']},
        },
        yaxis={
            'tickfont': {'color': FBI_COLORS['text'], 'size': 12},
        },
        height=200,
        margin=dict(l=120, r=30, t=40, b=50),
        showlegend=False,
        title={
            'text': 'Dark Triad Assessment',
            'font': {'size': 14, 'color': FBI_COLORS['text']},
            'x': 0.5,
        }
    )

    return fig


def create_threat_matrix(analysis_text: str) -> Optional[Any]:
    """
    Create horizontal bar chart for threat assessment scores.

    Args:
        analysis_text: The FBI synthesis text

    Returns:
        Plotly figure or None if scores cannot be extracted
    """
    if not PLOTLY_AVAILABLE:
        return None

    scores = extract_threat_scores(analysis_text)
    if not scores:
        return None

    categories = list(scores.keys())
    values = [scores[c] * 100 if scores[c] is not None else 0 for c in categories]

    if len([v for v in values if v > 0]) < 1:
        logger.info("Threat matrix: no threat scores extracted")
        return None

    # Color coding for threat levels
    colors = []
    for value in values:
        if value >= 70:
            colors.append(FBI_COLORS['danger'])
        elif value >= 50:
            colors.append(FBI_COLORS['warning'])
        else:
            colors.append(FBI_COLORS['primary'])

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.0f}' for v in values],
        textposition='inside',
        textfont={'color': FBI_COLORS['text'], 'size': 12},
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        xaxis={
            'range': [0, 100],
            'title': 'Risk Level (0-100)',
            'gridcolor': FBI_COLORS['panel'],
            'tickfont': {'color': FBI_COLORS['text_secondary']},
        },
        yaxis={
            'tickfont': {'color': FBI_COLORS['text'], 'size': 11},
        },
        height=220,
        margin=dict(l=130, r=30, t=40, b=50),
        showlegend=False,
        title={
            'text': 'Threat Assessment Matrix',
            'font': {'size': 14, 'color': FBI_COLORS['text']},
            'x': 0.5,
        }
    )

    return fig


def create_all_visualizations(result: Dict) -> Dict[str, Any]:
    """
    Create all available visualizations from analysis result.

    Args:
        result: Complete profiling result dictionary

    Returns:
        Dictionary of {chart_name: plotly_figure_or_None}
    """
    visualizations = {
        'confidence_gauge': None,
        'confidence_bars': None,
        'big_five_radar': None,
        'dark_triad_bars': None,
        'threat_matrix': None,
    }

    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available - returning empty visualizations")
        return visualizations

    # Get relevant data
    confidence_data = result.get('confidence', {})
    analyses = result.get('analyses', {})
    fbi_text = analyses.get('fbi_behavioral_synthesis', '')

    # Create confidence visualizations
    try:
        visualizations['confidence_gauge'] = create_confidence_gauge(confidence_data)
    except Exception as e:
        logger.warning(f"Failed to create confidence gauge: {e}")

    try:
        visualizations['confidence_bars'] = create_component_confidence_bars(confidence_data)
    except Exception as e:
        logger.warning(f"Failed to create confidence bars: {e}")

    # Create personality visualizations from FBI synthesis
    try:
        visualizations['big_five_radar'] = create_big_five_radar(fbi_text)
    except Exception as e:
        logger.warning(f"Failed to create Big Five radar: {e}")

    try:
        visualizations['dark_triad_bars'] = create_dark_triad_bars(fbi_text)
    except Exception as e:
        logger.warning(f"Failed to create Dark Triad bars: {e}")

    try:
        visualizations['threat_matrix'] = create_threat_matrix(fbi_text)
    except Exception as e:
        logger.warning(f"Failed to create threat matrix: {e}")

    # Log what was successfully created
    created = [k for k, v in visualizations.items() if v is not None]
    if created:
        logger.info(f"Created visualizations: {', '.join(created)}")
    else:
        logger.info("No visualizations could be created from available data")

    return visualizations


def check_plotly_available() -> bool:
    """Check if Plotly is available for visualizations."""
    return PLOTLY_AVAILABLE
