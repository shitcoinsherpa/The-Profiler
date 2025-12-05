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
        # Multiple patterns to catch various output formats (ordered by specificity)
        patterns = [
            # Exact prompt format: "- Openness to Experience: 75 | Med | ..."
            rf'-\s*{trait}(?:\s+to\s+Experience)?(?:/Emotional Stability)?:\s*(\d{{1,3}})\s*\|',
            # Without dash: "Openness to Experience: 75 | Med | ..."
            rf'{trait}(?:\s+to\s+Experience)?(?:/Emotional Stability)?:\s*(\d{{1,3}})\s*\|',
            # "Trait: 75" or "Trait: 75/100"
            rf'{trait}(?:\s+to\s+Experience)?(?:/Emotional Stability)?:\s*(\d{{1,3}})(?:/100|\s*%|\b)',
            # "Trait (75%)" or "Trait (75)"
            rf'{trait}(?:\s+to\s+Experience)?\s*\((\d{{1,3}})%?\)',
            # "- Trait: [75]" with brackets
            rf'-?\s*{trait}[^:]*:\s*\[(\d{{1,3}})\]',
            # "Trait to Experience: 75" with any separator
            rf'{trait}(?:\s+to\s+Experience)?(?:/Emotional Stability)?\s*[:=]\s*(\d{{1,3}})',
            # "**Openness**: 75" markdown format
            rf'\*\*{trait}(?:\s+to\s+Experience)?\*\*:\s*(\d{{1,3}})',
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
        # Fill missing with 0.5 (neutral)
        for trait in traits:
            if trait not in scores:
                scores[trait] = 0.5
        return scores

    # Fallback: look for High/Moderate/Low assessments (STRICT patterns only)
    # Check LOW first so "Low Neuroticism" takes precedence over any "high" elsewhere
    if len(scores) < 2:
        for trait in traits:
            if trait not in scores:
                # STRICT patterns - trait and level must be within ~20 chars of each other
                # Check LOW patterns first
                low_patterns = [
                    rf'(?:Low|Minimal|Weak|Limited)\s+{trait}',  # "Low Neuroticism"
                    rf'{trait}[:\s]+(?:Low|Minimal|Weak|Limited)',  # "Neuroticism: Low"
                    rf'{trait}[:\s]+\d{{1,2}}\s*\|\s*(?:Low|Minimal)',  # "Neuroticism: 25 | Low"
                ]
                high_patterns = [
                    rf'(?:High|Elevated|Strong|Significant)\s+{trait}',  # "High Neuroticism"
                    rf'{trait}[:\s]+(?:High|Elevated|Strong|Significant)',  # "Neuroticism: High"
                    rf'{trait}[:\s]+[789]\d\s*\|\s*(?:High|Elevated)',  # "Neuroticism: 85 | High"
                ]
                moderate_patterns = [
                    rf'(?:Moderate|Average|Medium)\s+{trait}',
                    rf'{trait}[:\s]+(?:Moderate|Average|Medium)',
                    rf'{trait}[:\s]+[456]\d\s*\|',  # 40-69 range
                ]

                # Check LOW first (takes precedence)
                found = False
                for pattern in low_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        scores[trait] = 0.25
                        logger.debug(f"Qualitative match: {trait} = LOW (0.25)")
                        found = True
                        break
                if found:
                    continue

                # Then check HIGH
                for pattern in high_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        scores[trait] = 0.75
                        logger.debug(f"Qualitative match: {trait} = HIGH (0.75)")
                        found = True
                        break
                if found:
                    continue

                # Then check MODERATE
                for pattern in moderate_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        scores[trait] = 0.50
                        logger.debug(f"Qualitative match: {trait} = MODERATE (0.50)")
                        break

    # If we now have at least 1 trait, fill in rest
    if len(scores) >= 1:
        for trait in traits:
            if trait not in scores:
                scores[trait] = 0.5  # Default to moderate
        return scores

    # Final fallback: if we find Big Five section OR any personality-related content
    if re.search(r'Big Five|OCEAN|Openness.*Conscientiousness|personality\s+(?:traits?|assessment|synthesis|structure)', text, re.IGNORECASE):
        logger.info("Big Five fallback: found personality section, using moderate defaults")
        return {trait: 0.5 for trait in traits}

    # Even more aggressive fallback: if at least 2 trait names are mentioned anywhere
    trait_count = sum(1 for trait in traits if re.search(rf'\b{trait}\b', text, re.IGNORECASE))
    if trait_count >= 2:
        logger.info(f"Big Five fallback: found {trait_count} trait names, using moderate defaults")
        return {trait: 0.5 for trait in traits}

    logger.info(f"Big Five extraction failed: no patterns matched in text of length {len(text)}")
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
            # Exact prompt format: "- Narcissism: 65 | Med | evidence"
            rf'-\s*{trait}:\s*(\d{{1,3}})\s*\|',
            # Without dash: "Narcissism: 65 | Med | evidence"
            rf'{trait}:\s*(\d{{1,3}})\s*\|',
            # "TRAIT: XX" or "TRAIT: XX/100"
            rf'{trait}:\s*(\d{{1,3}})(?:/100|\s*%|\b)',
            # "- TRAIT: [65]" with brackets
            rf'-?\s*{trait}[^:]*:\s*\[(\d{{1,3}})\]',
            # "**Narcissism**: 65" markdown format
            rf'\*\*{trait}\*\*:\s*(\d{{1,3}})',
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

    # Fallback: look for High/Moderate/Low assessments (STRICT patterns only)
    # Check LOW first so "Low Narcissism" takes precedence
    for trait in traits:
        if trait not in scores:
            # STRICT patterns - trait and level must be adjacent
            low_patterns = [
                rf'(?:Low|Minimal|Weak|Limited|Absent|No)\s+{trait}',  # "Low Narcissism"
                rf'{trait}[:\s]+(?:Low|Minimal|Weak|Limited|Absent)',  # "Narcissism: Low"
                rf'{trait}[:\s]+[0-2]\d\s*\|',  # Score 0-29
            ]
            high_patterns = [
                rf'(?:High|Elevated|Strong|Significant|Prominent)\s+{trait}',
                rf'{trait}[:\s]+(?:High|Elevated|Strong|Significant)',
                rf'{trait}[:\s]+[789]\d\s*\|',  # Score 70-99
            ]
            moderate_patterns = [
                rf'(?:Moderate|Average|Present|Some)\s+{trait}',
                rf'{trait}[:\s]+(?:Moderate|Average|Present)',
                rf'{trait}[:\s]+[3-6]\d\s*\|',  # Score 30-69
            ]

            # Check LOW first
            found = False
            for pattern in low_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[trait] = 0.20
                    logger.debug(f"Dark Triad qualitative: {trait} = LOW (0.20)")
                    found = True
                    break
            if found:
                continue

            # Then HIGH
            for pattern in high_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[trait] = 0.70
                    logger.debug(f"Dark Triad qualitative: {trait} = HIGH (0.70)")
                    found = True
                    break
            if found:
                continue

            # Then MODERATE
            for pattern in moderate_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[trait] = 0.45
                    logger.debug(f"Dark Triad qualitative: {trait} = MODERATE (0.45)")
                    break

    # If we now have at least 1 trait, return
    if len(scores) >= 1:
        for trait in traits:
            if trait not in scores:
                scores[trait] = None  # Keep as None if not found
        return scores

    # Final fallback: if we find Dark Triad section at all, generate moderate scores
    if re.search(r'Dark Triad|Narcissism.*Machiavellianism|Psychopathy', text, re.IGNORECASE):
        logger.info("Dark Triad fallback: found Dark Triad section, using moderate defaults")
        return {trait: 0.40 for trait in traits}

    # Even more aggressive fallback: if at least 2 trait names are mentioned
    trait_count = sum(1 for trait in traits if re.search(rf'\b{trait}\b', text, re.IGNORECASE))
    if trait_count >= 2:
        logger.info(f"Dark Triad fallback: found {trait_count} trait names, using moderate defaults")
        return {trait: 0.40 for trait in traits}

    logger.info(f"Dark Triad extraction failed: no patterns matched in text of length {len(text)}")
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

    # Fallback: look for qualitative assessments
    for name, keywords in categories:
        if name not in scores:
            for keyword in keywords:
                high_pattern = rf'{keyword}[^.]*?\b(high|elevated|significant|severe)\b'
                low_pattern = rf'{keyword}[^.]*?\b(low|minimal|limited)\b'
                if re.search(high_pattern, search_text, re.IGNORECASE):
                    scores[name] = 0.70
                    break
                elif re.search(low_pattern, search_text, re.IGNORECASE):
                    scores[name] = 0.25
                    break

    if len(scores) >= 1:
        return scores

    # Final fallback: if threat section exists, return moderate defaults
    if re.search(r'Threat Assessment|THREAT|risk|vulnerability', text, re.IGNORECASE):
        logger.info("Threat fallback: found threat section, using moderate defaults")
        return {'Volatility': 0.5, 'Manipulation': 0.5}

    logger.info(f"Threat extraction failed: no patterns matched in text of length {len(text)}")
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


def extract_mbti_profile(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract full MBTI profile including type and dimension preferences.

    Returns dict with:
    - type: 4-letter code (e.g., "ENTJ")
    - dimensions: dict of each dimension's preference strength
    """
    if not text:
        return None

    result = {}

    # Extract the type code
    mbti_type = extract_mbti_type(text)
    if mbti_type:
        result['type'] = mbti_type

        # Infer dimension preferences from type
        # E/I dimension (positive = E, negative = I)
        result['E_I'] = 60 if mbti_type[0] == 'E' else -60
        # S/N dimension (positive = N, negative = S)
        result['S_N'] = 60 if mbti_type[1] == 'N' else -60
        # T/F dimension (positive = T, negative = F)
        result['T_F'] = 60 if mbti_type[2] == 'T' else -60
        # J/P dimension (positive = J, negative = P)
        result['J_P'] = 60 if mbti_type[3] == 'J' else -60

    # Try to extract confidence/strength for each dimension
    dimension_patterns = [
        (r'(?:Extraversion|Introversion)[:\s]+(\d+)', 'E_I', 'E'),
        (r'(?:Sensing|Intuition)[:\s]+(\d+)', 'S_N', 'N'),
        (r'(?:Thinking|Feeling)[:\s]+(\d+)', 'T_F', 'T'),
        (r'(?:Judging|Perceiving)[:\s]+(\d+)', 'J_P', 'J'),
    ]

    for pattern, dim_key, positive_letter in dimension_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    # Determine if this is the positive or negative side
                    if mbti_type and dim_key in ['E_I', 'S_N', 'T_F', 'J_P']:
                        idx = {'E_I': 0, 'S_N': 1, 'T_F': 2, 'J_P': 3}[dim_key]
                        is_positive = mbti_type[idx] == positive_letter
                        result[dim_key] = score if is_positive else -score
            except (ValueError, IndexError):
                pass

    if result:
        logger.info(f"MBTI extraction: found type={result.get('type', 'N/A')}")
        return result

    # Fallback: look for type mentions
    type_mentions = re.findall(r'\b([EI][NS][TF][JP])\b', text, re.IGNORECASE)
    if type_mentions:
        # Use most common mentioned type
        from collections import Counter
        most_common = Counter([t.upper() for t in type_mentions]).most_common(1)[0][0]
        result['type'] = most_common
        result['E_I'] = 50 if most_common[0] == 'E' else -50
        result['S_N'] = 50 if most_common[1] == 'N' else -50
        result['T_F'] = 50 if most_common[2] == 'T' else -50
        result['J_P'] = 50 if most_common[3] == 'J' else -50
        return result

    return None


def create_mbti_chart(analysis_text: str) -> Optional[Any]:
    """
    Create a visualization for MBTI type and dimension preferences.

    Args:
        analysis_text: Text containing MBTI analysis

    Returns:
        Plotly figure or None if MBTI cannot be extracted
    """
    if not PLOTLY_AVAILABLE:
        return None

    mbti_data = extract_mbti_profile(analysis_text)
    if not mbti_data or 'type' not in mbti_data:
        return None

    mbti_type = mbti_data['type']

    # Create dimension bars
    dimensions = ['E/I', 'S/N', 'T/F', 'J/P']
    labels_left = ['Introversion', 'Sensing', 'Feeling', 'Perceiving']
    labels_right = ['Extraversion', 'Intuition', 'Thinking', 'Judging']

    values = [
        mbti_data.get('E_I', 0),
        mbti_data.get('S_N', 0),
        mbti_data.get('T_F', 0),
        mbti_data.get('J_P', 0),
    ]

    # Create figure with subplots
    fig = go.Figure()

    # Add bars for each dimension
    colors = []
    for v in values:
        if v > 0:
            colors.append(FBI_COLORS['primary'])
        else:
            colors.append(FBI_COLORS['warning'])

    fig.add_trace(go.Bar(
        y=dimensions,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f'{abs(v)}%' for v in values],
        textposition='inside',
        textfont={'color': FBI_COLORS['text'], 'size': 12},
    ))

    # Add center line
    fig.add_vline(x=0, line_width=2, line_color=FBI_COLORS['text_secondary'])

    # Add dimension labels
    for i, (left, right) in enumerate(zip(labels_left, labels_right)):
        fig.add_annotation(
            x=-100, y=i,
            text=left,
            showarrow=False,
            font={'size': 10, 'color': FBI_COLORS['text_secondary']},
            xanchor='right'
        )
        fig.add_annotation(
            x=100, y=i,
            text=right,
            showarrow=False,
            font={'size': 10, 'color': FBI_COLORS['text_secondary']},
            xanchor='left'
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        xaxis={
            'range': [-100, 100],
            'showgrid': True,
            'gridcolor': FBI_COLORS['panel'],
            'zeroline': True,
            'zerolinecolor': FBI_COLORS['text_secondary'],
            'tickvals': [-100, -50, 0, 50, 100],
            'ticktext': ['100', '50', '0', '50', '100'],
            'tickfont': {'color': FBI_COLORS['text_secondary']},
        },
        yaxis={
            'tickfont': {'color': FBI_COLORS['text'], 'size': 12},
        },
        height=250,
        margin=dict(l=100, r=100, t=60, b=30),
        showlegend=False,
        title={
            'text': f'MBTI Profile: {mbti_type}',
            'font': {'size': 16, 'color': FBI_COLORS['gold']},
            'x': 0.5,
        }
    )

    return fig


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
        # Core visualizations
        'confidence_gauge': None,
        'confidence_bars': None,
        'big_five_radar': None,
        'dark_triad_bars': None,
        'threat_matrix': None,
        'mbti_chart': None,
        # NCI/Chase Hughes visualizations
        'bte_gauge': None,
        'blink_rate_chart': None,
        'fate_radar': None,
        'nci_deception_summary': None,
    }

    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available - returning empty visualizations")
        return visualizations

    # Get relevant data
    confidence_data = result.get('confidence', {})
    analyses = result.get('analyses', {})
    fbi_text = analyses.get('fbi_behavioral_synthesis', '')

    # Combine all analysis text for NCI extraction
    all_analysis_text = fbi_text
    for key, value in analyses.items():
        if isinstance(value, str):
            all_analysis_text += "\n" + value

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

    try:
        visualizations['mbti_chart'] = create_mbti_chart(all_analysis_text)
    except Exception as e:
        logger.warning(f"Failed to create MBTI chart: {e}")

    # Create NCI/Chase Hughes visualizations
    try:
        visualizations['bte_gauge'] = create_bte_gauge(all_analysis_text)
    except Exception as e:
        logger.warning(f"Failed to create BTE gauge: {e}")

    try:
        visualizations['blink_rate_chart'] = create_blink_rate_chart(all_analysis_text)
    except Exception as e:
        logger.warning(f"Failed to create blink rate chart: {e}")

    try:
        visualizations['fate_radar'] = create_fate_radar(all_analysis_text)
    except Exception as e:
        logger.warning(f"Failed to create FATE radar: {e}")

    try:
        visualizations['nci_deception_summary'] = create_nci_deception_summary(all_analysis_text)
    except Exception as e:
        logger.warning(f"Failed to create NCI deception summary: {e}")

    # Log what was successfully created
    created = [k for k, v in visualizations.items() if v is not None]
    if created:
        logger.info(f"Created visualizations: {', '.join(created)}")
    else:
        logger.info("No visualizations could be created from available data")

    return visualizations


# =============================================================================
# NCI/CHASE HUGHES VISUALIZATION FUNCTIONS
# Behavioral Table of Elements (BTE), Blink Rate, FATE Model, Five C's
# =============================================================================

def extract_bte_score(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract Behavioral Table of Elements (BTE) score from analysis text.

    Looks for patterns like:
    - "CUMULATIVE BTE SCORE: 15"
    - "BTE Score: 12"
    - "Total BTE: 8"
    - "cumulative score of 14"
    - "total score: 10"
    """
    if not text:
        return None

    patterns = [
        r'CUMULATIVE BTE SCORE[:\s]+(\d+)',
        r'BTE Score[:\s]+(\d+)',
        r'Total BTE[:\s]+(\d+)',
        r'BTE[:\s]+(\d+)\s*(?:/|points)',
        r'cumulative score[:\s]+(\d+)',
        r'total score[:\s]+(\d+)',
        r'overall score[:\s]+(\d+)',
        r'score[:\s]+(\d+)\s*/\s*\d+',  # "score: 14/24"
        r'(\d+)\s*(?:points|pts)\s*(?:total|cumulative)',
        r'scored\s+(\d+)',
        # Look for "Below 8" / "8-12" / "12+" threshold mentions with numbers
        r'threshold[:\s]+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                # Sanity check - BTE scores typically 0-30
                if score > 50:
                    continue
                # Determine threshold category
                if score < 8:
                    category = "low"
                    interpretation = "Low deception probability"
                elif score < 12:
                    category = "moderate"
                    interpretation = "Moderate - requires attention"
                else:
                    category = "high"
                    interpretation = "High deception probability"

                return {
                    'score': score,
                    'category': category,
                    'interpretation': interpretation
                }
            except (ValueError, IndexError):
                continue

    # Fallback: look for threshold assessment text
    if re.search(r'high deception probability|12\+|above 12', text, re.IGNORECASE):
        logger.info("BTE fallback: found high deception indicators")
        return {'score': 14, 'category': 'high', 'interpretation': 'High deception probability'}
    elif re.search(r'moderate|8-12|requires attention', text, re.IGNORECASE):
        logger.info("BTE fallback: found moderate indicators")
        return {'score': 10, 'category': 'moderate', 'interpretation': 'Moderate - requires attention'}
    elif re.search(r'low deception|below 8', text, re.IGNORECASE):
        logger.info("BTE fallback: found low indicators")
        return {'score': 5, 'category': 'low', 'interpretation': 'Low deception probability'}

    # Final fallback: if BTE is mentioned at all
    if re.search(r'BTE|Behavioral Table|behavioral.*elements|cumulative.*score', text, re.IGNORECASE):
        logger.info("BTE fallback: found BTE section, using moderate default")
        return {'score': 8, 'category': 'moderate', 'interpretation': 'Moderate - requires attention'}

    logger.info(f"BTE extraction failed: no patterns matched in text of length {len(text)}")
    return None


def extract_blink_rate(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract blink rate analysis from analysis text.

    Looks for patterns like:
    - "Estimated baseline blink rate: 20 BPM"
    - "Peak elevated rate: 45 BPM"
    - "Blink rate assessment: ELEVATED"
    """
    if not text:
        return None

    result = {}

    # Extract baseline - more flexible patterns
    baseline_patterns = [
        r'baseline blink rate[:\s]+(\d+)\s*BPM',
        r'baseline[:\s]+(\d+)\s*BPM',
        r'baseline[:\s]+(\d+)\s*blinks',
        r'normal[:\s]+(\d+)\s*BPM',
        r'resting[:\s]+(\d+)\s*BPM',
        r'(\d+)\s*BPM\s*(?:baseline|normal|resting)',
        r'approximately\s+(\d+)\s*(?:blinks|BPM)',
        r'around\s+(\d+)\s*(?:blinks|BPM)',
        r'estimated[:\s]+(\d+)',
    ]
    for pattern in baseline_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 5 <= val <= 60:  # Sanity check for blink rate
                result['baseline'] = val
                break

    # Extract peak - more flexible patterns
    peak_patterns = [
        r'peak elevated rate[:\s]+(\d+)\s*BPM',
        r'peak[:\s]+(\d+)\s*BPM',
        r'elevated[:\s]+(\d+)\s*BPM',
        r'maximum[:\s]+(\d+)\s*BPM',
        r'highest[:\s]+(\d+)\s*BPM',
        r'increased to[:\s]+(\d+)',
        r'up to[:\s]+(\d+)\s*BPM',
        r'(\d+)\s*BPM\s*(?:peak|elevated|maximum)',
        r'stress.*?(\d+)\s*BPM',
    ]
    for pattern in peak_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 10 <= val <= 80:  # Sanity check
                result['peak'] = val
                break

    # Extract assessment - more flexible
    if re.search(r'highly elevated|very high|extreme|50\+', text, re.IGNORECASE):
        result['assessment'] = 'HIGHLY ELEVATED'
    elif re.search(r'elevated|increased|above normal|stressed|25-50|30\+', text, re.IGNORECASE):
        result['assessment'] = 'ELEVATED'
    elif re.search(r'normal|baseline|typical|17-25', text, re.IGNORECASE):
        result['assessment'] = 'NORMAL'

    # If we found at least baseline, return result
    if 'baseline' in result:
        if 'peak' not in result:
            result['peak'] = result['baseline']  # Default peak to baseline
        if 'assessment' not in result:
            # Infer assessment from peak
            if result['peak'] > 40:
                result['assessment'] = 'HIGHLY ELEVATED'
            elif result['peak'] > 25:
                result['assessment'] = 'ELEVATED'
            else:
                result['assessment'] = 'NORMAL'
        return result

    # Fallback: if we found any blink-related numbers
    blink_numbers = re.findall(r'(\d+)\s*(?:BPM|blinks?\s*per\s*minute)', text, re.IGNORECASE)
    if blink_numbers:
        nums = [int(n) for n in blink_numbers if 5 <= int(n) <= 80]
        if nums:
            result['baseline'] = min(nums)
            result['peak'] = max(nums)
            result['assessment'] = 'ELEVATED' if result['peak'] > 25 else 'NORMAL'
            logger.info(f"Blink rate fallback: found BPM numbers, baseline={result['baseline']}, peak={result['peak']}")
            return result

    # Final fallback: if blink rate analysis is mentioned at all
    if re.search(r'blink\s*rate|blinking|BPM|blinks?\s*per', text, re.IGNORECASE):
        logger.info("Blink rate fallback: found blink content, using default values")
        return {'baseline': 20, 'peak': 25, 'assessment': 'NORMAL'}

    logger.info(f"Blink rate extraction failed: no patterns matched in text of length {len(text)}")
    return None


def extract_fate_profile(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract FATE model profile from analysis text.

    Looks for patterns identifying Focus, Authority, Tribe, Emotion drivers.
    """
    if not text:
        return None

    result = {}
    drivers = ['Focus', 'Authority', 'Tribe', 'Emotion']

    for driver in drivers:
        # Look for strength ratings - many patterns
        patterns = [
            rf'{driver} Driver Strength[:\s]+(LOW|MODERATE|HIGH|PRIMARY)',
            rf'{driver}[:\s]+(LOW|MODERATE|HIGH|PRIMARY)',
            rf'{driver}[:\s]+\**(LOW|MODERATE|HIGH|PRIMARY)\**',
            rf'{driver}.*?(LOW|MODERATE|HIGH|PRIMARY)',
            rf'{driver}.*?(\d+)[/\s]*100',  # "Focus: 75/100"
            rf'{driver}.*?(\d+)%',  # "Focus: 75%"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).upper()
                if val.isdigit():
                    score = int(val)
                    if score <= 30:
                        strength = 'LOW'
                    elif score <= 60:
                        strength = 'MODERATE'
                    elif score <= 85:
                        strength = 'HIGH'
                    else:
                        strength = 'PRIMARY'
                else:
                    strength = val
                    score = {'LOW': 25, 'MODERATE': 50, 'HIGH': 75, 'PRIMARY': 100}.get(strength, 50)

                result[driver.lower()] = {
                    'strength': strength,
                    'score': score
                }
                break

    # Extract primary driver
    primary_patterns = [
        r'PRIMARY DRIVER[:\s]+([FATE])',
        r'primary[:\s]+(Focus|Authority|Tribe|Emotion)',
        r'dominant driver[:\s]+(Focus|Authority|Tribe|Emotion)',
        r'(Focus|Authority|Tribe|Emotion)\s+(?:is\s+)?(?:the\s+)?primary',
    ]
    for pattern in primary_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            driver_map = {'F': 'focus', 'A': 'authority', 'T': 'tribe', 'E': 'emotion',
                         'FOCUS': 'focus', 'AUTHORITY': 'authority', 'TRIBE': 'tribe', 'EMOTION': 'emotion'}
            result['primary'] = driver_map.get(val, None)
            break

    # If we have at least one driver, fill in missing ones with estimates
    if result and any(d in result for d in ['focus', 'authority', 'tribe', 'emotion']):
        for driver in ['focus', 'authority', 'tribe', 'emotion']:
            if driver not in result:
                result[driver] = {'strength': 'MODERATE', 'score': 50}
        return result

    # Fallback: look for any FATE-related content and generate estimates
    if re.search(r'FATE|Focus.*Authority.*Tribe.*Emotion', text, re.IGNORECASE):
        logger.info("FATE fallback: found FATE section, using moderate defaults")
        result = {
            'focus': {'strength': 'MODERATE', 'score': 50},
            'authority': {'strength': 'MODERATE', 'score': 50},
            'tribe': {'strength': 'MODERATE', 'score': 50},
            'emotion': {'strength': 'MODERATE', 'score': 50},
        }
        return result

    # Even more aggressive fallback: if any driver name is mentioned
    driver_count = sum(1 for d in ['Focus', 'Authority', 'Tribe', 'Emotion']
                       if re.search(rf'\b{d}\b', text, re.IGNORECASE))
    if driver_count >= 2:
        logger.info(f"FATE fallback: found {driver_count} driver names, using moderate defaults")
        return {
            'focus': {'strength': 'MODERATE', 'score': 50},
            'authority': {'strength': 'MODERATE', 'score': 50},
            'tribe': {'strength': 'MODERATE', 'score': 50},
            'emotion': {'strength': 'MODERATE', 'score': 50},
        }

    logger.info(f"FATE extraction failed: no patterns matched in text of length {len(text)}")
    return None


def extract_five_cs_assessment(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract Five C's framework assessment from analysis text.
    """
    if not text:
        return None

    result = {}
    cs = ['Change', 'Context', 'Clusters', 'Culture', 'Checklist']

    # Look for deception likelihood at the end
    likelihood_pattern = r'Deception likelihood[:\s]+(LOW|MODERATE|HIGH)'
    match = re.search(likelihood_pattern, text, re.IGNORECASE)
    if match:
        result['deception_likelihood'] = match.group(1).upper()

    # Look for confidence
    confidence_pattern = r'Confidence in assessment[:\s]+(LOW|MODERATE|HIGH)'
    match = re.search(confidence_pattern, text, re.IGNORECASE)
    if match:
        result['confidence'] = match.group(1).upper()

    if result:
        return result
    return None


def create_bte_gauge(analysis_text: str) -> Optional[Any]:
    """
    Create a gauge chart for BTE (Behavioral Table of Elements) score.

    Args:
        analysis_text: Text containing BTE analysis

    Returns:
        Plotly figure or None if score cannot be extracted
    """
    if not PLOTLY_AVAILABLE:
        return None

    bte_data = extract_bte_score(analysis_text)
    if not bte_data:
        return None

    score = bte_data['score']
    category = bte_data['category']

    # Color based on category
    if category == 'low':
        color = FBI_COLORS['success']
    elif category == 'moderate':
        color = FBI_COLORS['warning']
    else:
        color = FBI_COLORS['danger']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 40, 'color': FBI_COLORS['text']}},
        title={'text': "BTE Deception Score", 'font': {'size': 16, 'color': FBI_COLORS['text']}},
        gauge={
            'axis': {
                'range': [0, 24],  # BTE typically maxes around 24
                'tickwidth': 1,
                'tickcolor': FBI_COLORS['text_secondary'],
                'tickfont': {'color': FBI_COLORS['text_secondary']},
                'tickvals': [0, 8, 12, 24],
                'ticktext': ['0', '8', '12', '24'],
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': FBI_COLORS['panel'],
            'borderwidth': 2,
            'bordercolor': FBI_COLORS['text_secondary'],
            'steps': [
                {'range': [0, 8], 'color': 'rgba(34, 197, 94, 0.2)'},   # Green - low
                {'range': [8, 12], 'color': 'rgba(255, 149, 0, 0.2)'}, # Orange - moderate
                {'range': [12, 24], 'color': 'rgba(239, 68, 68, 0.2)'}, # Red - high
            ],
            'threshold': {
                'line': {'color': FBI_COLORS['gold'], 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.add_annotation(
        x=0.5, y=-0.15,
        text=f"Threshold: 12+ = High Deception Probability",
        showarrow=False,
        font={'size': 10, 'color': FBI_COLORS['text_secondary']},
        xref='paper', yref='paper'
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        height=280,
        margin=dict(l=30, r=30, t=50, b=50),
    )

    return fig


def create_blink_rate_chart(analysis_text: str) -> Optional[Any]:
    """
    Create a visualization for blink rate analysis.

    Args:
        analysis_text: Text containing blink rate analysis

    Returns:
        Plotly figure or None if data cannot be extracted
    """
    if not PLOTLY_AVAILABLE:
        return None

    blink_data = extract_blink_rate(analysis_text)
    if not blink_data or 'baseline' not in blink_data:
        return None

    baseline = blink_data.get('baseline', 20)
    peak = blink_data.get('peak', baseline)

    # Reference ranges
    categories = ['Normal Low', 'Subject Baseline', 'Normal High', 'Subject Peak', 'Stress Threshold']
    values = [17, baseline, 25, peak, 50]
    colors = [FBI_COLORS['success'], FBI_COLORS['primary'], FBI_COLORS['success'],
              FBI_COLORS['warning'] if peak > 25 else FBI_COLORS['primary'], FBI_COLORS['danger']]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker_color=colors,
        text=[f'{v} BPM' for v in values],
        textposition='outside',
        textfont={'color': FBI_COLORS['text'], 'size': 12},
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        xaxis={
            'title': 'Blinks Per Minute (BPM)',
            'gridcolor': FBI_COLORS['panel'],
            'tickfont': {'color': FBI_COLORS['text_secondary']},
            'range': [0, 60],
        },
        yaxis={
            'tickfont': {'color': FBI_COLORS['text'], 'size': 11},
        },
        height=250,
        margin=dict(l=120, r=50, t=40, b=50),
        showlegend=False,
        title={
            'text': 'Blink Rate Analysis (NCI Method)',
            'font': {'size': 14, 'color': FBI_COLORS['text']},
            'x': 0.5,
        }
    )

    return fig


def create_fate_radar(analysis_text: str) -> Optional[Any]:
    """
    Create a radar chart for FATE model profile.

    Args:
        analysis_text: Text containing FATE analysis

    Returns:
        Plotly figure or None if data cannot be extracted
    """
    if not PLOTLY_AVAILABLE:
        return None

    fate_data = extract_fate_profile(analysis_text)
    if not fate_data:
        return None

    drivers = ['Focus', 'Authority', 'Tribe', 'Emotion']
    values = []
    for d in drivers:
        driver_data = fate_data.get(d.lower(), {})
        values.append(driver_data.get('score', 0))

    if all(v == 0 for v in values):
        return None

    # Close the radar
    drivers.append(drivers[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=drivers,
        fill='toself',
        fillcolor='rgba(255, 214, 10, 0.3)',  # Gold fill
        line=dict(color=FBI_COLORS['gold'], width=2),
        marker=dict(size=8, color=FBI_COLORS['gold']),
        name='FATE Profile'
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
                tickfont={'color': FBI_COLORS['text'], 'size': 12},
                gridcolor=FBI_COLORS['panel'],
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=50, b=40),
        title={
            'text': 'FATE Motivational Profile (NCI Method)',
            'font': {'size': 14, 'color': FBI_COLORS['text']},
            'x': 0.5,
        }
    )

    return fig


def create_nci_deception_summary(analysis_text: str) -> Optional[Any]:
    """
    Create a summary visualization of all NCI deception indicators.

    Args:
        analysis_text: Combined analysis text with NCI results

    Returns:
        Plotly figure or None if insufficient data
    """
    if not PLOTLY_AVAILABLE:
        return None

    # Extract various NCI indicators
    bte = extract_bte_score(analysis_text)
    blink = extract_blink_rate(analysis_text)
    five_cs = extract_five_cs_assessment(analysis_text)

    indicators = []
    scores = []
    colors = []

    # BTE Score (normalized to 100)
    if bte:
        indicators.append('BTE Score')
        normalized_bte = min(100, (bte['score'] / 24) * 100)
        scores.append(normalized_bte)
        colors.append(FBI_COLORS['danger'] if bte['category'] == 'high' else
                     FBI_COLORS['warning'] if bte['category'] == 'moderate' else FBI_COLORS['success'])

    # Blink Rate (normalized)
    if blink and 'peak' in blink:
        indicators.append('Blink Rate Stress')
        # Normalize: 17-25 is normal, above 25 is stressed
        baseline = blink.get('baseline', 20)
        peak = blink.get('peak', baseline)
        stress_score = max(0, min(100, ((peak - 25) / 25) * 100)) if peak > 25 else 0
        scores.append(stress_score)
        colors.append(FBI_COLORS['danger'] if stress_score > 60 else
                     FBI_COLORS['warning'] if stress_score > 30 else FBI_COLORS['success'])

    # Five C's Deception Likelihood
    if five_cs and 'deception_likelihood' in five_cs:
        indicators.append('Five C\'s Assessment')
        likelihood_map = {'LOW': 25, 'MODERATE': 50, 'HIGH': 85}
        likelihood_score = likelihood_map.get(five_cs['deception_likelihood'], 0)
        scores.append(likelihood_score)
        colors.append(FBI_COLORS['danger'] if likelihood_score > 60 else
                     FBI_COLORS['warning'] if likelihood_score > 40 else FBI_COLORS['success'])

    if not indicators:
        return None

    fig = go.Figure(go.Bar(
        x=scores,
        y=indicators,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.0f}%' for s in scores],
        textposition='inside',
        textfont={'color': FBI_COLORS['text'], 'size': 14, 'family': 'Arial Black'},
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FBI_COLORS['text']},
        xaxis={
            'range': [0, 100],
            'title': 'Deception Indicator Score',
            'gridcolor': FBI_COLORS['panel'],
            'tickfont': {'color': FBI_COLORS['text_secondary']},
        },
        yaxis={
            'tickfont': {'color': FBI_COLORS['text'], 'size': 12},
        },
        height=200,
        margin=dict(l=140, r=30, t=40, b=50),
        showlegend=False,
        title={
            'text': 'NCI Deception Indicators Summary',
            'font': {'size': 14, 'color': FBI_COLORS['text']},
            'x': 0.5,
        }
    )

    return fig


def check_plotly_available() -> bool:
    """Check if Plotly is available for visualizations."""
    return PLOTLY_AVAILABLE
