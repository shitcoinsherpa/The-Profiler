"""
Signal Collapsing Layer - Deduplicates and aggregates timestamped behavioral events.

This middleware processes all sub-analysis outputs before synthesis to:
1. Extract timestamped events from each analysis
2. Group events by timestamp (within 2-second windows)
3. Create "High Confidence Events" when multiple analyses flag the same moment
4. Reduce token count and improve synthesis quality
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BehavioralEvent:
    """A single behavioral event extracted from analysis."""
    timestamp_seconds: float
    timestamp_str: str
    source: str  # Which analysis produced this
    description: str
    event_type: str  # gesture, expression, vocal, etc.


@dataclass
class CollapsedEvent:
    """Multiple analyses pointing to same moment."""
    timestamp_seconds: float
    timestamp_str: str
    sources: List[str]
    descriptions: List[str]
    confidence: str  # LOW, MEDIUM, HIGH, CRITICAL
    summary: str


def parse_timestamp(ts_str: str) -> float:
    """Convert timestamp string to seconds.

    Handles formats:
    - 0:32, 1:05, 2:30 (MM:SS)
    - 0:32-0:35 (ranges - uses start)
    - ~0:32 (approximate)
    - 32s, 1m32s
    """
    ts_str = ts_str.strip().lstrip('~').lstrip('@')

    # Handle ranges - take start
    if '-' in ts_str:
        ts_str = ts_str.split('-')[0].strip()

    # Handle MM:SS format
    if ':' in ts_str:
        parts = ts_str.split(':')
        try:
            if len(parts) == 2:
                mins, secs = parts
                return float(mins) * 60 + float(secs.rstrip('s'))
            elif len(parts) == 3:
                hours, mins, secs = parts
                return float(hours) * 3600 + float(mins) * 60 + float(secs.rstrip('s'))
        except ValueError:
            return -1

    # Handle Xs or XmYs format
    if 's' in ts_str.lower():
        ts_str = ts_str.lower().replace('s', '')
        if 'm' in ts_str:
            parts = ts_str.split('m')
            try:
                return float(parts[0]) * 60 + float(parts[1] if parts[1] else 0)
            except ValueError:
                return -1
        try:
            return float(ts_str)
        except ValueError:
            return -1

    return -1


def extract_events(analysis_text: str, source_name: str) -> List[BehavioralEvent]:
    """Extract timestamped events from analysis text."""
    events = []

    # Pattern to match timestamps with context
    # Matches: 0:32, 1:05, ~0:45, @1:20, 0:30-0:35, etc.
    timestamp_pattern = r'[\~@]?(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?|\d+[ms]?\d*s?)'

    lines = analysis_text.split('\n')
    for line in lines:
        # Find timestamps in line
        matches = re.finditer(timestamp_pattern, line)
        for match in matches:
            ts_str = match.group(0)
            ts_seconds = parse_timestamp(ts_str)

            if ts_seconds >= 0:
                # Extract context around timestamp
                # Get the sentence or line containing the timestamp
                description = line.strip()

                # Determine event type
                event_type = classify_event(description)

                events.append(BehavioralEvent(
                    timestamp_seconds=ts_seconds,
                    timestamp_str=ts_str,
                    source=source_name,
                    description=description,
                    event_type=event_type
                ))

    return events


def classify_event(description: str) -> str:
    """Classify event type based on description content."""
    desc_lower = description.lower()

    if any(w in desc_lower for w in ['blink', 'eye', 'gaze', 'pupil']):
        return 'ocular'
    elif any(w in desc_lower for w in ['hand', 'gesture', 'arm', 'finger', 'touch']):
        return 'gesture'
    elif any(w in desc_lower for w in ['smile', 'frown', 'expression', 'micro', 'facs', 'au']):
        return 'expression'
    elif any(w in desc_lower for w in ['voice', 'pitch', 'pause', 'speech', 'vocal', 'tone']):
        return 'vocal'
    elif any(w in desc_lower for w in ['posture', 'lean', 'body', 'shoulder', 'head']):
        return 'posture'
    elif any(w in desc_lower for w in ['stress', 'anxiety', 'tension', 'load']):
        return 'stress_marker'
    elif any(w in desc_lower for w in ['decept', 'lie', 'fabricat', 'incongruent']):
        return 'deception_indicator'
    else:
        return 'behavioral'


def collapse_events(
    events: List[BehavioralEvent],
    time_window: float = 2.0
) -> List[CollapsedEvent]:
    """
    Collapse events within time windows into aggregated events.

    Args:
        events: List of extracted events
        time_window: Seconds within which events are considered same moment

    Returns:
        List of collapsed events with confidence levels
    """
    if not events:
        return []

    # Sort by timestamp
    events = sorted(events, key=lambda e: e.timestamp_seconds)

    # Group events by time window
    groups: List[List[BehavioralEvent]] = []
    current_group = [events[0]]

    for event in events[1:]:
        if event.timestamp_seconds - current_group[0].timestamp_seconds <= time_window:
            current_group.append(event)
        else:
            groups.append(current_group)
            current_group = [event]
    groups.append(current_group)

    # Create collapsed events
    collapsed = []
    for group in groups:
        sources = list(set(e.source for e in group))
        num_sources = len(sources)

        # Determine confidence based on number of independent sources
        if num_sources >= 5:
            confidence = "CRITICAL"
        elif num_sources >= 3:
            confidence = "HIGH"
        elif num_sources >= 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Create summary
        avg_timestamp = sum(e.timestamp_seconds for e in group) / len(group)
        mins = int(avg_timestamp // 60)
        secs = int(avg_timestamp % 60)
        ts_str = f"{mins}:{secs:02d}"

        # Group descriptions by type
        by_type = defaultdict(list)
        for e in group:
            by_type[e.event_type].append(e.description)

        # Create concise summary
        summary_parts = []
        for etype, descs in by_type.items():
            # Take first description for each type
            summary_parts.append(f"[{etype.upper()}] {descs[0][:100]}")

        summary = f"[{confidence}] {ts_str} - {num_sources} analyses flagged this moment:\n" + \
                  "\n".join(summary_parts[:5])  # Limit to 5 types

        collapsed.append(CollapsedEvent(
            timestamp_seconds=avg_timestamp,
            timestamp_str=ts_str,
            sources=sources,
            descriptions=[e.description for e in group],
            confidence=confidence,
            summary=summary
        ))

    return collapsed


def generate_collapsed_summary(collapsed_events: List[CollapsedEvent]) -> str:
    """Generate a summary section for synthesis input."""
    if not collapsed_events:
        return "No significant timestamped events detected across analyses."

    # Sort by confidence then timestamp
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_events = sorted(
        collapsed_events,
        key=lambda e: (priority_order.get(e.confidence, 4), e.timestamp_seconds)
    )

    lines = [
        "═══════════════════════════════════════════════════════════════",
        "SIGNAL COLLAPSED EVENT SUMMARY",
        "(Events deduplicated across all analyses)",
        "═══════════════════════════════════════════════════════════════",
        ""
    ]

    # Group by confidence
    critical = [e for e in sorted_events if e.confidence == "CRITICAL"]
    high = [e for e in sorted_events if e.confidence == "HIGH"]
    medium = [e for e in sorted_events if e.confidence == "MEDIUM"]

    if critical:
        lines.append("*** CRITICAL HOTSPOTS (5+ analyses) ***")
        for e in critical:
            lines.append(f"  [{e.timestamp_str}] {len(e.sources)} sources: {', '.join(e.sources)}")
        lines.append("")

    if high:
        lines.append("** HIGH CONFIDENCE EVENTS (3-4 analyses) **")
        for e in high:
            lines.append(f"  [{e.timestamp_str}] {len(e.sources)} sources: {', '.join(e.sources)}")
        lines.append("")

    if medium:
        lines.append("* MEDIUM CONFIDENCE EVENTS (2 analyses) *")
        for e in medium[:10]:  # Limit medium events
            lines.append(f"  [{e.timestamp_str}] {len(e.sources)} sources: {', '.join(e.sources)}")
        lines.append("")

    lines.append("═══════════════════════════════════════════════════════════════")

    return "\n".join(lines)


def collapse_analysis_outputs(
    stage_results: Dict[str, str],
    time_window: float = 2.0
) -> Tuple[str, List[CollapsedEvent]]:
    """
    Main entry point: Process all stage results and return collapsed summary.

    Args:
        stage_results: Dict of {analysis_name: analysis_text}
        time_window: Seconds for event grouping

    Returns:
        Tuple of (collapsed_summary_text, list_of_collapsed_events)
    """
    all_events = []

    for source_name, analysis_text in stage_results.items():
        events = extract_events(analysis_text, source_name)
        all_events.extend(events)

    collapsed = collapse_events(all_events, time_window)
    summary = generate_collapsed_summary(collapsed)

    return summary, collapsed
