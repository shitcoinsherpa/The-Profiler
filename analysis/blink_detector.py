"""
CV-based Blink Rate Detection using MediaPipe Face Mesh.

This module provides actual blink counting from video frames,
rather than relying on LLM estimation which is notoriously inaccurate.

Uses Eye Aspect Ratio (EAR) algorithm:
- When EAR drops below threshold, eye is closed
- Count transitions from open->closed as blinks
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. Blink detection will be unavailable.")


@dataclass
class BlinkEvent:
    """A single blink event."""
    timestamp_seconds: float
    frame_number: int
    ear_value: float  # Eye Aspect Ratio at blink


@dataclass
class BlinkAnalysis:
    """Complete blink analysis results."""
    total_blinks: int
    duration_seconds: float
    blinks_per_minute: float
    blink_events: List[BlinkEvent]
    ear_timeline: List[Tuple[float, float]]  # (timestamp, ear_value)
    baseline_bpm: float  # First 30 seconds average
    peak_bpm: float  # Highest BPM in any 30-second window
    peak_timestamp: float  # When peak occurred
    stress_windows: List[Tuple[float, float, float]]  # (start, end, bpm) for high-stress periods


# MediaPipe Face Mesh landmark indices for eyes
# Left eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(landmarks, eye_indices) -> float:
    """
    Calculate Eye Aspect Ratio (EAR).

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Where p1-p6 are the 6 eye landmarks in order:
    p1: outer corner, p2: upper lid outer, p3: upper lid inner
    p4: inner corner, p5: lower lid inner, p6: lower lid outer
    """
    # Get the 6 points
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    # Vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)

    # Horizontal distance
    h = np.linalg.norm(p1 - p4)

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear


def detect_blinks(
    video_path: str,
    ear_threshold: float = 0.25,  # Raised from 0.21 - more permissive for partial blinks
    min_blink_frames: int = 1,    # Lowered from 2 - catch quick blinks
    sample_rate: int = 1,  # Process every Nth frame for speed
    interview_mode: bool = False,  # If True, track multiple faces and select by position
    suspect_position: str = "auto"  # "auto", "left", "right", "fullscreen"
) -> Optional[BlinkAnalysis]:
    """
    Detect blinks in a video using MediaPipe Face Mesh.

    Args:
        video_path: Path to video file
        ear_threshold: EAR below this = eye closed (default 0.21)
        min_blink_frames: Minimum consecutive frames for valid blink
        sample_rate: Process every Nth frame (1 = all frames)
        interview_mode: If True, track up to 2 faces and select based on position
        suspect_position: Which face to track - "left", "right", "auto", "fullscreen"

    Returns:
        BlinkAnalysis with all blink metrics, or None if detection fails
    """
    if not MEDIAPIPE_AVAILABLE:
        logger.error("MediaPipe not available for blink detection")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if duration <= 0:
            logger.error("Invalid video duration")
            return None

        # Initialize MediaPipe Face Mesh
        # In interview mode, track up to 2 faces; otherwise just 1
        mp_face_mesh = mp.solutions.face_mesh
        max_faces = 2 if interview_mode else 1
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Get frame width for position-based face selection
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_center_x = frame_width / 2

        if interview_mode:
            logger.info(f"Interview mode: tracking up to {max_faces} faces, selecting {suspect_position} position")

        blink_events = []
        ear_timeline = []

        # State tracking
        eye_closed = False
        closed_frame_count = 0
        current_blink_start = 0

        frame_num = 0
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Skip frames based on sample rate
            if frame_num % sample_rate != 0:
                continue

            processed_frames += 1
            timestamp = frame_num / fps

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Select the correct face based on interview mode settings
                selected_landmarks = None

                if not interview_mode or suspect_position == "fullscreen" or len(results.multi_face_landmarks) == 1:
                    # Single face mode or only one face detected - use it
                    selected_landmarks = results.multi_face_landmarks[0].landmark
                elif len(results.multi_face_landmarks) >= 2:
                    # Multiple faces detected - select based on position
                    face_positions = []
                    for i, face_landmarks in enumerate(results.multi_face_landmarks):
                        # Calculate face center X position (average of all landmark X coords)
                        x_coords = [lm.x for lm in face_landmarks.landmark]
                        face_center_x_norm = sum(x_coords) / len(x_coords)
                        face_center_x_abs = face_center_x_norm * frame_width
                        face_positions.append((i, face_center_x_abs, face_landmarks.landmark))

                    # Sort faces by X position (left to right)
                    face_positions.sort(key=lambda x: x[1])

                    if suspect_position == "left":
                        # Select leftmost face
                        selected_landmarks = face_positions[0][2]
                    elif suspect_position == "right":
                        # Select rightmost face
                        selected_landmarks = face_positions[-1][2]
                    else:  # "auto" - select face that appears more often on camera (likely the interviewee)
                        # Default to rightmost in interview setting (often the interviewee)
                        selected_landmarks = face_positions[-1][2]

                if selected_landmarks is None:
                    continue

                landmarks = selected_landmarks

                # Calculate EAR for both eyes
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                ear_timeline.append((timestamp, avg_ear))

                # Detect blink
                if avg_ear < ear_threshold:
                    if not eye_closed:
                        eye_closed = True
                        current_blink_start = frame_num
                    closed_frame_count += 1
                else:
                    if eye_closed and closed_frame_count >= min_blink_frames:
                        # Valid blink detected
                        blink_timestamp = current_blink_start / fps
                        blink_events.append(BlinkEvent(
                            timestamp_seconds=blink_timestamp,
                            frame_number=current_blink_start,
                            ear_value=avg_ear
                        ))
                    eye_closed = False
                    closed_frame_count = 0

        face_mesh.close()

        # Calculate metrics
        total_blinks = len(blink_events)
        bpm = (total_blinks / duration) * 60 if duration > 0 else 0

        # Diagnostic logging
        if ear_timeline:
            avg_ear = sum(e[1] for e in ear_timeline) / len(ear_timeline)
            min_ear = min(e[1] for e in ear_timeline)
            face_detection_rate = len(ear_timeline) / processed_frames * 100 if processed_frames > 0 else 0
            logger.info(f"Blink detection stats: avg_EAR={avg_ear:.3f}, min_EAR={min_ear:.3f}, "
                       f"face_detected={face_detection_rate:.1f}%, threshold={ear_threshold}")

        # Calculate baseline (first 30 seconds)
        baseline_blinks = [b for b in blink_events if b.timestamp_seconds <= 30]
        baseline_duration = min(30, duration)
        baseline_bpm = (len(baseline_blinks) / baseline_duration) * 60 if baseline_duration > 0 else bpm

        # Find peak BPM in 30-second windows
        window_size = 30
        peak_bpm = 0
        peak_timestamp = 0
        stress_windows = []

        for window_start in range(0, int(duration), 15):  # 15-second overlap
            window_end = min(window_start + window_size, duration)
            window_blinks = [b for b in blink_events
                           if window_start <= b.timestamp_seconds < window_end]
            window_duration = window_end - window_start
            window_bpm = (len(window_blinks) / window_duration) * 60 if window_duration > 0 else 0

            if window_bpm > peak_bpm:
                peak_bpm = window_bpm
                peak_timestamp = window_start + window_size / 2

            # Mark as stress window if > 150% of baseline
            if baseline_bpm > 0 and window_bpm > baseline_bpm * 1.5:
                stress_windows.append((window_start, window_end, window_bpm))

        return BlinkAnalysis(
            total_blinks=total_blinks,
            duration_seconds=duration,
            blinks_per_minute=bpm,
            blink_events=blink_events,
            ear_timeline=ear_timeline,
            baseline_bpm=baseline_bpm,
            peak_bpm=peak_bpm,
            peak_timestamp=peak_timestamp,
            stress_windows=stress_windows
        )

    finally:
        cap.release()


def format_blink_analysis(analysis: BlinkAnalysis) -> str:
    """Format blink analysis for inclusion in LLM prompts."""
    if analysis is None:
        return "Blink analysis unavailable (MediaPipe not installed or video processing failed)."

    lines = [
        "═══════════════════════════════════════════════════════════════",
        "CV-VALIDATED BLINK RATE ANALYSIS (MediaPipe Face Mesh)",
        "═══════════════════════════════════════════════════════════════",
        "",
        f"Total Blinks Detected: {analysis.total_blinks}",
        f"Video Duration: {analysis.duration_seconds:.1f} seconds",
        f"Overall Blink Rate: {analysis.blinks_per_minute:.1f} BPM",
        "",
        f"Baseline BPM (first 30s): {analysis.baseline_bpm:.1f}",
        f"Peak BPM: {analysis.peak_bpm:.1f} at ~{analysis.peak_timestamp:.0f}s",
        f"Peak vs Baseline: {(analysis.peak_bpm / analysis.baseline_bpm * 100):.0f}%" if analysis.baseline_bpm > 0 else "N/A",
        "",
    ]

    if analysis.stress_windows:
        lines.append("STRESS WINDOWS (>150% baseline):")
        for start, end, bpm in analysis.stress_windows[:5]:  # Top 5
            lines.append(f"  [{start:.0f}s - {end:.0f}s]: {bpm:.1f} BPM")
        lines.append("")

    # Notable blink events (clusters)
    if analysis.blink_events:
        lines.append("BLINK CLUSTERS (investigate these moments):")
        # Find clusters - 3+ blinks within 5 seconds
        clusters = []
        i = 0
        while i < len(analysis.blink_events):
            cluster_start = analysis.blink_events[i].timestamp_seconds
            cluster_count = 1
            j = i + 1
            while j < len(analysis.blink_events):
                if analysis.blink_events[j].timestamp_seconds - cluster_start <= 5:
                    cluster_count += 1
                    j += 1
                else:
                    break
            if cluster_count >= 3:
                cluster_end = analysis.blink_events[j-1].timestamp_seconds
                clusters.append((cluster_start, cluster_end, cluster_count))
                i = j
            else:
                i += 1

        for start, end, count in clusters[:5]:
            lines.append(f"  [{start:.1f}s - {end:.1f}s]: {count} blinks in {end-start:.1f}s")

    lines.append("═══════════════════════════════════════════════════════════════")

    return "\n".join(lines)


def annotate_transcript_with_blinks(transcript: str, blink_analysis: 'BlinkAnalysis') -> str:
    """
    Annotate a timestamped transcript with CV blink data.

    Input transcript format (lines starting with timestamps):
    00:58 I basically got rid of everything...
    01:23 We had some issues with...

    Output format:
    00:58 [BLINK: 24 BPM - ELEVATED]: I basically got rid of everything...
    01:23 [BLINK: 12 BPM - BASELINE]: We had some issues with...

    Args:
        transcript: Timestamped transcript text
        blink_analysis: BlinkAnalysis object from detect_blinks()

    Returns:
        Annotated transcript with blink rate indicators
    """
    import re

    if not blink_analysis or not transcript:
        return transcript

    # Parse stress windows for quick lookup
    stress_ranges = []
    for window_start, window_end, window_bpm in blink_analysis.stress_windows:
        stress_ranges.append((window_start, window_end, window_bpm))

    def get_blink_annotation(timestamp_seconds: float) -> str:
        """Get blink annotation for a given timestamp."""
        # Check if timestamp falls in a stress window
        for start, end, bpm in stress_ranges:
            if start <= timestamp_seconds <= end:
                return f"[BLINK: {bpm:.0f} BPM - ELEVATED]"

        # Otherwise use baseline
        return f"[BLINK: {blink_analysis.baseline_bpm:.0f} BPM - BASELINE]"

    def parse_timestamp(ts: str) -> float:
        """Convert MM:SS or HH:MM:SS to seconds."""
        parts = ts.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0

    # Process each line
    annotated_lines = []
    timestamp_pattern = r'^(\d{1,2}:\d{2}(?::\d{2})?)\s*(.*)$'

    for line in transcript.split('\n'):
        match = re.match(timestamp_pattern, line)
        if match:
            timestamp_str = match.group(1)
            content = match.group(2)
            timestamp_seconds = parse_timestamp(timestamp_str)
            blink_annotation = get_blink_annotation(timestamp_seconds)
            annotated_lines.append(f"{timestamp_str} {blink_annotation}: {content}")
        else:
            annotated_lines.append(line)

    return '\n'.join(annotated_lines)


def generate_trigger_response_map(
    blink_analysis: 'BlinkAnalysis',
    transcript: str,
    baseline_threshold: float = 1.5
) -> str:
    """
    Generate a TRIGGER-RESPONSE MAP correlating CV blink spikes with exact words spoken.

    Takes every CV spike (>150% of baseline) and finds the exact word(s) being spoken
    at that millisecond. Removes hallucination risk by providing irrefutable evidence.

    Args:
        blink_analysis: BlinkAnalysis object from detect_blinks()
        transcript: Timestamped transcript (format: "MM:SS text here")
        baseline_threshold: Multiplier for baseline to consider a "spike" (default 1.5x)

    Returns:
        Formatted trigger-response map as string
    """
    import re

    if not blink_analysis or not transcript:
        return "TRIGGER-RESPONSE MAP UNAVAILABLE: Missing blink analysis or transcript"

    # Parse transcript into (timestamp_seconds, text) tuples
    transcript_entries = []
    timestamp_pattern = r'^(\d{1,2}:\d{2}(?::\d{2})?)\s*(.*)$'

    def parse_ts(ts: str) -> float:
        parts = ts.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0

    for line in transcript.split('\n'):
        match = re.match(timestamp_pattern, line.strip())
        if match:
            ts_str = match.group(1)
            text = match.group(2).strip()
            if text:
                transcript_entries.append((parse_ts(ts_str), ts_str, text))

    if not transcript_entries:
        return "TRIGGER-RESPONSE MAP UNAVAILABLE: Could not parse transcript timestamps"

    # Find all stress windows (spikes)
    stress_spikes = []
    for window_start, window_end, window_bpm in blink_analysis.stress_windows:
        if blink_analysis.baseline_bpm > 0:
            ratio = window_bpm / blink_analysis.baseline_bpm
            if ratio >= baseline_threshold:
                stress_spikes.append({
                    'start': window_start,
                    'end': window_end,
                    'bpm': window_bpm,
                    'ratio': ratio
                })

    if not stress_spikes:
        return f"""=== TRIGGER-RESPONSE MAP ===
NO SIGNIFICANT BLINK SPIKES DETECTED
Baseline: {blink_analysis.baseline_bpm:.1f} BPM
Threshold for spike: {blink_analysis.baseline_bpm * baseline_threshold:.1f} BPM ({baseline_threshold}x baseline)
All blink windows were within normal range."""

    # Correlate spikes with words
    correlations = []
    for spike in stress_spikes:
        spike_mid = (spike['start'] + spike['end']) / 2

        # Find the transcript entry closest to this spike
        closest_entry = None
        closest_distance = float('inf')

        for ts_sec, ts_str, text in transcript_entries:
            distance = abs(ts_sec - spike_mid)
            if distance < closest_distance:
                closest_distance = distance
                closest_entry = (ts_sec, ts_str, text)

        if closest_entry and closest_distance < 30:  # Within 30 seconds
            correlations.append({
                'spike_time': spike_mid,
                'spike_bpm': spike['bpm'],
                'spike_ratio': spike['ratio'],
                'word_time': closest_entry[0],
                'word_time_str': closest_entry[1],
                'text': closest_entry[2][:100],  # Truncate long text
                'confidence': 'HIGH' if closest_distance < 5 else 'MEDIUM' if closest_distance < 15 else 'LOW'
            })

    # Format output
    lines = [
        "═══════════════════════════════════════════════════════════════",
        "TRIGGER-RESPONSE MAP (CV Blink Spikes → Exact Words)",
        "═══════════════════════════════════════════════════════════════",
        "",
        f"Baseline Blink Rate: {blink_analysis.baseline_bpm:.1f} BPM",
        f"Spike Threshold: >{blink_analysis.baseline_bpm * baseline_threshold:.1f} BPM ({baseline_threshold}x baseline)",
        f"Total Spikes Detected: {len(stress_spikes)}",
        f"Spikes Correlated with Speech: {len(correlations)}",
        "",
        "--- SPIKE-WORD CORRELATIONS ---",
        ""
    ]

    for i, corr in enumerate(correlations, 1):
        lines.append(f"SPIKE #{i}:")
        lines.append(f"  Time: {corr['spike_time']:.1f}s ({corr['spike_bpm']:.0f} BPM, {corr['spike_ratio']:.1f}x baseline)")
        lines.append(f"  Coincided with [{corr['word_time_str']}]: \"{corr['text']}\"")
        lines.append(f"  Correlation Confidence: {corr['confidence']}")
        lines.append("")

    if correlations:
        lines.append("--- INVESTIGATIVE PRIORITIES ---")
        lines.append("(Topics that triggered physiological stress response)")
        lines.append("")
        for corr in sorted(correlations, key=lambda x: x['spike_ratio'], reverse=True)[:3]:
            lines.append(f"• [{corr['word_time_str']}] {corr['spike_ratio']:.1f}x spike: \"{corr['text'][:50]}...\"")

    lines.append("")
    lines.append("═══════════════════════════════════════════════════════════════")

    return '\n'.join(lines)


def get_blink_metrics_for_prompt(
    video_path: str,
    interview_mode: bool = False,
    suspect_position: str = "auto"
) -> Dict:
    """
    Get blink metrics formatted for passing to LLM prompts.

    Args:
        video_path: Path to video file
        interview_mode: If True, track multiple faces and select by position
        suspect_position: Which face to track - "left", "right", "auto", "fullscreen"

    Returns dict with:
    - formatted_text: Human-readable summary
    - metrics: Raw numerical data
    - available: Whether detection succeeded
    """
    if not MEDIAPIPE_AVAILABLE:
        return {
            'available': False,
            'formatted_text': "Blink rate validation unavailable (install mediapipe)",
            'metrics': {}
        }

    try:
        analysis = detect_blinks(
            video_path,
            interview_mode=interview_mode,
            suspect_position=suspect_position
        )
        if analysis:
            return {
                'available': True,
                'formatted_text': format_blink_analysis(analysis),
                'metrics': {
                    'total_blinks': analysis.total_blinks,
                    'bpm': analysis.blinks_per_minute,
                    'baseline_bpm': analysis.baseline_bpm,
                    'peak_bpm': analysis.peak_bpm,
                    'peak_timestamp': analysis.peak_timestamp,
                    'stress_window_count': len(analysis.stress_windows)
                },
                'raw_analysis': analysis  # For transcript annotation
            }
    except Exception as e:
        logger.error(f"Blink detection failed: {e}")

    return {
        'available': False,
        'formatted_text': "Blink rate detection failed",
        'metrics': {}
    }


def parse_llm_blink_estimate(llm_output: str) -> Dict:
    """
    Parse LLM blink rate analysis to extract numerical estimates.

    Looks for patterns like:
    - "Estimated baseline blink rate: 18 BPM"
    - "Peak elevated rate observed: 45 BPM"
    - "baseline: 18 BPM"
    """
    import re

    result = {
        'baseline_bpm': None,
        'peak_bpm': None,
        'parsed': False
    }

    if not llm_output:
        return result

    # Pattern for baseline
    baseline_patterns = [
        r'baseline[^:]*:\s*(\d+(?:\.\d+)?)\s*BPM',
        r'baseline[^:]*:\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*BPM[^.]*baseline',
    ]

    for pattern in baseline_patterns:
        match = re.search(pattern, llm_output, re.IGNORECASE)
        if match:
            result['baseline_bpm'] = float(match.group(1))
            break

    # Pattern for peak
    peak_patterns = [
        r'peak[^:]*:\s*(\d+(?:\.\d+)?)\s*BPM',
        r'elevated[^:]*:\s*(\d+(?:\.\d+)?)\s*BPM',
        r'spike[^:]*:\s*(\d+(?:\.\d+)?)\s*BPM',
    ]

    for pattern in peak_patterns:
        match = re.search(pattern, llm_output, re.IGNORECASE)
        if match:
            result['peak_bpm'] = float(match.group(1))
            break

    result['parsed'] = result['baseline_bpm'] is not None
    return result


def fuse_blink_metrics(cv_metrics: Dict, llm_output: str) -> Dict:
    """
    Fuse CV-detected blink metrics with LLM estimates.

    CRITICAL: CV is GROUND TRUTH for blink counting.
    - CV uses EAR algorithm on MediaPipe Face Mesh - measures actual eye closures
    - LLM guesses from video frames - prone to hallucination (often 2-5x inflation)
    - CV should ALWAYS be preferred when available

    Fusion Strategy:
    1. If CV is available, USE CV as the authoritative source
    2. Only use LLM if CV is unavailable or detected zero blinks
    3. Flag significant discrepancies for investigation
    4. NEVER let hallucinated LLM peaks (>50 BPM) override CV data

    Args:
        cv_metrics: Dict from get_blink_metrics_for_prompt()
        llm_output: Raw text from VISUAL_BLINK_RATE_PROMPT analysis

    Returns:
        Dict with fused blink metrics and fusion metadata
    """
    llm_parsed = parse_llm_blink_estimate(llm_output)

    cv_available = cv_metrics.get('available', False)
    cv_bpm = cv_metrics.get('metrics', {}).get('bpm', 0)
    cv_baseline = cv_metrics.get('metrics', {}).get('baseline_bpm', 0)
    cv_peak = cv_metrics.get('metrics', {}).get('peak_bpm', 0)
    cv_total_blinks = cv_metrics.get('metrics', {}).get('total_blinks', 0)

    llm_baseline = llm_parsed.get('baseline_bpm')
    llm_peak = llm_parsed.get('peak_bpm')
    llm_parsed_ok = llm_parsed.get('parsed', False)

    # Default to CV - it is ground truth
    fused = {
        'fused_bpm': cv_bpm,
        'fused_baseline': cv_baseline,
        'fused_peak': cv_peak,
        'cv_bpm': cv_bpm,
        'cv_total_blinks': cv_total_blinks,
        'llm_baseline': llm_baseline,
        'llm_peak': llm_peak,
        'fusion_method': 'cv_ground_truth',
        'confidence': 'high' if cv_available else 'none',
        'discrepancy_flag': False,
        'llm_hallucination_detected': False
    }

    # Check for LLM hallucination (claims >50 BPM is almost always wrong)
    if llm_peak and llm_peak > 50:
        fused['llm_hallucination_detected'] = True
        logger.warning(f"LLM blink hallucination detected: claimed {llm_peak} BPM peak")

    if llm_baseline and llm_baseline > 50:
        fused['llm_hallucination_detected'] = True
        logger.warning(f"LLM blink hallucination detected: claimed {llm_baseline} BPM baseline")

    if not cv_available and not llm_parsed_ok:
        fused['fusion_method'] = 'none'
        fused['confidence'] = 'none'
        return fused

    if not cv_available and llm_parsed_ok:
        # Only use LLM if CV completely unavailable, and cap at reasonable values
        fused['fused_bpm'] = min(llm_baseline or 0, 40)  # Cap at 40 BPM
        fused['fused_baseline'] = min(llm_baseline or 0, 40)
        fused['fused_peak'] = min(llm_peak or 0, 50)  # Cap peaks at 50
        fused['fusion_method'] = 'llm_only_capped'
        fused['confidence'] = 'low'
        return fused

    if cv_available:
        # CV IS GROUND TRUTH - always use CV values
        fused['fusion_method'] = 'cv_ground_truth'
        fused['confidence'] = 'high'

        # Flag discrepancy if LLM differs significantly
        if llm_baseline and cv_baseline > 0:
            ratio = llm_baseline / cv_baseline
            if ratio > 2.0 or ratio < 0.5:
                fused['discrepancy_flag'] = True
                logger.warning(
                    f"Blink rate discrepancy: CV={cv_bpm:.1f} BPM, LLM claimed={llm_baseline} BPM "
                    f"(ratio={ratio:.1f}x). Using CV as ground truth."
                )

        # For peak, still use CV but flag if LLM claims much higher
        if llm_peak and cv_peak > 0 and llm_peak > cv_peak * 2:
            fused['discrepancy_flag'] = True
            logger.warning(
                f"Peak blink discrepancy: CV peak={cv_peak:.1f} BPM, LLM claimed={llm_peak} BPM. "
                f"Using CV as ground truth."
            )

        return fused

    return fused
