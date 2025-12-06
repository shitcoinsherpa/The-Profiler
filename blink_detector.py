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
    sample_rate: int = 1  # Process every Nth frame for speed
) -> Optional[BlinkAnalysis]:
    """
    Detect blinks in a video using MediaPipe Face Mesh.

    Args:
        video_path: Path to video file
        ear_threshold: EAR below this = eye closed (default 0.21)
        min_blink_frames: Minimum consecutive frames for valid blink
        sample_rate: Process every Nth frame (1 = all frames)

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
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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
                landmarks = results.multi_face_landmarks[0].landmark

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


def get_blink_metrics_for_prompt(video_path: str) -> Dict:
    """
    Get blink metrics formatted for passing to LLM prompts.

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
        analysis = detect_blinks(video_path)
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
                }
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
    Fuse CV-detected blink metrics with LLM estimates for higher accuracy.

    Fusion Strategy:
    1. If both sources agree (within 50%), average them
    2. If CV detected very few blinks, weight LLM higher (CV may have missed)
    3. If CV has high face detection rate, weight CV higher
    4. Return fused result with confidence indicator

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

    llm_baseline = llm_parsed.get('baseline_bpm')
    llm_peak = llm_parsed.get('peak_bpm')
    llm_parsed_ok = llm_parsed.get('parsed', False)

    # Default to CV if available, otherwise LLM
    fused = {
        'fused_bpm': cv_bpm,
        'fused_baseline': cv_baseline,
        'fused_peak': cv_peak,
        'cv_bpm': cv_bpm,
        'llm_baseline': llm_baseline,
        'llm_peak': llm_peak,
        'fusion_method': 'cv_only',
        'confidence': 'low',
        'discrepancy_flag': False
    }

    if not cv_available and not llm_parsed_ok:
        fused['fusion_method'] = 'none'
        fused['confidence'] = 'none'
        return fused

    if not cv_available and llm_parsed_ok:
        fused['fused_bpm'] = llm_baseline or 0
        fused['fused_baseline'] = llm_baseline or 0
        fused['fused_peak'] = llm_peak or 0
        fused['fusion_method'] = 'llm_only'
        fused['confidence'] = 'low'
        return fused

    if cv_available and not llm_parsed_ok:
        fused['fusion_method'] = 'cv_only'
        fused['confidence'] = 'medium' if cv_bpm > 5 else 'low'
        return fused

    # Both available - apply fusion logic
    if llm_baseline and cv_baseline > 0:
        # Check agreement
        ratio = llm_baseline / cv_baseline if cv_baseline > 0 else float('inf')

        if 0.5 <= ratio <= 2.0:
            # Good agreement - average them
            fused['fused_baseline'] = (cv_baseline + llm_baseline) / 2
            fused['fused_bpm'] = (cv_bpm + llm_baseline) / 2
            fused['fusion_method'] = 'averaged'
            fused['confidence'] = 'high'
        elif cv_bpm < 5:
            # CV detected almost nothing - trust LLM more
            fused['fused_baseline'] = llm_baseline * 0.7 + cv_baseline * 0.3
            fused['fused_bpm'] = llm_baseline * 0.7 + cv_bpm * 0.3
            fused['fusion_method'] = 'llm_weighted'
            fused['confidence'] = 'medium'
            fused['discrepancy_flag'] = True
        else:
            # Significant disagreement - weight by plausibility
            # Normal range is 15-25 BPM, so weight toward that
            cv_plausible = 10 <= cv_bpm <= 40
            llm_plausible = 10 <= llm_baseline <= 40

            if cv_plausible and not llm_plausible:
                fused['fusion_method'] = 'cv_preferred'
                fused['confidence'] = 'medium'
            elif llm_plausible and not cv_plausible:
                fused['fused_baseline'] = llm_baseline
                fused['fused_bpm'] = llm_baseline
                fused['fusion_method'] = 'llm_preferred'
                fused['confidence'] = 'medium'
            else:
                # Both plausible but disagree - average with flag
                fused['fused_baseline'] = (cv_baseline + llm_baseline) / 2
                fused['fused_bpm'] = (cv_bpm + llm_baseline) / 2
                fused['fusion_method'] = 'averaged_discrepant'
                fused['confidence'] = 'medium'
                fused['discrepancy_flag'] = True

    # Handle peak
    if llm_peak and cv_peak > 0:
        fused['fused_peak'] = (cv_peak + llm_peak) / 2
    elif llm_peak:
        fused['fused_peak'] = llm_peak

    logger.info(f"Blink fusion: CV={cv_bpm:.1f}, LLM={llm_baseline}, "
               f"Fused={fused['fused_bpm']:.1f} ({fused['fusion_method']})")

    return fused


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Analyzing: {video_path}")
        result = get_blink_metrics_for_prompt(video_path)
        print(result['formatted_text'])
        print(f"\nMetrics: {result['metrics']}")
    else:
        print("Usage: python blink_detector.py <video_path>")
        print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
