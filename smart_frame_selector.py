"""
Intelligent frame selection module.
Selects the most informative frames using scene change detection and content analysis.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    """Score for a frame candidate."""
    frame_index: int
    timestamp: float
    scene_change_score: float  # How different from previous frame
    blur_score: float  # Lower is blurrier
    brightness_score: float  # Normalized brightness
    face_detected: bool  # Whether a face was detected
    overall_score: float  # Combined score


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate difference between two frames using histogram comparison.

    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)

    Returns:
        Difference score (0-1, higher means more different)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize histograms
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Compare histograms
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return diff


def calculate_blur_score(frame: np.ndarray) -> float:
    """
    Calculate blur score using Laplacian variance.
    Higher score means sharper image.

    Args:
        frame: Input frame (BGR)

    Returns:
        Blur score (higher = sharper)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_brightness(frame: np.ndarray) -> float:
    """
    Calculate normalized brightness of a frame.

    Args:
        frame: Input frame (BGR)

    Returns:
        Brightness score (0-1)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0
    return brightness


def detect_face(frame: np.ndarray, cascade_path: str = None) -> bool:
    """
    Detect if a face is present in the frame.

    Args:
        frame: Input frame (BGR)
        cascade_path: Optional path to Haar cascade file

    Returns:
        True if face detected
    """
    try:
        # Use OpenCV's pre-trained face detector
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return len(faces) > 0

    except Exception as e:
        logger.debug(f"Face detection failed: {e}")
        return False


def detect_scene_changes(
    video_path: str,
    threshold: float = 0.3,
    min_scene_length: int = 30  # Minimum frames between scene changes
) -> List[int]:
    """
    Detect scene change frames in a video.

    Args:
        video_path: Path to video file
        threshold: Difference threshold for scene change detection
        min_scene_length: Minimum frames between detected scene changes

    Returns:
        List of frame indices where scene changes occur
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        scene_changes = [0]  # Always include first frame
        prev_frame = None
        frame_idx = 0
        last_scene_change = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # Calculate difference from previous frame
                diff = calculate_frame_difference(prev_frame, frame)

                # Check if this is a scene change
                if diff > threshold and (frame_idx - last_scene_change) >= min_scene_length:
                    scene_changes.append(frame_idx)
                    last_scene_change = frame_idx
                    logger.debug(f"Scene change detected at frame {frame_idx} (diff={diff:.3f})")

            prev_frame = frame.copy()
            frame_idx += 1

        return scene_changes

    finally:
        cap.release()


def score_frames(
    video_path: str,
    candidate_indices: List[int],
    check_faces: bool = True
) -> List[FrameScore]:
    """
    Score candidate frames based on quality metrics.

    Args:
        video_path: Path to video file
        candidate_indices: List of frame indices to score
        check_faces: Whether to run face detection

    Returns:
        List of FrameScore objects
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        scores = []
        prev_frame = None

        for frame_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Calculate metrics
            blur_score = calculate_blur_score(frame)
            brightness = calculate_brightness(frame)

            # Scene change score (difference from previous analyzed frame)
            scene_score = 0.0
            if prev_frame is not None:
                scene_score = calculate_frame_difference(prev_frame, frame)

            # Face detection
            face_detected = detect_face(frame) if check_faces else False

            # Calculate overall score
            # Prioritize: sharpness, good brightness, scene diversity, faces
            brightness_penalty = abs(brightness - 0.5) * 2  # Penalty for too dark/bright
            overall = (
                (blur_score / 1000.0) * 0.3 +  # Normalize blur score
                (1 - brightness_penalty) * 0.2 +
                scene_score * 0.3 +
                (1.0 if face_detected else 0.0) * 0.2
            )

            scores.append(FrameScore(
                frame_index=frame_idx,
                timestamp=frame_idx / fps if fps > 0 else 0,
                scene_change_score=scene_score,
                blur_score=blur_score,
                brightness_score=brightness,
                face_detected=face_detected,
                overall_score=overall
            ))

            prev_frame = frame.copy()

        return scores

    finally:
        cap.release()


def select_best_frames(
    video_path: str,
    num_frames: int = 5,
    use_scene_detection: bool = True,
    use_face_detection: bool = True,
    scene_threshold: float = 0.3
) -> Tuple[List[int], List[FrameScore]]:
    """
    Select the best frames from a video for analysis.

    Strategy:
    1. Detect scene changes to find key moments
    2. Score frames based on quality (sharpness, brightness, faces)
    3. Select diverse, high-quality frames

    Args:
        video_path: Path to video file
        num_frames: Number of frames to select
        use_scene_detection: Whether to use scene change detection
        use_face_detection: Whether to use face detection
        scene_threshold: Threshold for scene change detection

    Returns:
        Tuple of (selected_frame_indices, frame_scores)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= num_frames:
            # If video has fewer frames than requested, return all
            return list(range(total_frames)), []

        # Generate candidate frames
        candidates = set()

        # Always include first and last frames
        candidates.add(0)
        candidates.add(total_frames - 1)

        # Add evenly spaced frames as baseline
        for i in range(num_frames):
            idx = int(i * (total_frames - 1) / (num_frames - 1))
            candidates.add(idx)

        # Add scene change frames if enabled
        if use_scene_detection:
            try:
                scene_changes = detect_scene_changes(
                    video_path,
                    threshold=scene_threshold
                )
                candidates.update(scene_changes)
                logger.info(f"Detected {len(scene_changes)} scene changes")
            except Exception as e:
                logger.warning(f"Scene detection failed: {e}")

        # Add some random samples for diversity
        np.random.seed(42)  # Reproducible
        random_samples = np.random.choice(
            range(total_frames),
            size=min(num_frames * 2, total_frames),
            replace=False
        )
        candidates.update(random_samples.tolist())

        # Score all candidate frames
        candidate_list = sorted(candidates)
        logger.info(f"Scoring {len(candidate_list)} candidate frames")

        scores = score_frames(
            video_path,
            candidate_list,
            check_faces=use_face_detection
        )

        # Sort by overall score
        scores.sort(key=lambda x: x.overall_score, reverse=True)

        # Select top frames ensuring good temporal distribution
        selected = []
        selected_times = []
        min_time_gap = (total_frames / fps) / (num_frames * 2) if fps > 0 else 0

        for score in scores:
            if len(selected) >= num_frames:
                break

            # Check temporal distribution
            too_close = False
            for t in selected_times:
                if abs(score.timestamp - t) < min_time_gap:
                    too_close = True
                    break

            if not too_close:
                selected.append(score.frame_index)
                selected_times.append(score.timestamp)

        # If we don't have enough, fill with evenly spaced frames
        if len(selected) < num_frames:
            for i in range(num_frames):
                idx = int(i * (total_frames - 1) / (num_frames - 1))
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= num_frames:
                        break

        # Sort by frame index for temporal order
        selected.sort()

        logger.info(f"Selected frames: {selected}")
        return selected[:num_frames], scores

    finally:
        cap.release()


def extract_smart_frames(
    video_path: str,
    num_frames: int = 5,
    target_size: int = 768,
    jpeg_quality: int = 85,
    use_smart_selection: bool = True
) -> Tuple[List[str], dict, List[FrameScore]]:
    """
    Extract frames using intelligent selection.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target frame size
        jpeg_quality: JPEG compression quality
        use_smart_selection: Whether to use smart selection (False = evenly spaced)

    Returns:
        Tuple of (base64_frames, metadata, frame_scores)
    """
    import base64

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Select frames
        if use_smart_selection:
            frame_indices, scores = select_best_frames(
                video_path,
                num_frames=num_frames,
                use_scene_detection=True,
                use_face_detection=True
            )
        else:
            # Evenly spaced
            frame_indices = [
                int(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]
            scores = []

        # Extract and encode frames
        base64_frames = []
        extracted_indices = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Resize with padding
            h, w = frame.shape[:2]
            scale = min(target_size / w, target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create canvas with padding
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            # Encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            success, encoded = cv2.imencode('.jpg', canvas, encode_param)

            if success:
                base64_str = base64.b64encode(encoded).decode('utf-8')
                base64_frames.append(base64_str)
                extracted_indices.append(frame_idx)

        # Build metadata
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': total_frames / fps if fps > 0 else 0,
            'resolution': (width, height),
            'frames_extracted': len(base64_frames),
            'frame_indices': extracted_indices,
            'smart_selection_used': use_smart_selection,
            'target_size': target_size,
            'jpeg_quality': jpeg_quality
        }

        # Add selection info if smart selection was used
        if use_smart_selection and scores:
            selected_scores = [s for s in scores if s.frame_index in extracted_indices]
            metadata['frame_scores'] = [
                {
                    'index': s.frame_index,
                    'timestamp': round(s.timestamp, 2),
                    'blur_score': round(s.blur_score, 2),
                    'brightness': round(s.brightness_score, 2),
                    'face_detected': s.face_detected,
                    'overall_score': round(s.overall_score, 4)
                }
                for s in selected_scores
            ]

        return base64_frames, metadata, scores

    finally:
        cap.release()
