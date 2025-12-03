"""
Video frame extraction module.
Extracts evenly-spaced frames from video and converts to base64 JPEG.
"""

import cv2
import base64
import numpy as np
import os
from typing import List, Tuple


def extract_frames_from_video(
    video_path: str,
    num_frames: int = 5,
    target_size: int = 768,
    jpeg_quality: int = 85,
    max_file_size_mb: float = 100.0,
    min_duration_sec: float = 10.0,
    max_duration_sec: float = 300.0
) -> Tuple[List[str], dict]:
    """
    Extract evenly-spaced frames from video and return as base64-encoded JPEG strings.

    Args:
        video_path: Path to video file (.mp4, .mov, .avi, .webm)
        num_frames: Number of frames to extract (default: 5)
        target_size: Target frame size in pixels (default: 768x768)
        jpeg_quality: JPEG quality 0-100 (default: 85)
        max_file_size_mb: Maximum allowed video file size in MB
        min_duration_sec: Minimum video duration in seconds
        max_duration_sec: Maximum video duration in seconds

    Returns:
        Tuple of (list of base64 strings, metadata dict)

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video is invalid or doesn't meet requirements
    """

    # Validate file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check file extension
    _, ext = os.path.splitext(video_path)
    supported_formats = ['.mp4', '.mov', '.avi', '.webm']
    if ext.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    # Check file size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise ValueError(
            f"Video file too large: {file_size_mb:.2f}MB "
            f"(maximum: {max_file_size_mb}MB)"
        )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate properties
        if total_frames <= 0 or fps <= 0:
            raise ValueError(
                "Invalid video: Cannot determine frame count or FPS. "
                "The video may be corrupted."
            )

        if width <= 0 or height <= 0:
            raise ValueError("Invalid video: Invalid frame dimensions")

        # Calculate duration
        duration_sec = total_frames / fps

        # Check duration constraints
        if duration_sec < min_duration_sec:
            raise ValueError(
                f"Video too short: {duration_sec:.1f}s "
                f"(minimum: {min_duration_sec}s)"
            )

        if duration_sec > max_duration_sec:
            raise ValueError(
                f"Video too long: {duration_sec:.1f}s "
                f"(maximum: {max_duration_sec}s)"
            )

        # Calculate frame indices for evenly-spaced extraction
        if num_frames >= total_frames:
            # Extract all frames if fewer than requested
            frame_indices = list(range(total_frames))
        else:
            # Calculate evenly-spaced indices
            # Include first and last frame
            frame_indices = [
                int(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]

        # Extract and encode frames
        base64_frames = []
        extracted_indices = []

        for frame_idx in frame_indices:
            # Set position to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Failed to read frame at index {frame_idx}, skipping...")
                continue

            # Resize frame with padding to maintain aspect ratio
            resized_frame = _resize_with_padding(frame, target_size)

            # Convert to base64 JPEG
            base64_str = _frame_to_base64_jpeg(resized_frame, jpeg_quality)

            base64_frames.append(base64_str)
            extracted_indices.append(frame_idx)

        if not base64_frames:
            raise ValueError("No frames could be extracted from video")

        # Prepare metadata
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration_sec,
            'resolution': (width, height),
            'file_size_mb': file_size_mb,
            'frames_extracted': len(base64_frames),
            'frame_indices': extracted_indices,
            'target_size': target_size,
            'jpeg_quality': jpeg_quality
        }

        return base64_frames, metadata

    finally:
        cap.release()


def _resize_with_padding(frame: np.ndarray, target_size: int = 768) -> np.ndarray:
    """
    Resize frame to target_size x target_size maintaining aspect ratio.
    Adds black padding if aspect ratio doesn't match.

    Args:
        frame: OpenCV image (numpy array in BGR format)
        target_size: Target size for both width and height

    Returns:
        Resized frame with padding as numpy array
    """
    h, w = frame.shape[:2]

    # Calculate scale to fit within target size
    scale = min(target_size / w, target_size / h)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize using INTER_AREA (best for downscaling)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Calculate position to center the image
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    # Place resized frame on canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def _frame_to_base64_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    """
    Convert OpenCV frame to base64-encoded JPEG string.

    Args:
        frame: OpenCV image (numpy array in BGR format)
        quality: JPEG quality (0-100, default 85)

    Returns:
        Base64 encoded JPEG string

    Raises:
        ValueError: If encoding fails
    """
    # Encode frame to JPEG with quality setting
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode('.jpg', frame, encode_param)

    if not success:
        raise ValueError("Failed to encode frame to JPEG")

    # Convert to base64 string
    base64_string = base64.b64encode(encoded_image).decode('utf-8')

    return base64_string


def validate_video_file(video_path: str) -> dict:
    """
    Validate video file and return metadata without extracting frames.
    Useful for quick validation before processing.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If video cannot be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or fps <= 0:
            raise ValueError("Invalid video properties")

        duration_sec = total_frames / fps
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        return {
            'valid': True,
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration_sec,
            'resolution': (width, height),
            'file_size_mb': file_size_mb
        }
    finally:
        cap.release()
