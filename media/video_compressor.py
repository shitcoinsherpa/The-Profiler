"""
Video compression module for reducing large video file sizes before API processing.
Uses ffmpeg to reencode videos to 720p H.264 with optimized settings.
"""

import subprocess
import os
import tempfile
import logging
import shutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Compression threshold in bytes (48MB)
COMPRESSION_THRESHOLD_BYTES = 48 * 1024 * 1024

# Target settings for compressed video
TARGET_HEIGHT = 720  # 720p
TARGET_VIDEO_BITRATE = "1500k"  # Good quality for analysis
TARGET_AUDIO_BITRATE = "128k"  # Preserve voice quality
TARGET_AUDIO_SAMPLE_RATE = 16000  # Good for speech


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
    return {}


def compress_video_for_api(
    video_path: str,
    output_path: Optional[str] = None,
    target_height: int = TARGET_HEIGHT,
    video_bitrate: str = TARGET_VIDEO_BITRATE,
    audio_bitrate: str = TARGET_AUDIO_BITRATE
) -> Tuple[str, dict]:
    """
    Compress video to reduce file size for API processing.

    Args:
        video_path: Path to input video
        output_path: Optional output path (uses temp file if not provided)
        target_height: Target height in pixels (width scales proportionally)
        video_bitrate: Target video bitrate (e.g., "1500k")
        audio_bitrate: Target audio bitrate (e.g., "128k")

    Returns:
        Tuple of (compressed_video_path, compression_info_dict)

    Raises:
        RuntimeError: If ffmpeg is not available or compression fails
    """
    if not check_ffmpeg_available():
        raise RuntimeError("ffmpeg not found. Install ffmpeg to enable video compression.")

    original_size = os.path.getsize(video_path)

    # Generate output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(temp_dir, f"{base_name}_compressed.mp4")

    logger.info(f"Compressing video: {video_path} ({original_size / 1024 / 1024:.1f}MB)")

    # Build ffmpeg command
    # -vf scale=-2:{height} maintains aspect ratio, -2 ensures divisible by 2
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-vf", f"scale=-2:{target_height}",
        "-c:v", "libx264",
        "-preset", "fast",  # Balance speed/quality
        "-crf", "23",  # Constant rate factor (18-28 reasonable range)
        "-b:v", video_bitrate,
        "-maxrate", video_bitrate,
        "-bufsize", "3000k",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(TARGET_AUDIO_SAMPLE_RATE),
        "-movflags", "+faststart",  # Enable streaming
        output_path
    ]

    try:
        logger.info(f"Running ffmpeg compression...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for long videos
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise RuntimeError(f"ffmpeg compression failed: {result.stderr[:500]}")

        compressed_size = os.path.getsize(output_path)
        reduction_pct = (1 - compressed_size / original_size) * 100

        compression_info = {
            'original_path': video_path,
            'compressed_path': output_path,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'original_size_mb': original_size / 1024 / 1024,
            'compressed_size_mb': compressed_size / 1024 / 1024,
            'reduction_percent': reduction_pct,
            'target_height': target_height,
            'video_bitrate': video_bitrate,
            'audio_bitrate': audio_bitrate
        }

        logger.info(
            f"Compression complete: {original_size / 1024 / 1024:.1f}MB -> "
            f"{compressed_size / 1024 / 1024:.1f}MB ({reduction_pct:.1f}% reduction)"
        )

        return output_path, compression_info

    except subprocess.TimeoutExpired:
        raise RuntimeError("Video compression timed out after 10 minutes")


def maybe_compress_video(
    video_path: str,
    threshold_bytes: int = COMPRESSION_THRESHOLD_BYTES
) -> Tuple[str, Optional[dict]]:
    """
    Compress video if it exceeds the size threshold.

    Args:
        video_path: Path to video file
        threshold_bytes: Size threshold in bytes (default: 48MB)

    Returns:
        Tuple of (video_path_to_use, compression_info_or_none)
        If video is under threshold, returns original path and None.
        If compressed, returns compressed path and compression info dict.
    """
    file_size = os.path.getsize(video_path)

    if file_size <= threshold_bytes:
        logger.info(
            f"Video size ({file_size / 1024 / 1024:.1f}MB) under threshold "
            f"({threshold_bytes / 1024 / 1024:.0f}MB) - no compression needed"
        )
        return video_path, None

    logger.info(
        f"Video size ({file_size / 1024 / 1024:.1f}MB) exceeds threshold "
        f"({threshold_bytes / 1024 / 1024:.0f}MB) - compressing..."
    )

    if not check_ffmpeg_available():
        logger.warning(
            "ffmpeg not available - cannot compress video. "
            "Processing with original file (may cause API issues with large files)."
        )
        return video_path, None

    try:
        compressed_path, compression_info = compress_video_for_api(video_path)
        return compressed_path, compression_info
    except Exception as e:
        logger.error(f"Video compression failed: {e}. Using original file.")
        return video_path, None


def cleanup_compressed_video(compression_info: Optional[dict]) -> None:
    """
    Clean up temporary compressed video file.

    Args:
        compression_info: Compression info dict from maybe_compress_video,
                         or None if no compression occurred
    """
    if compression_info is None:
        return

    compressed_path = compression_info.get('compressed_path')
    if compressed_path and os.path.exists(compressed_path):
        try:
            os.remove(compressed_path)
            logger.debug(f"Cleaned up compressed video: {compressed_path}")
        except Exception as e:
            logger.warning(f"Could not clean up compressed video: {e}")
