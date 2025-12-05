"""
Audio extraction module.
Extracts audio from video and converts to base64 for API analysis.
"""

import os
import base64
import subprocess
import tempfile
import logging
from typing import Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_path(path: str) -> str:
    """
    Sanitize a file path to prevent command injection.

    Args:
        path: Input file path

    Returns:
        Sanitized path

    Raises:
        ValueError: If path contains suspicious characters
    """
    # Convert to Path object and resolve to absolute path
    try:
        clean_path = Path(path).resolve()
    except Exception as e:
        raise ValueError(f"Invalid path: {e}")

    # Check for null bytes (common injection technique)
    if '\x00' in str(clean_path):
        raise ValueError("Path contains null bytes")

    # On Windows, check for dangerous patterns
    if os.name == 'nt':
        # Check for command chaining characters
        dangerous_patterns = ['|', '&', ';', '`', '$', '>', '<', '\n', '\r']
        path_str = str(clean_path)
        for pattern in dangerous_patterns:
            if pattern in path_str:
                raise ValueError(f"Path contains potentially dangerous character: {pattern}")

    # Verify the file exists
    if not clean_path.exists():
        raise FileNotFoundError(f"File not found: {clean_path}")

    # Verify it's a file, not a directory
    if not clean_path.is_file():
        raise ValueError(f"Path is not a file: {clean_path}")

    return str(clean_path)


def extract_audio_from_video(video_path: str) -> Tuple[str, dict]:
    """
    Extract audio from video file and convert to base64.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (base64 audio string, metadata dict)

    Raises:
        Exception: If audio extraction fails
    """
    temp_audio_path = None
    try:
        # Sanitize input path to prevent command injection
        safe_video_path = sanitize_path(video_path)
        logger.debug(f"Extracting audio from: {safe_video_path}")

        # Create temporary file for audio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_audio_path = temp_audio.name
        temp_audio.close()

        # Extract audio using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', safe_video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-ar', '16000',  # 16kHz sample rate (good for speech)
            '-ac', '1',  # Mono
            '-b:a', '64k',  # 64kbps bitrate
            '-y',  # Overwrite output
            temp_audio_path
        ]

        # Run ffmpeg
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")

        # Read audio file and convert to base64
        with open(temp_audio_path, 'rb') as f:
            audio_bytes = f.read()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

        # Get file size
        audio_size_kb = len(audio_bytes) / 1024

        # Clean up temp file
        os.unlink(temp_audio_path)

        metadata = {
            'format': 'mp3',
            'sample_rate': 16000,
            'channels': 1,
            'bitrate': '64k',
            'size_kb': round(audio_size_kb, 2)
        }

        return base64_audio, metadata

    except Exception as e:
        # Clean up temp file if it exists
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise Exception(f"Audio extraction failed: {str(e)}")
