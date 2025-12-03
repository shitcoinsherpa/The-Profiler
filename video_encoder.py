"""
Video encoding module for base64 conversion.
"""

import base64
import os


def encode_video_to_base64(video_path: str) -> str:
    """
    Encode video file to base64 string.

    Args:
        video_path: Path to video file

    Returns:
        Base64-encoded string of the video

    Raises:
        Exception: If encoding fails
    """
    try:
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            base64_video = base64.b64encode(video_bytes).decode('utf-8')

        return base64_video

    except Exception as e:
        raise Exception(f"Video encoding failed: {str(e)}")
