"""
Video downloader module for importing videos from URLs.
Supports YouTube, Vimeo, and direct video URLs.
"""

import os
import re
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Download directory
DOWNLOADS_DIR = Path(__file__).parent / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.

    Args:
        url: String to check

    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False


def is_youtube_url(url: str) -> bool:
    """
    Check if URL is a YouTube video URL.

    Args:
        url: URL to check

    Returns:
        True if YouTube URL, False otherwise
    """
    youtube_patterns = [
        r'(youtube\.com/watch\?v=)',
        r'(youtu\.be/)',
        r'(youtube\.com/embed/)',
        r'(youtube\.com/v/)',
        r'(youtube\.com/shorts/)',
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def is_supported_url(url: str) -> Tuple[bool, str]:
    """
    Check if URL is supported for download.

    Args:
        url: URL to check

    Returns:
        Tuple of (is_supported, platform_name)
    """
    if not is_valid_url(url):
        return False, "invalid"

    # YouTube
    if is_youtube_url(url):
        return True, "YouTube"

    # Vimeo
    if 'vimeo.com' in url:
        return True, "Vimeo"

    # Twitter/X
    if 'twitter.com' in url or 'x.com' in url:
        return True, "Twitter/X"

    # TikTok
    if 'tiktok.com' in url:
        return True, "TikTok"

    # Direct video URLs
    video_extensions = ['.mp4', '.webm', '.mov', '.avi']
    parsed = urlparse(url)
    if any(parsed.path.lower().endswith(ext) for ext in video_extensions):
        return True, "Direct URL"

    # Try yt-dlp for other URLs (it supports many sites)
    return True, "Other"


def download_video(
    url: str,
    max_duration: int = 300,
    max_filesize_mb: int = 100,
    output_dir: Optional[str] = None
) -> Tuple[str, dict]:
    """
    Download video from URL using yt-dlp.

    Args:
        url: Video URL to download
        max_duration: Maximum video duration in seconds
        max_filesize_mb: Maximum file size in MB
        output_dir: Output directory (uses temp if not specified)

    Returns:
        Tuple of (local_file_path, metadata_dict)

    Raises:
        ImportError: If yt-dlp is not installed
        ValueError: If URL is invalid or video doesn't meet requirements
        Exception: If download fails
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required for video downloads. "
            "Install with: pip install yt-dlp"
        )

    # Validate URL
    is_supported, platform = is_supported_url(url)
    if not is_supported:
        raise ValueError(f"Invalid or unsupported URL: {url}")

    logger.info(f"Downloading video from {platform}: {url}")

    # Set output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = DOWNLOADS_DIR

    out_dir.mkdir(exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4][height<=1080]/best[ext=mp4]/best',
        'outtmpl': str(out_dir / '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        # Duration and size filters
        'match_filter': yt_dlp.utils.match_filter_func(
            f"duration <= {max_duration}"
        ) if max_duration else None,
        # Restrict to safe formats
        'restrictfilenames': True,
        # Download only video, not playlists
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, extract info without downloading
            logger.debug("Extracting video info...")
            info = ydl.extract_info(url, download=False)

            if info is None:
                raise ValueError("Could not extract video information")

            # Check duration
            duration = info.get('duration', 0)
            if duration and duration > max_duration:
                raise ValueError(
                    f"Video is too long: {duration}s (max: {max_duration}s)"
                )

            # Check if it's a live stream
            if info.get('is_live'):
                raise ValueError("Live streams are not supported")

            # Estimate file size (if available)
            filesize = info.get('filesize') or info.get('filesize_approx', 0)
            if filesize:
                filesize_mb = filesize / (1024 * 1024)
                if filesize_mb > max_filesize_mb:
                    raise ValueError(
                        f"Video file too large: {filesize_mb:.1f}MB (max: {max_filesize_mb}MB)"
                    )

            # Now download
            logger.debug("Downloading video...")
            ydl.download([url])

            # Get the output filename
            video_id = info.get('id', 'video')
            video_ext = info.get('ext', 'mp4')
            output_path = out_dir / f"{video_id}.{video_ext}"

            if not output_path.exists():
                # Try to find the downloaded file
                for ext in ['mp4', 'webm', 'mkv', 'mov']:
                    potential_path = out_dir / f"{video_id}.{ext}"
                    if potential_path.exists():
                        output_path = potential_path
                        break

            if not output_path.exists():
                raise Exception("Download completed but file not found")

            # Verify file size
            actual_size_mb = output_path.stat().st_size / (1024 * 1024)
            if actual_size_mb > max_filesize_mb:
                output_path.unlink()  # Delete the file
                raise ValueError(
                    f"Downloaded file too large: {actual_size_mb:.1f}MB (max: {max_filesize_mb}MB)"
                )

            # Build metadata
            metadata = {
                'title': info.get('title', 'Unknown'),
                'duration_seconds': duration,
                'uploader': info.get('uploader', 'Unknown'),
                'platform': platform,
                'url': url,
                'file_size_mb': actual_size_mb,
                'resolution': f"{info.get('width', 0)}x{info.get('height', 0)}",
                'video_id': video_id,
            }

            logger.info(
                f"Downloaded: {metadata['title']} ({duration}s, {actual_size_mb:.1f}MB)"
            )

            return str(output_path), metadata

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if 'Video unavailable' in error_msg:
            raise ValueError("Video is unavailable or private")
        elif 'age-restricted' in error_msg.lower():
            raise ValueError("Video is age-restricted and cannot be downloaded")
        else:
            raise Exception(f"Download failed: {error_msg}")


def cleanup_downloads(max_age_hours: int = 24):
    """
    Clean up old downloaded files.

    Args:
        max_age_hours: Delete files older than this many hours
    """
    import time

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for file_path in DOWNLOADS_DIR.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted old download: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")


def get_video_info(url: str) -> dict:
    """
    Get video information without downloading.

    Args:
        url: Video URL

    Returns:
        Dictionary with video metadata
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp is required")

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if info is None:
                return {'error': 'Could not extract video info'}

            return {
                'title': info.get('title', 'Unknown'),
                'duration_seconds': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'thumbnail': info.get('thumbnail'),
                'description': info.get('description', '')[:200],
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date'),
                'is_live': info.get('is_live', False),
            }
    except Exception as e:
        return {'error': str(e)}
