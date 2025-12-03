"""
Content-based caching layer for analyzed videos.
Caches analysis results based on video content hash to avoid redundant API calls.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class CacheEntry:
    """Represents a cached analysis result."""
    video_hash: str
    models_hash: str
    timestamp: str
    result: Dict
    hit_count: int = 0
    last_accessed: str = ""


class VideoCache:
    """
    Content-based cache for video analysis results.
    Uses video file hash and model configuration to identify cached results.
    """

    def __init__(self, cache_dir: str = None, max_age_days: int = 30):
        """
        Initialize the video cache.

        Args:
            cache_dir: Custom cache directory path
            max_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age_days = max_age_days
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()

    def _load_index(self):
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    @staticmethod
    def compute_video_hash(video_path: str, chunk_size: int = 65536) -> str:
        """
        Compute a hash of the video file content.

        Args:
            video_path: Path to video file
            chunk_size: Size of chunks to read

        Returns:
            SHA-256 hash of the video content
        """
        hasher = hashlib.sha256()

        try:
            with open(video_path, 'rb') as f:
                # Read file in chunks to handle large files
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)

            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash video: {e}")
            raise

    @staticmethod
    def compute_models_hash(models_config: Dict) -> str:
        """
        Compute a hash of the model configuration.

        Args:
            models_config: Dictionary of model selections

        Returns:
            MD5 hash of the model configuration
        """
        config_str = json.dumps(models_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def get_cache_key(self, video_path: str, models_config: Dict) -> str:
        """
        Generate a cache key for a video + models combination.

        Args:
            video_path: Path to video file
            models_config: Dictionary of model selections

        Returns:
            Cache key string
        """
        video_hash = self.compute_video_hash(video_path)
        models_hash = self.compute_models_hash(models_config)
        return f"{video_hash[:16]}_{models_hash}"

    def get(
        self,
        video_path: str,
        models_config: Dict
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a cached result exists for this video + models combination.

        Args:
            video_path: Path to video file
            models_config: Dictionary of model selections

        Returns:
            Tuple of (cache_hit, result_or_None)
        """
        try:
            cache_key = self.get_cache_key(video_path, models_config)

            if cache_key not in self.index:
                logger.debug(f"Cache miss: {cache_key[:20]}...")
                return False, None

            entry_info = self.index[cache_key]

            # Check if cache entry has expired
            timestamp = datetime.fromisoformat(entry_info['timestamp'])
            if datetime.now() - timestamp > timedelta(days=self.max_age_days):
                logger.info(f"Cache expired: {cache_key[:20]}...")
                self._remove_entry(cache_key)
                return False, None

            # Load the cached result
            cache_file = self.cache_dir / f"{cache_key}.json"
            if not cache_file.exists():
                logger.warning(f"Cache file missing: {cache_key[:20]}...")
                self._remove_entry(cache_key)
                return False, None

            with open(cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)

            # Update access stats
            entry_info['hit_count'] = entry_info.get('hit_count', 0) + 1
            entry_info['last_accessed'] = datetime.now().isoformat()
            self._save_index()

            logger.info(f"Cache hit: {cache_key[:20]}... (hits: {entry_info['hit_count']})")
            return True, result

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return False, None

    def put(
        self,
        video_path: str,
        models_config: Dict,
        result: Dict
    ) -> bool:
        """
        Store an analysis result in the cache.

        Args:
            video_path: Path to video file
            models_config: Dictionary of model selections
            result: Analysis result to cache

        Returns:
            True if cached successfully
        """
        try:
            cache_key = self.get_cache_key(video_path, models_config)

            # Save the result to a file
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Update the index
            self.index[cache_key] = {
                'video_hash': self.compute_video_hash(video_path)[:16],
                'models_hash': self.compute_models_hash(models_config),
                'timestamp': datetime.now().isoformat(),
                'hit_count': 0,
                'last_accessed': datetime.now().isoformat(),
                'video_name': Path(video_path).name,
                'file_size_kb': os.path.getsize(cache_file) / 1024
            }
            self._save_index()

            logger.info(f"Cached result: {cache_key[:20]}...")
            return True

        except Exception as e:
            logger.error(f"Cache put error: {e}")
            return False

    def _remove_entry(self, cache_key: str):
        """Remove a cache entry."""
        try:
            # Remove the cache file
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()

            # Remove from index
            if cache_key in self.index:
                del self.index[cache_key]
                self._save_index()

        except Exception as e:
            logger.error(f"Failed to remove cache entry: {e}")

    def invalidate(self, video_path: str = None, all_entries: bool = False) -> int:
        """
        Invalidate cache entries.

        Args:
            video_path: Specific video to invalidate (all model configs)
            all_entries: If True, clear entire cache

        Returns:
            Number of entries invalidated
        """
        count = 0

        if all_entries:
            # Clear entire cache
            for cache_key in list(self.index.keys()):
                self._remove_entry(cache_key)
                count += 1
            logger.info(f"Cleared entire cache ({count} entries)")
            return count

        if video_path:
            # Invalidate all entries for a specific video
            video_hash = self.compute_video_hash(video_path)[:16]
            for cache_key in list(self.index.keys()):
                if self.index[cache_key].get('video_hash') == video_hash:
                    self._remove_entry(cache_key)
                    count += 1
            logger.info(f"Invalidated {count} entries for video")
            return count

        return 0

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        count = 0
        cutoff = datetime.now() - timedelta(days=self.max_age_days)

        for cache_key in list(self.index.keys()):
            entry_info = self.index[cache_key]
            timestamp = datetime.fromisoformat(entry_info['timestamp'])
            if timestamp < cutoff:
                self._remove_entry(cache_key)
                count += 1

        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")

        return count

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self.index)
        total_hits = sum(e.get('hit_count', 0) for e in self.index.values())

        # Calculate total cache size
        total_size_kb = sum(e.get('file_size_kb', 0) for e in self.index.values())

        return {
            'total_entries': total_entries,
            'total_hits': total_hits,
            'total_size_kb': round(total_size_kb, 2),
            'total_size_mb': round(total_size_kb / 1024, 2),
            'cache_dir': str(self.cache_dir),
            'max_age_days': self.max_age_days
        }

    def list_entries(self, limit: int = 20) -> list:
        """
        List recent cache entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of cache entry summaries
        """
        entries = []
        for cache_key, info in sorted(
            self.index.items(),
            key=lambda x: x[1].get('last_accessed', ''),
            reverse=True
        )[:limit]:
            entries.append({
                'key': cache_key[:20] + '...',
                'video_name': info.get('video_name', 'unknown'),
                'timestamp': info.get('timestamp', '')[:10],
                'hits': info.get('hit_count', 0),
                'size_kb': round(info.get('file_size_kb', 0), 1)
            })
        return entries


# Global cache instance
_cache_instance: Optional[VideoCache] = None


def get_cache(max_age_days: int = 30) -> VideoCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = VideoCache(max_age_days=max_age_days)
    return _cache_instance


def check_cache(video_path: str, models_config: Dict) -> Tuple[bool, Optional[Dict]]:
    """
    Convenience function to check cache for a video.

    Args:
        video_path: Path to video file
        models_config: Model configuration dictionary

    Returns:
        Tuple of (cache_hit, result_or_None)
    """
    cache = get_cache()
    return cache.get(video_path, models_config)


def store_in_cache(video_path: str, models_config: Dict, result: Dict) -> bool:
    """
    Convenience function to store a result in cache.

    Args:
        video_path: Path to video file
        models_config: Model configuration dictionary
        result: Analysis result to cache

    Returns:
        True if stored successfully
    """
    cache = get_cache()
    return cache.put(video_path, models_config, result)
