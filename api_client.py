"""
OpenRouter API client for multimodal analysis.
Handles communication with various AI models via OpenRouter.
"""

import os
import time
import threading
import logging
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    Thread-safe implementation for controlling request rates.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum sustained requests per minute
            burst_size: Maximum burst size (bucket capacity)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size

        # Token bucket state
        self.tokens = float(burst_size)
        self.last_update = time.time()

        # Refill rate: tokens per second
        self.refill_rate = requests_per_minute / 60.0

        # Thread safety
        self._lock = threading.Lock()

        # Request tracking for logging
        self.total_requests = 0
        self.throttled_requests = 0

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire a token for making a request.
        Blocks until a token is available or timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._refill()

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.total_requests += 1
                    return True

                # Calculate wait time
                wait_time = (1.0 - self.tokens) / self.refill_rate

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed + wait_time > timeout:
                with self._lock:
                    self.throttled_requests += 1
                logger.warning(f"Rate limiter timeout after {elapsed:.1f}s")
                return False

            # Wait and try again
            logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
            time.sleep(min(wait_time, 1.0))

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.refill_rate
        )

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "throttled_requests": self.throttled_requests,
                "current_tokens": self.tokens,
                "requests_per_minute": self.requests_per_minute
            }

    def reset(self):
        """Reset rate limiter state."""
        with self._lock:
            self.tokens = float(self.burst_size)
            self.last_update = time.time()
            self.total_requests = 0
            self.throttled_requests = 0


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(requests_per_minute: int = 60) -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
    return _rate_limiter


class OpenRouterClient:
    """
    Client for interacting with OpenRouter API.
    Provides methods for sending multimodal requests with text, images, and audio.
    Includes built-in rate limiting to prevent API throttling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: bool = True,
        requests_per_minute: int = 60
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)
            rate_limit: Whether to enable rate limiting (default: True)
            requests_per_minute: Maximum requests per minute when rate limiting

        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Provide api_key parameter or set OPENROUTER_API_KEY environment variable."
            )

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        # Rate limiting
        self.rate_limit_enabled = rate_limit
        if rate_limit:
            self.rate_limiter = get_rate_limiter(requests_per_minute)
        else:
            self.rate_limiter = None

    def _apply_rate_limit(self, timeout: float = 60.0) -> bool:
        """
        Apply rate limiting before making a request.

        Args:
            timeout: Maximum time to wait for rate limit

        Returns:
            True if request can proceed, False if rate limited

        Raises:
            Exception: If rate limit timeout exceeded
        """
        if not self.rate_limit_enabled or self.rate_limiter is None:
            return True

        if not self.rate_limiter.acquire(timeout=timeout):
            raise Exception(
                "Rate limit exceeded. Too many requests in a short period. "
                "Please wait before retrying."
            )
        return True

    def analyze_with_vision(
        self,
        prompt: str,
        base64_images: List[str],
        model: str = "openai/gpt-4.1",
        max_tokens: int = 3000,
        temperature: float = 0.7,
        timeout: int = 120,
        response_format: Optional[dict] = None
    ) -> str:
        """
        Send vision analysis request to specified model via OpenRouter.

        Args:
            prompt: Text prompt for analysis
            base64_images: List of base64-encoded JPEG strings
            model: Model ID to use (default: openai/gpt-4.1)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            response_format: Optional response format for structured JSON output

        Returns:
            Model response text

        Raises:
            Exception: If API request fails
        """
        logger.info(f"Vision analysis with model: {model}")
        return self._send_multimodal_request(
            model=model,
            prompt=prompt,
            base64_images=base64_images,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format
        )

    def analyze_with_multimodal(
        self,
        prompt: str,
        base64_images: List[str] = None,
        base64_audio: str = None,
        base64_video: str = None,
        model: str = "google/gemini-2.5-flash",
        max_tokens: int = 3000,
        temperature: float = 0.7,
        timeout: int = 120,
        response_format: Optional[dict] = None
    ) -> str:
        """
        Send multimodal request (images + audio) to specified model via OpenRouter.

        Args:
            prompt: Text prompt for analysis
            base64_images: List of base64-encoded JPEG strings (optional)
            base64_audio: Base64-encoded audio string (optional)
            base64_video: Base64-encoded video file (optional)
            model: Model ID to use (default: google/gemini-2.5-flash)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            response_format: Optional response format for structured JSON output

        Returns:
            Model response text

        Raises:
            Exception: If API request fails
        """
        logger.info(f"Multimodal analysis with model: {model}")
        return self._send_multimodal_request(
            model=model,
            prompt=prompt,
            base64_images=base64_images or [],
            base64_audio=base64_audio,
            base64_video=base64_video,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format
        )

    def analyze_audio(
        self,
        prompt: str,
        base64_audio: str,
        model: str = "google/gemini-2.5-flash",
        max_tokens: int = 3000,
        temperature: float = 0.7,
        timeout: int = 120,
        response_format: Optional[dict] = None
    ) -> str:
        """
        Send audio-only analysis request to specified model via OpenRouter.

        Args:
            prompt: Text prompt for analysis
            base64_audio: Base64-encoded audio string
            model: Model ID to use (must support audio - Gemini models)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            response_format: Optional response format for structured JSON output

        Returns:
            Model response text

        Raises:
            Exception: If API request fails
        """
        logger.info(f"Audio analysis with model: {model}")
        return self._send_multimodal_request(
            model=model,
            prompt=prompt,
            base64_audio=base64_audio,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format
        )

    def synthesize_text(
        self,
        prompt: str,
        previous_analyses: str,
        model: str = "openai/gpt-4.1",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        timeout: int = 120,
        response_format: Optional[dict] = None
    ) -> str:
        """
        Send text-only synthesis request to specified model via OpenRouter.
        Used for FBI behavioral synthesis combining previous analyses.

        Args:
            prompt: System prompt for synthesis
            previous_analyses: Combined text from previous analyses
            model: Model ID to use (default: openai/gpt-4.1)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            response_format: Optional response format for structured JSON output

        Returns:
            Model response text

        Raises:
            Exception: If API request fails
        """
        # Apply rate limiting
        self._apply_rate_limit(timeout=timeout)

        logger.info(f"Text synthesis with model: {model}")
        try:
            # Build request parameters
            request_params = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": previous_analyses
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout
            }

            # Add response_format if provided
            if response_format is not None:
                request_params["response_format"] = response_format
                logger.debug(f"Using structured output for synthesis")

            response = self.client.chat.completions.create(**request_params)

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"OpenRouter API error ({model} synthesis): {str(e)}")

    def _send_multimodal_request(
        self,
        model: str,
        prompt: str,
        base64_images: List[str] = None,
        base64_audio: str = None,
        base64_video: str = None,
        max_tokens: int = 3000,
        temperature: float = 0.7,
        timeout: int = 120,
        response_format: Optional[dict] = None
    ) -> str:
        """
        Internal method to send multimodal request with images, audio, and/or video.

        Args:
            model: Model identifier (e.g., "openai/gpt-5.1")
            prompt: Text prompt
            base64_images: List of base64-encoded images (optional)
            base64_audio: Base64-encoded audio (optional)
            base64_video: Base64-encoded video (optional)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            response_format: Optional response format for structured output

        Returns:
            Model response text

        Raises:
            Exception: If API request fails
        """
        # Apply rate limiting
        self._apply_rate_limit(timeout=timeout)

        try:
            # Build content array with text
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            # Add video if provided (takes precedence over images/audio)
            if base64_video:
                # Determine video mime type - try mp4 first
                content.append({
                    "type": "image_url",  # OpenRouter uses image_url for video too
                    "image_url": {
                        "url": f"data:video/mp4;base64,{base64_video}"
                    }
                })
            else:
                # Add images if provided
                if base64_images:
                    for base64_image in base64_images:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # Use high detail for better analysis
                            }
                        })

                # Add audio if provided
                if base64_audio:
                    content.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": "mp3"
                        }
                    })

            # Build request parameters
            request_params = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout
            }

            # Add response_format if provided (for structured output)
            if response_format is not None:
                request_params["response_format"] = response_format
                logger.debug(f"Using structured output with schema: {response_format.get('json_schema', {}).get('name', 'unknown')}")

            # Make API request
            response = self.client.chat.completions.create(**request_params)

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"OpenRouter API error ({model}): {str(e)}")

