"""
Centralized configuration schema using Pydantic.
Provides type-safe configuration with validation for all system settings.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
import os


class VideoSettings(BaseModel):
    """Video processing configuration."""

    min_duration_seconds: int = Field(
        default=10,
        ge=1,
        description="Minimum video duration in seconds"
    )
    max_duration_seconds: int = Field(
        default=300,
        le=600,
        description="Maximum video duration in seconds"
    )
    max_file_size_mb: int = Field(
        default=100,
        le=500,
        description="Maximum file size in megabytes"
    )
    supported_formats: List[str] = Field(
        default=[".mp4", ".mov", ".avi", ".webm", ".mkv"],
        description="Supported video file extensions"
    )
    frame_extraction_count: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Number of frames to extract for analysis"
    )
    max_resolution_height: int = Field(
        default=1080,
        description="Maximum resolution height for processing"
    )

    @model_validator(mode='after')
    def validate_duration_range(self):
        if self.min_duration_seconds >= self.max_duration_seconds:
            raise ValueError(
                "min_duration_seconds must be less than max_duration_seconds"
            )
        return self


class AudioSettings(BaseModel):
    """Audio processing configuration."""

    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels (1=mono, 2=stereo)"
    )
    bitrate: str = Field(
        default="64k",
        description="Audio bitrate for extraction"
    )
    format: str = Field(
        default="mp3",
        description="Audio format for extraction"
    )


class APISettings(BaseModel):
    """API configuration for OpenRouter."""

    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        description="API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Delay between retries in seconds"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute"
    )


class ModelDefaults(BaseModel):
    """Default model selections for each stage."""

    essence_model: str = Field(
        default="openai/gpt-4.1",
        description="Model for Sam Christensen essence analysis"
    )
    multimodal_model: str = Field(
        default="google/gemini-2.5-flash",
        description="Model for multimodal behavioral analysis"
    )
    audio_model: str = Field(
        default="google/gemini-2.5-flash",
        description="Model for audio/voice analysis"
    )
    liwc_model: str = Field(
        default="google/gemini-2.5-flash",
        description="Model for LIWC linguistic analysis"
    )
    synthesis_model: str = Field(
        default="openai/gpt-4.1",
        description="Model for FBI behavioral synthesis"
    )


class DatabaseSettings(BaseModel):
    """Database configuration."""

    db_path: str = Field(
        default="profiler_data.db",
        description="Path to SQLite database file"
    )
    backup_enabled: bool = Field(
        default=False,
        description="Enable automatic database backups"
    )
    backup_interval_hours: int = Field(
        default=24,
        ge=1,
        description="Backup interval in hours"
    )
    max_profiles_per_subject: int = Field(
        default=100,
        ge=1,
        description="Maximum number of profiles to keep per subject"
    )


class DownloadSettings(BaseModel):
    """Video download configuration for URL imports."""

    downloads_dir: str = Field(
        default="downloads",
        description="Directory for downloaded videos"
    )
    cleanup_age_hours: int = Field(
        default=24,
        ge=1,
        description="Delete downloaded files older than this many hours"
    )
    enable_youtube: bool = Field(
        default=True,
        description="Enable YouTube video imports"
    )
    enable_vimeo: bool = Field(
        default=True,
        description="Enable Vimeo video imports"
    )
    enable_twitter: bool = Field(
        default=True,
        description="Enable Twitter/X video imports"
    )
    enable_tiktok: bool = Field(
        default=True,
        description="Enable TikTok video imports"
    )


class SecuritySettings(BaseModel):
    """Security configuration."""

    encrypt_api_key: bool = Field(
        default=True,
        description="Encrypt API key at rest"
    )
    key_file_path: str = Field(
        default=".key",
        description="Path to encryption key file"
    )
    sanitize_paths: bool = Field(
        default=True,
        description="Sanitize file paths to prevent injection"
    )
    restrict_file_permissions: bool = Field(
        default=True,
        description="Set restrictive permissions on sensitive files"
    )


class UISettings(BaseModel):
    """User interface configuration."""

    server_host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    server_port: int = Field(
        default=7860,
        ge=1,
        le=65535,
        description="Server port"
    )
    share: bool = Field(
        default=False,
        description="Create public Gradio share link"
    )
    show_error: bool = Field(
        default=True,
        description="Show detailed error messages"
    )
    theme: str = Field(
        default="dark",
        description="UI theme (dark only, light mode is for sociopaths)"
    )

    @field_validator('theme')
    @classmethod
    def validate_theme(cls, v):
        if v.lower() != 'dark':
            raise ValueError("Only dark theme is supported. Light mode is for sociopaths.")
        return v.lower()


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_dir: str = Field(
        default="logs",
        description="Directory for log files"
    )
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        description="Number of log file backups to keep"
    )
    console_output: bool = Field(
        default=True,
        description="Enable console logging output"
    )
    file_output: bool = Field(
        default=True,
        description="Enable file logging output"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class AppConfig(BaseSettings):
    """
    Main application configuration.
    Loads from environment variables with PROFILER_ prefix.
    """

    # Sub-configurations
    video: VideoSettings = Field(default_factory=VideoSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    api: APISettings = Field(default_factory=APISettings)
    models: ModelDefaults = Field(default_factory=ModelDefaults)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    downloads: DownloadSettings = Field(default_factory=DownloadSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    ui: UISettings = Field(default_factory=UISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Root-level settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    app_name: str = Field(
        default="FBI Behavioral Profiler",
        description="Application name"
    )
    version: str = Field(
        default="1.0.0",
        description="Application version"
    )

    model_config = {
        "env_prefix": "PROFILER_",
        "env_nested_delimiter": "__",
        "extra": "ignore"
    }

    @classmethod
    def load_from_file(cls, path: str) -> "AppConfig":
        """
        Load configuration from a JSON or YAML file.

        Args:
            path: Path to configuration file

        Returns:
            AppConfig instance
        """
        import json

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(file_path, 'r') as f:
            if path.endswith('.json'):
                data = json.load(f)
            elif path.endswith(('.yaml', '.yml')):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                raise ValueError("Config file must be .json or .yaml/.yml")

        return cls(**data)

    def save_to_file(self, path: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            path: Output file path
        """
        import json

        data = self.model_dump()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_flat_dict(self) -> Dict[str, Any]:
        """
        Get a flattened dictionary of all settings.

        Returns:
            Dictionary with dot-notation keys
        """
        def flatten(d: dict, parent_key: str = '') -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return flatten(self.model_dump())


# Global config instance
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.
    Creates a new instance if none exists.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


def load_config(path: str = None) -> AppConfig:
    """
    Load configuration from file or create default.

    Args:
        path: Optional path to config file

    Returns:
        AppConfig instance
    """
    global _config_instance

    if path and Path(path).exists():
        _config_instance = AppConfig.load_from_file(path)
    else:
        _config_instance = AppConfig()

    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None
