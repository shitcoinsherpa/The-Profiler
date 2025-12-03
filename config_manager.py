"""
Configuration manager for secure API key storage.
Handles encryption/decryption of sensitive data.
"""

import os
import stat
import base64
import logging
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages encrypted storage of API keys and configuration.
    """

    def __init__(self, config_file: str = ".env"):
        """
        Initialize config manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.key_file = ".key"

        # Ensure config file exists
        if not os.path.exists(self.config_file):
            self._create_default_config()

        # Initialize encryption key
        self.cipher = self._get_or_create_cipher()

    def _create_default_config(self):
        """Create default .env file if it doesn't exist."""
        default_content = """# OpenRouter API Configuration
# Get your API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=
"""
        with open(self.config_file, 'w') as f:
            f.write(default_content)

    def _get_or_create_cipher(self):
        """
        Get or create encryption cipher.
        Uses a machine-specific key for encryption.
        Sets restrictive file permissions on key file.
        """
        if os.path.exists(self.key_file):
            # Load existing key
            with open(self.key_file, 'rb') as f:
                key = f.read()
            logger.debug("Loaded existing encryption key")
        else:
            # Generate new Fernet key (simple and correct)
            key = Fernet.generate_key()

            # Save key with restrictive permissions
            with open(self.key_file, 'wb') as f:
                f.write(key)

            # Set restrictive file permissions (owner read/write only)
            # On Windows, this sets the file to not be world-readable
            try:
                if os.name == 'nt':  # Windows
                    import subprocess
                    # Use icacls to restrict permissions on Windows
                    subprocess.run(
                        ['icacls', self.key_file, '/inheritance:r', '/grant:r', f'{os.getlogin()}:F'],
                        capture_output=True,
                        check=False
                    )
                else:  # Unix/Linux/Mac
                    os.chmod(self.key_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
                logger.info("Created new encryption key with restricted permissions")
            except Exception as e:
                logger.warning(f"Could not set restrictive permissions on key file: {e}")

        return Fernet(key)

    def save_api_key(self, api_key: str) -> bool:
        """
        Save API key to configuration file with encryption.

        Args:
            api_key: OpenRouter API key

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Encrypt the API key
            encrypted_key = self.cipher.encrypt(api_key.encode()).decode()

            # Read current config
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    lines = f.readlines()
            else:
                lines = []

            # Update or add API key line
            updated = False
            new_lines = []

            for line in lines:
                if line.strip().startswith('OPENROUTER_API_KEY='):
                    new_lines.append(f'OPENROUTER_API_KEY_ENCRYPTED={encrypted_key}\n')
                    updated = True
                elif line.strip().startswith('OPENROUTER_API_KEY_ENCRYPTED='):
                    new_lines.append(f'OPENROUTER_API_KEY_ENCRYPTED={encrypted_key}\n')
                    updated = True
                else:
                    new_lines.append(line)

            # If not found, add it
            if not updated:
                new_lines.append(f'\nOPENROUTER_API_KEY_ENCRYPTED={encrypted_key}\n')

            # Write back to file
            with open(self.config_file, 'w') as f:
                f.writelines(new_lines)

            # Also set in environment for immediate use
            os.environ['OPENROUTER_API_KEY'] = api_key

            return True

        except Exception as e:
            print(f"Error saving API key: {e}")
            return False

    def load_api_key(self) -> str:
        """
        Load and decrypt API key from configuration file.

        Returns:
            Decrypted API key or empty string if not found
        """
        try:
            # First check environment variable (unencrypted - for backward compatibility)
            if os.getenv('OPENROUTER_API_KEY'):
                return os.getenv('OPENROUTER_API_KEY')

            # Read config file
            if not os.path.exists(self.config_file):
                return ""

            with open(self.config_file, 'r') as f:
                lines = f.readlines()

            # Look for encrypted key
            for line in lines:
                if line.strip().startswith('OPENROUTER_API_KEY_ENCRYPTED='):
                    encrypted_key = line.split('=', 1)[1].strip()
                    if encrypted_key:
                        # Decrypt
                        decrypted = self.cipher.decrypt(encrypted_key.encode()).decode()
                        # Set in environment for use
                        os.environ['OPENROUTER_API_KEY'] = decrypted
                        return decrypted

            # Fallback: check for unencrypted key
            for line in lines:
                if line.strip().startswith('OPENROUTER_API_KEY='):
                    key = line.split('=', 1)[1].strip()
                    if key and key != 'your_openrouter_api_key_here':
                        os.environ['OPENROUTER_API_KEY'] = key
                        return key

            return ""

        except Exception as e:
            print(f"Error loading API key: {e}")
            return ""

    def has_api_key(self) -> bool:
        """
        Check if API key is configured.

        Returns:
            True if API key exists, False otherwise
        """
        key = self.load_api_key()
        return bool(key and key.strip())

    def test_api_key(self, api_key: str = None) -> tuple[bool, str]:
        """
        Test if API key is valid by making a minimal API call.

        Args:
            api_key: Optional API key to test (uses saved key if not provided)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if api_key is None:
            api_key = self.load_api_key()

        if not api_key:
            return False, "No API key provided"

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )

            # Make a minimal test request
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",  # Use cheapest model for testing
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )

            return True, "✓ API key is valid and working"

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                return False, "✗ Invalid API key - authentication failed"
            elif "402" in error_msg or "credits" in error_msg.lower():
                return False, "✗ API key valid but no credits available"
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                return False, "✓ API key valid (rate limited - this is OK)"
            else:
                return False, f"✗ Connection error: {error_msg}"

    def clear_api_key(self) -> bool:
        """
        Clear the saved API key.

        Returns:
            True if cleared successfully
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    if not line.strip().startswith('OPENROUTER_API_KEY'):
                        new_lines.append(line)

                with open(self.config_file, 'w') as f:
                    f.writelines(new_lines)
                    f.write('\nOPENROUTER_API_KEY=\n')

            # Clear from environment
            if 'OPENROUTER_API_KEY' in os.environ:
                del os.environ['OPENROUTER_API_KEY']

            return True

        except Exception as e:
            print(f"Error clearing API key: {e}")
            return False
