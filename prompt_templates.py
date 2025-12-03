"""
Custom prompt templates system for the behavioral profiling application.
Allows users to save, load, and customize prompts for each analysis stage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from prompts import (
    SAM_CHRISTENSEN_PROMPT,
    GEMINI_COMPREHENSIVE_PROMPT,
    AUDIO_ANALYSIS_PROMPT,
    LIWC_ANALYSIS_PROMPT,
    FBI_SYNTHESIS_PROMPT
)

logger = logging.getLogger(__name__)

# Template storage directory
TEMPLATES_DIR = Path(__file__).parent / ".templates"
TEMPLATES_DIR.mkdir(exist_ok=True)


# Stage identifiers
PROMPT_STAGES = {
    "essence": "Sam Christensen Visual Essence",
    "multimodal": "Multimodal Behavioral Analysis",
    "audio": "Audio/Voice Analysis",
    "liwc": "LIWC Linguistic Analysis",
    "synthesis": "FBI Behavioral Synthesis"
}

# Default prompts mapping
DEFAULT_PROMPTS = {
    "essence": SAM_CHRISTENSEN_PROMPT,
    "multimodal": GEMINI_COMPREHENSIVE_PROMPT,
    "audio": AUDIO_ANALYSIS_PROMPT,
    "liwc": LIWC_ANALYSIS_PROMPT,
    "synthesis": FBI_SYNTHESIS_PROMPT
}


@dataclass
class PromptTemplate:
    """Represents a custom prompt template."""
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    stage: str = ""  # essence, multimodal, audio, liwc, synthesis
    prompt_text: str = ""
    created_at: str = ""
    updated_at: str = ""
    is_default: bool = False
    is_active: bool = False


class PromptTemplateManager:
    """
    Manages custom prompt templates with file-based storage.
    Allows saving, loading, and switching between templates.
    """

    def __init__(self, templates_dir: str = None):
        """
        Initialize the template manager.

        Args:
            templates_dir: Custom directory for template storage
        """
        self.templates_dir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        self.templates_dir.mkdir(exist_ok=True)

        self.index_file = self.templates_dir / "templates_index.json"
        self.active_file = self.templates_dir / "active_templates.json"

        self._load_index()
        self._load_active_templates()
        self._ensure_defaults()

    def _load_index(self):
        """Load the templates index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load templates index: {e}")
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """Save the templates index to disk."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save templates index: {e}")

    def _load_active_templates(self):
        """Load which templates are currently active for each stage."""
        if self.active_file.exists():
            try:
                with open(self.active_file, 'r', encoding='utf-8') as f:
                    self.active_templates = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load active templates: {e}")
                self.active_templates = {}
        else:
            self.active_templates = {}

        # Ensure all stages have an active template (default to "default")
        for stage in PROMPT_STAGES.keys():
            if stage not in self.active_templates:
                self.active_templates[stage] = "default"

    def _save_active_templates(self):
        """Save active template selections to disk."""
        try:
            with open(self.active_file, 'w', encoding='utf-8') as f:
                json.dump(self.active_templates, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save active templates: {e}")

    def _ensure_defaults(self):
        """Ensure default templates exist for all stages."""
        for stage, stage_name in PROMPT_STAGES.items():
            default_id = f"default_{stage}"

            if default_id not in self.index:
                # Create default template entry
                self.index[default_id] = {
                    'id': default_id,
                    'name': f"Default {stage_name}",
                    'description': f"Built-in default prompt for {stage_name}",
                    'stage': stage,
                    'is_default': True,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }

        self._save_index()

    def _generate_template_id(self, name: str, stage: str) -> str:
        """Generate a unique template ID."""
        base_id = f"{stage}_{name.lower().replace(' ', '_')}"
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '_')

        # Ensure uniqueness
        counter = 0
        template_id = base_id
        while template_id in self.index:
            counter += 1
            template_id = f"{base_id}_{counter}"

        return template_id

    def get_default_prompt(self, stage: str) -> str:
        """Get the built-in default prompt for a stage."""
        return DEFAULT_PROMPTS.get(stage, "")

    def get_active_prompt(self, stage: str) -> str:
        """
        Get the currently active prompt for a stage.

        Args:
            stage: The analysis stage

        Returns:
            The prompt text to use
        """
        template_id = self.active_templates.get(stage, "default")

        if template_id == "default" or template_id.startswith("default_"):
            return self.get_default_prompt(stage)

        template = self.get_template(template_id)
        if template and template.prompt_text:
            return template.prompt_text

        # Fallback to default
        return self.get_default_prompt(stage)

    def get_active_prompts(self) -> Dict[str, str]:
        """Get all active prompts for all stages."""
        return {stage: self.get_active_prompt(stage) for stage in PROMPT_STAGES.keys()}

    def set_active_template(self, stage: str, template_id: str) -> bool:
        """
        Set the active template for a stage.

        Args:
            stage: The analysis stage
            template_id: ID of the template to activate

        Returns:
            True if successful
        """
        if stage not in PROMPT_STAGES:
            return False

        # Verify template exists (or is default)
        if template_id != "default" and not template_id.startswith("default_"):
            if template_id not in self.index:
                return False

        self.active_templates[stage] = template_id
        self._save_active_templates()

        logger.info(f"Set active template for {stage}: {template_id}")
        return True

    def save_template(
        self,
        name: str,
        stage: str,
        prompt_text: str,
        description: str = "",
        template_id: str = None
    ) -> Tuple[bool, str, Optional[PromptTemplate]]:
        """
        Save a custom prompt template.

        Args:
            name: Template name
            stage: Analysis stage this template is for
            prompt_text: The prompt text content
            description: Optional description
            template_id: Optional ID for updating existing template

        Returns:
            Tuple of (success, message, template)
        """
        if not name or not name.strip():
            return False, "Template name is required", None

        if stage not in PROMPT_STAGES:
            return False, f"Invalid stage: {stage}", None

        if not prompt_text or not prompt_text.strip():
            return False, "Prompt text is required", None

        name = name.strip()
        prompt_text = prompt_text.strip()
        now = datetime.now().isoformat()

        # Check if updating existing or creating new
        if template_id and template_id in self.index:
            # Update existing
            if self.index[template_id].get('is_default', False):
                return False, "Cannot modify default templates", None

            self.index[template_id]['name'] = name
            self.index[template_id]['description'] = description
            self.index[template_id]['updated_at'] = now
        else:
            # Create new
            template_id = self._generate_template_id(name, stage)
            self.index[template_id] = {
                'id': template_id,
                'name': name,
                'description': description,
                'stage': stage,
                'is_default': False,
                'created_at': now,
                'updated_at': now
            }

        # Save the prompt content to file
        try:
            prompt_file = self.templates_dir / f"{template_id}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_text)

            self._save_index()

            template = PromptTemplate(
                id=template_id,
                name=name,
                description=description,
                stage=stage,
                prompt_text=prompt_text,
                created_at=self.index[template_id]['created_at'],
                updated_at=now,
                is_default=False,
                is_active=self.active_templates.get(stage) == template_id
            )

            logger.info(f"Saved template: {name} ({template_id})")
            return True, f"Template '{name}' saved successfully", template

        except Exception as e:
            logger.error(f"Failed to save template: {e}")
            return False, f"Failed to save template: {str(e)}", None

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: The template ID

        Returns:
            PromptTemplate object or None
        """
        if template_id not in self.index:
            return None

        info = self.index[template_id]
        stage = info.get('stage', '')

        # Get prompt text
        if info.get('is_default', False):
            prompt_text = self.get_default_prompt(stage)
        else:
            prompt_file = self.templates_dir / f"{template_id}.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_text = f.read()
            else:
                prompt_text = ""

        return PromptTemplate(
            id=template_id,
            name=info.get('name', ''),
            description=info.get('description', ''),
            stage=stage,
            prompt_text=prompt_text,
            created_at=info.get('created_at', ''),
            updated_at=info.get('updated_at', ''),
            is_default=info.get('is_default', False),
            is_active=self.active_templates.get(stage) == template_id
        )

    def list_templates(self, stage: str = None) -> List[PromptTemplate]:
        """
        List all templates, optionally filtered by stage.

        Args:
            stage: Optional stage to filter by

        Returns:
            List of PromptTemplate objects
        """
        templates = []

        for template_id, info in self.index.items():
            if stage and info.get('stage') != stage:
                continue

            template_stage = info.get('stage', '')
            templates.append(PromptTemplate(
                id=template_id,
                name=info.get('name', ''),
                description=info.get('description', ''),
                stage=template_stage,
                prompt_text="",  # Don't load full text for listing
                created_at=info.get('created_at', ''),
                updated_at=info.get('updated_at', ''),
                is_default=info.get('is_default', False),
                is_active=self.active_templates.get(template_stage) == template_id
            ))

        # Sort: active first, then defaults, then by name
        templates.sort(key=lambda t: (not t.is_active, not t.is_default, t.name.lower()))

        return templates

    def get_templates_for_dropdown(self, stage: str) -> List[Tuple[str, str]]:
        """
        Get template choices for a Gradio dropdown.

        Args:
            stage: The analysis stage

        Returns:
            List of (display_name, template_id) tuples
        """
        templates = self.list_templates(stage)
        choices = []

        for template in templates:
            prefix = ""
            if template.is_active:
                prefix = "* "
            elif template.is_default:
                prefix = "[Default] "

            display = f"{prefix}{template.name}"
            choices.append((display, template.id))

        return choices

    def delete_template(self, template_id: str) -> Tuple[bool, str]:
        """
        Delete a custom template.

        Args:
            template_id: The template ID to delete

        Returns:
            Tuple of (success, message)
        """
        if template_id not in self.index:
            return False, "Template not found"

        if self.index[template_id].get('is_default', False):
            return False, "Cannot delete default templates"

        stage = self.index[template_id].get('stage', '')

        # If this is the active template, revert to default
        if self.active_templates.get(stage) == template_id:
            self.active_templates[stage] = "default"
            self._save_active_templates()

        # Remove template file
        prompt_file = self.templates_dir / f"{template_id}.txt"
        if prompt_file.exists():
            try:
                prompt_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete template file: {e}")

        # Remove from index
        name = self.index[template_id].get('name', template_id)
        del self.index[template_id]
        self._save_index()

        logger.info(f"Deleted template: {template_id}")
        return True, f"Template '{name}' deleted"

    def duplicate_template(self, template_id: str, new_name: str = None) -> Tuple[bool, str, Optional[str]]:
        """
        Create a copy of an existing template.

        Args:
            template_id: Source template ID
            new_name: Name for the copy (optional)

        Returns:
            Tuple of (success, message, new_template_id)
        """
        source = self.get_template(template_id)
        if not source:
            return False, "Source template not found", None

        if not new_name:
            new_name = f"{source.name} (Copy)"

        success, message, new_template = self.save_template(
            name=new_name,
            stage=source.stage,
            prompt_text=source.prompt_text,
            description=source.description
        )

        if success and new_template:
            return True, f"Created copy: {new_name}", new_template.id

        return False, message, None

    def export_template(self, template_id: str) -> Optional[Dict]:
        """
        Export a template as a dictionary for JSON export.

        Args:
            template_id: The template ID

        Returns:
            Dictionary representation of the template
        """
        template = self.get_template(template_id)
        if not template:
            return None

        return asdict(template)

    def import_template(self, data: Dict) -> Tuple[bool, str]:
        """
        Import a template from a dictionary.

        Args:
            data: Template data dictionary

        Returns:
            Tuple of (success, message)
        """
        required_fields = ['name', 'stage', 'prompt_text']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        success, message, _ = self.save_template(
            name=data['name'],
            stage=data['stage'],
            prompt_text=data['prompt_text'],
            description=data.get('description', '')
        )

        return success, message

    def reset_to_defaults(self, stage: str = None) -> int:
        """
        Reset active templates to defaults.

        Args:
            stage: Specific stage to reset, or None for all

        Returns:
            Number of stages reset
        """
        count = 0

        if stage:
            if stage in PROMPT_STAGES:
                self.active_templates[stage] = f"default_{stage}"
                count = 1
        else:
            for s in PROMPT_STAGES.keys():
                self.active_templates[s] = f"default_{s}"
                count += 1

        self._save_active_templates()
        logger.info(f"Reset {count} stage(s) to default templates")

        return count


# Global instance
_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager


def get_active_prompts() -> Dict[str, str]:
    """Convenience function to get all active prompts."""
    return get_template_manager().get_active_prompts()


def get_prompt_for_stage(stage: str) -> str:
    """Convenience function to get the active prompt for a stage."""
    return get_template_manager().get_active_prompt(stage)
