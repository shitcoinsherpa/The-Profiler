"""
Audio transcription module for extracting speech text from video/audio.
Uses Gemini models for accurate speech-to-text conversion.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Transcription prompt for Gemini
TRANSCRIPTION_PROMPT = """You are a professional transcriptionist. Transcribe the audio content exactly as spoken.

INSTRUCTIONS:
1. Transcribe ALL spoken words verbatim
2. Include speaker changes if multiple speakers are detected (mark as [Speaker 1], [Speaker 2], etc.)
3. Note significant non-verbal audio cues in brackets: [laughs], [sighs], [pause], [unclear]
4. Preserve filler words (um, uh, like, you know) as they appear
5. Use proper punctuation based on speech patterns
6. If audio quality is poor in sections, mark as [inaudible]
7. Include timestamps at natural paragraph breaks in format [MM:SS]

OUTPUT FORMAT:
Return the transcription in the following structure:

TRANSCRIPT:
[The full verbatim transcription with speaker labels and timestamps]

SUMMARY:
[2-3 sentence summary of what was discussed]

SPEAKERS:
[List detected speakers and brief description if identifiable]

AUDIO QUALITY:
[Rate audio quality: Excellent/Good/Fair/Poor and note any issues]

WORD COUNT: [approximate number of words spoken]
DURATION: [estimated speaking duration]
"""


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    transcript: str
    summary: str
    speakers: list
    word_count: int
    audio_quality: str
    raw_response: str
    success: bool
    error: Optional[str] = None


def transcribe_audio(
    base64_audio: str,
    api_client,
    model: str = "google/gemini-2.5-flash",
    max_tokens: int = 8000,
    timeout: int = 180
) -> TranscriptionResult:
    """
    Transcribe audio using Gemini model.

    Args:
        base64_audio: Base64-encoded audio data
        api_client: OpenRouterClient instance
        model: Model to use (must support audio)
        max_tokens: Maximum response tokens
        timeout: Request timeout in seconds

    Returns:
        TranscriptionResult with transcript and metadata
    """
    if not base64_audio:
        return TranscriptionResult(
            transcript="",
            summary="No audio provided",
            speakers=[],
            word_count=0,
            audio_quality="N/A",
            raw_response="",
            success=False,
            error="No audio data provided"
        )

    try:
        logger.info(f"Starting transcription with model: {model}")

        # Send transcription request
        response = api_client.analyze_audio(
            prompt=TRANSCRIPTION_PROMPT,
            base64_audio=base64_audio,
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more accurate transcription
            timeout=timeout
        )

        # Parse the response
        result = parse_transcription_response(response)
        result.raw_response = response
        result.success = True

        logger.info(f"Transcription complete: {result.word_count} words")
        return result

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return TranscriptionResult(
            transcript="",
            summary="",
            speakers=[],
            word_count=0,
            audio_quality="",
            raw_response="",
            success=False,
            error=str(e)
        )


def parse_transcription_response(response: str) -> TranscriptionResult:
    """
    Parse the structured transcription response.

    Args:
        response: Raw response from the model

    Returns:
        Parsed TranscriptionResult
    """
    transcript = ""
    summary = ""
    speakers = []
    word_count = 0
    audio_quality = "Unknown"

    lines = response.split('\n')
    current_section = None
    section_content = []

    for line in lines:
        line_stripped = line.strip()

        # Detect section headers
        if line_stripped.startswith('TRANSCRIPT:'):
            if current_section and section_content:
                _save_section(current_section, section_content, locals())
            current_section = 'transcript'
            section_content = []
            # Check if content is on the same line
            after_colon = line_stripped[11:].strip()
            if after_colon:
                section_content.append(after_colon)
        elif line_stripped.startswith('SUMMARY:'):
            if current_section and section_content:
                _save_section(current_section, section_content, locals())
            current_section = 'summary'
            section_content = []
            after_colon = line_stripped[8:].strip()
            if after_colon:
                section_content.append(after_colon)
        elif line_stripped.startswith('SPEAKERS:'):
            if current_section and section_content:
                _save_section(current_section, section_content, locals())
            current_section = 'speakers'
            section_content = []
        elif line_stripped.startswith('AUDIO QUALITY:'):
            if current_section and section_content:
                _save_section(current_section, section_content, locals())
            current_section = 'quality'
            section_content = []
            after_colon = line_stripped[14:].strip()
            if after_colon:
                section_content.append(after_colon)
        elif line_stripped.startswith('WORD COUNT:'):
            try:
                count_str = line_stripped[11:].strip()
                # Extract just the number
                import re
                numbers = re.findall(r'\d+', count_str)
                if numbers:
                    word_count = int(numbers[0])
            except:
                pass
        elif line_stripped.startswith('DURATION:'):
            pass  # We can extract this if needed
        elif current_section:
            section_content.append(line)

    # Save the last section
    if current_section and section_content:
        content = '\n'.join(section_content).strip()
        if current_section == 'transcript':
            transcript = content
        elif current_section == 'summary':
            summary = content
        elif current_section == 'speakers':
            speakers = [s.strip() for s in content.split('\n') if s.strip() and s.strip() != '-']
        elif current_section == 'quality':
            audio_quality = content

    # If parsing failed, use the whole response as transcript
    if not transcript:
        transcript = response
        # Estimate word count
        word_count = len(response.split())

    return TranscriptionResult(
        transcript=transcript,
        summary=summary,
        speakers=speakers,
        word_count=word_count if word_count > 0 else len(transcript.split()),
        audio_quality=audio_quality,
        raw_response="",
        success=True
    )


def _save_section(section: str, content: list, local_vars: dict):
    """Helper to save section content."""
    content_str = '\n'.join(content).strip()
    if section == 'transcript':
        local_vars['transcript'] = content_str
    elif section == 'summary':
        local_vars['summary'] = content_str
    elif section == 'speakers':
        local_vars['speakers'] = [s.strip() for s in content_str.split('\n') if s.strip()]
    elif section == 'quality':
        local_vars['audio_quality'] = content_str


def format_transcript_for_display(result: TranscriptionResult) -> str:
    """
    Format transcription result for UI display.

    Args:
        result: TranscriptionResult object

    Returns:
        Formatted string for display
    """
    if not result.success:
        return f"Transcription failed: {result.error}"

    output = []

    output.append("=" * 60)
    output.append("AUDIO TRANSCRIPTION")
    output.append("=" * 60)
    output.append("")

    if result.audio_quality:
        output.append(f"Audio Quality: {result.audio_quality}")
    output.append(f"Word Count: ~{result.word_count}")

    if result.speakers:
        output.append(f"Speakers Detected: {len(result.speakers)}")
        for speaker in result.speakers:
            output.append(f"  - {speaker}")

    output.append("")
    output.append("-" * 60)
    output.append("TRANSCRIPT")
    output.append("-" * 60)
    output.append("")
    output.append(result.transcript)

    if result.summary:
        output.append("")
        output.append("-" * 60)
        output.append("SUMMARY")
        output.append("-" * 60)
        output.append("")
        output.append(result.summary)

    return '\n'.join(output)


class TranscriptionCache:
    """Simple in-memory cache for transcriptions to avoid re-processing."""

    def __init__(self):
        self._cache: Dict[str, TranscriptionResult] = {}

    def get(self, audio_hash: str) -> Optional[TranscriptionResult]:
        """Get cached transcription."""
        return self._cache.get(audio_hash)

    def put(self, audio_hash: str, result: TranscriptionResult):
        """Cache a transcription result."""
        self._cache[audio_hash] = result

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global transcription cache
_transcription_cache = TranscriptionCache()


def get_transcription_cache() -> TranscriptionCache:
    """Get the global transcription cache."""
    return _transcription_cache
