"""
FBI-Style Behavioral Profiling System
Gradio application for multimodal AI-powered psychological assessment.
"""

import warnings
import asyncio
import os

# Suppress noisy Windows asyncio connection reset warnings
warnings.filterwarnings("ignore", message=".*ConnectionResetError.*")
if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize centralized logging before other imports
from infra.logger import setup_logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=True,
    console=True
)

import gradio as gr
import json
import tempfile
from datetime import datetime
from profiler import BehavioralProfiler, ModelSelection, run_dev_meta_analysis
from config.config_manager import ConfigManager
from config.models_config import (
    ESSENCE_MODEL_CHOICES,
    MULTIMODAL_MODEL_CHOICES,
    AUDIO_MODEL_CHOICES,
    LIWC_MODEL_CHOICES,
    SYNTHESIS_MODEL_CHOICES,
    get_default_model_for_stage,
    DEV_META_MODELS,
    DEFAULT_DEV_META_MODEL
)
from media.video_downloader import download_video, is_valid_url
from infra.database import get_database
from output.pdf_generator import generate_pdf_report, REPORTLAB_AVAILABLE
from infra.cache_manager import get_cache


# Create FBI-themed Gradio theme based on Glass
def create_fbi_theme():
    """Create custom FBI-themed Gradio theme based on Glass."""
    return gr.themes.Glass(
        primary_hue=gr.themes.Color(
            c50="#e8f4ff",
            c100="#d1e9ff",
            c200="#a3d3ff",
            c300="#75bdff",
            c400="#4a9eff",  # Main FBI blue
            c500="#2d8aef",
            c600="#1a6fc7",
            c700="#155a9f",
            c800="#104878",
            c900="#0a3050",
            c950="#051828",
        ),
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        # Dark mode optimizations
        body_background_fill="#0a0e1a",
        body_background_fill_dark="#0a0e1a",
        background_fill_primary="#0f1320",
        background_fill_primary_dark="#0f1320",
        background_fill_secondary="#1e2842",
        background_fill_secondary_dark="#1e2842",

        # Border styling
        border_color_primary="#2d3a5f",
        border_color_primary_dark="#2d3a5f",
        block_border_width="1px",

        # Button styling
        button_primary_background_fill="linear-gradient(135deg, #4a9eff 0%, #2d6bbf 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #4a9eff 0%, #2d6bbf 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #5eaaff 0%, #3d7bcf 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #5eaaff 0%, #3d7bcf 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",

        # Input styling
        input_background_fill="#0a0e1a",
        input_background_fill_dark="#0a0e1a",
        input_border_color="#2d3a5f",
        input_border_color_dark="#2d3a5f",

        # Text colors
        body_text_color="#e8eaf0",
        body_text_color_dark="#e8eaf0",
        body_text_color_subdued="#6b7280",
        body_text_color_subdued_dark="#6b7280",

        # Panel/block styling
        block_background_fill="#1e2842",
        block_background_fill_dark="#1e2842",
        panel_background_fill="#0f1320",
        panel_background_fill_dark="#0f1320",

        # Shadows and effects
        shadow_spread="4px",
        block_shadow="0 2px 8px rgba(0,0,0,0.3)",
        block_shadow_dark="0 2px 8px rgba(0,0,0,0.5)",
    )


# Minimal custom CSS - only for elements that need special handling
MINIMAL_CSS = """
/* Header styling */
.app-header {
    text-align: center;
    padding: 20px 0;
    margin-bottom: 10px;
    border-bottom: 1px solid #2d3a5f;
}

.app-title {
    font-size: 28px;
    font-weight: 700;
    color: #4a9eff;
    margin: 0 0 8px 0;
    letter-spacing: 1px;
}

.app-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin: 0;
}

.app-attribution {
    font-size: 12px;
    color: #4a9eff;
    margin-top: 8px;
}

.app-attribution a {
    color: #4a9eff;
    text-decoration: none;
}

.app-attribution a:hover {
    text-decoration: underline;
}

/* Section headers */
.section-header {
    color: #4a9eff;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 20px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d3a5f;
}

/* Progress visualization */
.progress-container {
    background-color: #1e2842;
    border: 1px solid #2d3a5f;
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}

.progress-steps {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    position: relative;
}

.progress-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    z-index: 1;
}

.step-circle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #0a0e1a;
    border: 3px solid #2d3a5f;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 14px;
    color: #6b7280;
    transition: all 0.3s ease;
}

.step-circle.active {
    border-color: #4a9eff;
    color: #4a9eff;
    box-shadow: 0 0 12px rgba(74, 158, 255, 0.4);
}

.step-circle.completed {
    background-color: #4a9eff;
    border-color: #4a9eff;
    color: #ffffff;
}

.step-label {
    margin-top: 8px;
    font-size: 10px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-align: center;
    max-width: 80px;
}

.step-label.active {
    color: #4a9eff;
}

.progress-line {
    position: absolute;
    top: 18px;
    left: 36px;
    right: 36px;
    height: 3px;
    background-color: #2d3a5f;
    z-index: 0;
}

.progress-line-fill {
    height: 100%;
    background: linear-gradient(90deg, #4a9eff, #2d6bbf);
    transition: width 0.5s ease;
}

.progress-bar-container {
    background-color: #0a0e1a;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 12px;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #4a9eff, #5eaaff);
    transition: width 0.3s ease;
}

.progress-percentage {
    text-align: right;
    font-size: 12px;
    color: #4a9eff;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Disclaimer/warning */
.disclaimer-box {
    background-color: rgba(255, 149, 0, 0.1);
    border: 1px solid #ff9500;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 16px 0;
    text-align: center;
}

.disclaimer-text {
    color: #ff9500;
    font-size: 12px;
    margin: 0;
}

/* Chart containers */
.chart-container {
    background-color: #0f1320;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

/* Monospace output text */
textarea {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

/* Force scrollability on FBI, Confidence, and Subject ID output boxes */
#fbi-output-box textarea,
#confidence-output-box textarea,
#subject-id-output-box textarea {
    overflow-y: scroll !important;
    max-height: 500px !important;
    min-height: 200px !important;
}

#fbi-output-box,
#confidence-output-box,
#subject-id-output-box {
    max-height: 550px !important;
    overflow-y: auto !important;
}

/* Force scrollability on all textareas in accordions */
.gradio-accordion textarea {
    overflow-y: auto !important;
    max-height: 600px !important;
}

/* Ensure accordion content can scroll */
.gradio-accordion .prose {
    overflow-y: auto !important;
    max-height: 700px !important;
}

/* Report textbox containers need scroll */
.gradio-textbox textarea {
    overflow-y: scroll !important;
}

/* Wrap container to ensure scroll works */
#fbi-output-box .wrap,
#confidence-output-box .wrap,
#subject-id-output-box .wrap {
    max-height: 500px !important;
    overflow-y: auto !important;
}

/* Subject ID container - force scrollability */
#subject-id-container {
    max-height: 450px !important;
    overflow-y: auto !important;
}

#subject-id-container textarea {
    overflow-y: scroll !important;
    max-height: 400px !important;
    min-height: 150px !important;
    resize: vertical !important;
}

#subject-id-container .gradio-textbox {
    max-height: 420px !important;
    overflow-y: auto !important;
}
"""

COLLAPSIBLE_JS = """
<script>
function toggleCollapsible(element) {
    element.classList.toggle('active');
    var content = element.nextElementSibling;
    content.classList.toggle('show');
}
</script>
"""


def format_analysis_with_collapsibles(text: str, expand_first: bool = True) -> str:
    """
    Parse analysis text and format with collapsible sections.

    Args:
        text: Raw analysis text from AI
        expand_first: Whether to expand the first section by default

    Returns:
        HTML string with collapsible sections
    """
    import re

    if not text or text.startswith("Analysis in progress") or text.startswith("ERROR"):
        return f'<div style="color: #6b7280; padding: 20px;">{text}</div>'

    # Split text into sections by common header patterns
    # Look for ## headers, ** bold headers **, or numbered sections like "1."
    section_pattern = r'(?:^|\n)((?:##\s*.+)|(?:\*\*.+\*\*)|(?:\d+\.\s*[A-Z].+))(?:\n|$)'

    sections = []
    current_section = None
    current_content = []

    lines = text.split('\n')

    for line in lines:
        # Check if this line is a section header
        is_header = False

        # ## Header
        if line.strip().startswith('##'):
            is_header = True
            header_text = line.strip().lstrip('#').strip()
        # **Bold Header**
        elif line.strip().startswith('**') and line.strip().endswith('**'):
            is_header = True
            header_text = line.strip().strip('*').strip()
        # Numbered header like "1. Section Name" or "1) Section Name"
        elif re.match(r'^\d+[\.\)]\s+[A-Z]', line.strip()):
            is_header = True
            header_text = line.strip()

        if is_header:
            # Save previous section
            if current_section is not None:
                sections.append((current_section, '\n'.join(current_content)))
            current_section = header_text
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section is not None:
        sections.append((current_section, '\n'.join(current_content)))
    elif current_content:
        # No sections found, treat entire text as one section
        sections.append(("Analysis Results", '\n'.join(current_content)))

    # Build HTML
    html_parts = [COLLAPSIBLE_JS]

    for i, (header, content) in enumerate(sections):
        # Determine if section should be expanded
        is_active = "active" if (expand_first and i == 0) else ""
        is_show = "show" if (expand_first and i == 0) else ""

        # Clean up content - escape HTML but preserve structure
        content_escaped = content.strip()
        content_escaped = content_escaped.replace('&', '&amp;')
        content_escaped = content_escaped.replace('<', '&lt;')
        content_escaped = content_escaped.replace('>', '&gt;')

        # Convert markdown-style formatting
        # Bold: **text** -> <strong>text</strong>
        content_escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #4a9eff;">\1</strong>', content_escaped)
        # Bullets: - text -> bullet
        content_escaped = re.sub(r'^- (.+)$', r'<span class="analysis-bullet">•</span>\1', content_escaped, flags=re.MULTILINE)
        content_escaped = re.sub(r'^\* (.+)$', r'<span class="analysis-bullet">•</span>\1', content_escaped, flags=re.MULTILINE)

        html_parts.append(f'''
        <div class="collapsible-container">
            <div class="collapsible-header {is_active}" onclick="toggleCollapsible(this)">
                <span class="collapsible-title">{header}</span>
                <span class="collapsible-icon">▼</span>
            </div>
            <div class="collapsible-content {is_show}">{content_escaped}</div>
        </div>
        ''')

    return ''.join(html_parts)


def generate_progress_html(current_step: int, status_message: str = "") -> str:
    """
    Generate HTML for the visual progress indicator.

    Args:
        current_step: Current step number (0-7, where 0 is not started, 7 is complete)
        status_message: Optional status message to display

    Returns:
        HTML string for progress display
    """
    steps = [
        ("1", "Video"),
        ("2", "Audio"),
        ("3", "Visual"),
        ("4", "Multi"),
        ("5", "Voice"),
        ("6", "Synth")
    ]

    # Progress: steps 1-6 are in progress, 7 means complete (100%)
    # Step 6 = synthesis running = 85%, only 7 (complete) = 100%
    if current_step == 0:
        progress_percent = 0
    elif current_step <= 6:
        progress_percent = (current_step / 7) * 100
    else:
        progress_percent = 100

    step_html = ""
    for i, (num, label) in enumerate(steps):
        step_num = i + 1
        if step_num < current_step:
            circle_class = "completed"
            label_class = ""
        elif step_num == current_step:
            circle_class = "active"
            label_class = "active"
        else:
            circle_class = ""
            label_class = ""

        checkmark = "✓" if step_num < current_step else num
        step_html += f'''
        <div class="progress-step">
            <div class="step-circle {circle_class}">{checkmark}</div>
            <div class="step-label {label_class}">{label}</div>
        </div>
        '''

    html = f'''
    <div class="progress-container">
        <div class="progress-steps">
            <div class="progress-line">
                <div class="progress-line-fill" style="width: {progress_percent}%"></div>
            </div>
            {step_html}
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar-fill" style="width: {progress_percent}%"></div>
        </div>
        <div class="progress-percentage">{progress_percent:.0f}% Complete</div>
    </div>
    '''
    return html


def save_api_key(api_key):
    """Save API key to encrypted storage."""
    config = ConfigManager()

    if not api_key or not api_key.strip():
        return "⚠️ Please enter an API key", "❌ Not Configured"

    # Validate format
    if not api_key.startswith('sk-'):
        return "⚠️ Invalid API key format (should start with 'sk-')", "❌ Invalid"

    # Save key
    success = config.save_api_key(api_key.strip())

    if success:
        return "✓ API key saved successfully (encrypted)", "✓ Configured"
    else:
        return "✗ Failed to save API key", "❌ Error"


def test_api_key(api_key):
    """Test if API key is valid."""
    config = ConfigManager()

    if not api_key or not api_key.strip():
        # Try to load saved key
        saved_key = config.load_api_key()
        if not saved_key:
            return "⚠️ No API key provided or saved"
        api_key = saved_key

    success, message = config.test_api_key(api_key.strip())
    return message


def load_saved_api_key():
    """Load the saved API key for display."""
    config = ConfigManager()
    key = config.load_api_key()
    if key:
        # Show only first 10 and last 4 characters
        masked = f"{key[:10]}...{key[-4:]}" if len(key) > 14 else key
        return masked, "✓ Configured"
    return "", "❌ Not Configured"


def get_subjects_list():
    """Get list of subjects for dropdown."""
    db = get_database()
    subjects = db.list_subjects()
    if not subjects:
        return []
    return [(f"{s.name} ({s.profile_count} profiles)", s.id) for s in subjects]


def load_subject_profiles(subject_id):
    """Load profiles for a selected subject."""
    if not subject_id:
        return "Select a subject to view their profiles.", [], None

    db = get_database()
    profiles = db.get_profiles_for_subject(subject_id)

    if not profiles:
        return "No profiles found for this subject.", [], None

    # Create summary
    subject = db.get_subject(subject_id)
    summary = f"""## {subject.name}

**Total Profiles:** {len(profiles)}
**First Profiled:** {profiles[-1]['timestamp'][:10] if profiles else 'N/A'}
**Last Profiled:** {profiles[0]['timestamp'][:10] if profiles else 'N/A'}
**Notes:** {subject.notes or 'None'}
"""

    # Create dropdown choices for profile selection
    profile_choices = [
        (f"Report #{p['report_number']} - {p['timestamp'][:10]} ({p['case_id'][:20]})", p['id'])
        for p in profiles
    ]

    return summary, profile_choices, profile_choices[0][1] if profile_choices else None


def load_profile_details(profile_id):
    """Load full details of a specific profile."""
    if not profile_id:
        return ("No profile selected.",) * 6

    db = get_database()
    profile = db.get_profile(profile_id=profile_id)

    if not profile:
        return ("Profile not found.",) * 6

    analyses = profile.get('analyses', {})

    meta = f"""**Case ID:** {profile['case_id']}
**Report #:** {profile['report_number']}
**Date:** {profile['timestamp']}
**Processing Time:** {profile['processing_time']:.2f}s
**Status:** {profile['status']}
**Notes:** {profile.get('notes', 'None')}
"""

    return (
        meta,
        analyses.get('sam_christensen_essence', 'Not available'),
        analyses.get('multimodal_behavioral', 'Not available'),
        analyses.get('audio_voice_analysis', 'Not available'),
        analyses.get('liwc_linguistic_analysis', 'Not available'),
        analyses.get('fbi_behavioral_synthesis', 'Not available')
    )


def refresh_subjects_dropdown():
    """Refresh the subjects dropdown with latest data."""
    choices = get_subjects_list()
    return gr.Dropdown(choices=choices, value=None)


def get_database_stats():
    """Get database statistics for display."""
    db = get_database()
    stats = db.get_stats()

    stats_html = f'''
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; padding: 16px;">
        <div style="background: #0a0e1a; border: 1px solid #2d3a5f; border-radius: 4px; padding: 16px; text-align: center;">
            <div style="color: #4a9eff; font-size: 28px; font-weight: bold;">{stats['total_subjects']}</div>
            <div style="color: #6b7280; font-size: 12px; text-transform: uppercase;">Subjects</div>
        </div>
        <div style="background: #0a0e1a; border: 1px solid #2d3a5f; border-radius: 4px; padding: 16px; text-align: center;">
            <div style="color: #4a9eff; font-size: 28px; font-weight: bold;">{stats['total_profiles']}</div>
            <div style="color: #6b7280; font-size: 12px; text-transform: uppercase;">Profiles</div>
        </div>
        <div style="background: #0a0e1a; border: 1px solid #2d3a5f; border-radius: 4px; padding: 16px; text-align: center;">
            <div style="color: #4a9eff; font-size: 28px; font-weight: bold;">{stats['average_processing_time']:.1f}s</div>
            <div style="color: #6b7280; font-size: 12px; text-transform: uppercase;">Avg Time</div>
        </div>
    </div>
    '''
    return stats_html


def run_profiling_analysis(video_file, essence_model, multimodal_model, audio_model, liwc_model, synthesis_model, subject_name, subject_notes, use_cache):
    """
    Main function to run profiling analysis on uploaded video.
    Yields progress updates and final results.

    Args:
        video_file: Gradio video file object
        essence_model: Model ID for Sam Christensen analysis
        multimodal_model: Model ID for multimodal analysis
        audio_model: Model ID for audio analysis
        liwc_model: Model ID for LIWC analysis
        synthesis_model: Model ID for FBI synthesis
        subject_name: Name of the subject being profiled
        subject_notes: Optional notes about the subject
        use_cache: Whether to use cached results

    Yields:
        Tuple of (progress_html, status, essence, multimodal, audio, liwc, fbi, nci, transcript, confidence, json_output, json_file)
    """
    # Helper to build yield tuple with optional viz outputs
    def build_yield(progress, status, essence, multimodal, audio, liwc, fbi, nci, transcript, confidence, json_out, file_out, mugshot=None, subject_id="", viz_conf=None, viz_big5=None, viz_dark=None, viz_threat=None, viz_mbti=None, viz_bte=None, viz_blink=None, viz_fate=None, viz_nci=None):
        base = (progress, status, essence, multimodal, audio, liwc, fbi, nci, transcript, confidence, json_out, file_out, mugshot, subject_id)
        if VISUALIZATIONS_AVAILABLE:
            return base + (viz_conf, viz_big5, viz_dark, viz_threat, viz_mbti, viz_bte, viz_blink, viz_fate, viz_nci)
        return base

    if video_file is None:
        error_msg = "⚠️ ERROR: No video file uploaded"
        yield build_yield(
            generate_progress_html(0), error_msg,
            "No video file provided", "No video file provided", "No video file provided",
            "No video file provided", "No video file provided", "No NCI analysis available",
            "No video file provided", "No video file provided", "{}", None
        )
        return

    try:
        # Check if API key is configured
        config = ConfigManager()
        if not config.has_api_key():
            error_msg = "⚠️ ERROR: API key not configured\n\nPlease configure your OpenRouter API key in the Settings section above."
            yield build_yield(
                generate_progress_html(0), error_msg,
                "API key required", "API key required", "API key required",
                "API key required", "API key required", "No NCI analysis available",
                "API key required", "API key required", "{}", None
            )
            return

        # Create model selection config
        model_config = ModelSelection(
            essence_model=essence_model,
            multimodal_model=multimodal_model,
            audio_model=audio_model,
            liwc_model=liwc_model,
            synthesis_model=synthesis_model
        )

        # Initialize profiler with selected models
        profiler = BehavioralProfiler(model_config=model_config)

        # Get video duration for time estimate (~5.5 min per minute of video)
        try:
            from media.frame_extractor import validate_video_file
            video_meta = validate_video_file(video_file)
            video_duration_min = video_meta.get('duration_seconds', 60) / 60
            estimated_min = int(video_duration_min * 5.5)
            time_estimate = f" (Est. {estimated_min}-{estimated_min + 3} min)"
        except:
            time_estimate = ""


        # Shared state for progress
        current_status = [f"⏳ Initializing analysis pipeline...{time_estimate}"]
        current_step = [0]

        # Progress callback that updates shared state
        def update_progress(message, step):
            current_status[0] = message
            current_step[0] = step

        # Yield initial status
        yield build_yield(
            generate_progress_html(0), current_status[0],
            "Analysis in progress...", "Analysis in progress...", "Analysis in progress...",
            "Analysis in progress...", "Analysis in progress...", "NCI analysis in progress...",
            "Transcription in progress...", "Confidence scoring in progress...", "{}", None
        )

        # Start profiling in a way that allows yielding
        import threading
        import time as time_module

        result_holder = [None]
        error_holder = [None]

        # Shared state for streaming partial results
        partial_results = {
            'essence': "Analysis in progress...",
            'multimodal': "Analysis in progress...",
            'audio': "Analysis in progress...",
            'liwc': "Analysis in progress...",
            'synthesis': "Analysis in progress...",
            'nci': "NCI analysis in progress...",
            'transcript': "Transcription in progress...",
            'confidence': "Confidence scoring in progress..."
        }
        results_lock = threading.Lock()

        # Results callback for streaming
        def update_results(stage, result):
            with results_lock:
                partial_results[stage] = result

        def run_profiling():
            try:
                result_holder[0] = profiler.profile_video(
                    video_path=video_file,
                    progress_callback=update_progress,
                    results_callback=update_results,
                    use_cache=use_cache
                )
            except Exception as e:
                error_holder[0] = e

        # Start analysis thread
        thread = threading.Thread(target=run_profiling)
        thread.start()

        # Yield progress updates while thread is running
        last_status = ""
        last_step = -1
        while thread.is_alive():
            if current_status[0] != last_status or current_step[0] != last_step:
                last_status = current_status[0]
                last_step = current_step[0]

            # Get current partial results
            with results_lock:
                current_essence = partial_results['essence']
                current_multimodal = partial_results['multimodal']
                current_audio = partial_results['audio']
                current_liwc = partial_results['liwc']
                current_synthesis = partial_results['synthesis']
                current_nci = partial_results['nci']
                current_transcript = partial_results['transcript']
                current_confidence = partial_results['confidence']

            yield build_yield(
                generate_progress_html(current_step[0]), current_status[0],
                current_essence, current_multimodal, current_audio,
                current_liwc, current_synthesis, current_nci, current_transcript,
                current_confidence, "{}", None
            )
            time_module.sleep(0.5)

        thread.join()

        # Check for errors
        if error_holder[0]:
            raise error_holder[0]

        result = result_holder[0]

        # Format results for display
        formatted = profiler.format_result_for_display(result)

        # Create downloadable JSON file
        json_output = json.dumps(result, indent=2, ensure_ascii=False)

        # Create temporary file for download
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        )
        temp_file.write(json_output)
        temp_file.close()

        # Save to database if subject name provided
        saved_msg = ""
        if subject_name and subject_name.strip():
            try:
                db = get_database()
                profile_record = db.save_profile(
                    result=result,
                    subject_name=subject_name.strip(),
                    video_source=video_file if isinstance(video_file, str) else "uploaded",
                    notes=subject_notes or ""
                )
                saved_msg = f"\n\n📁 Saved as Report #{profile_record.report_number} for '{subject_name.strip()}'"
            except Exception as save_err:
                saved_msg = f"\n\n⚠️ Failed to save to database: {str(save_err)}"

        # Generate visualizations from real data (do this before final status)
        viz_confidence = None
        viz_big_five = None
        viz_dark_triad = None
        viz_threat = None
        viz_mbti = None
        # NCI/Chase Hughes visualizations
        viz_bte = None
        viz_blink = None
        viz_fate = None
        viz_nci = None
        viz_status = ""

        if VISUALIZATIONS_AVAILABLE:
            # Yield intermediate status showing visualization generation
            yield build_yield(
                generate_progress_html(6), "⏳ Generating visualizations...",
                formatted['essence'], formatted['multimodal'], formatted['audio'],
                formatted['liwc'], formatted['fbi_profile'],
                formatted.get('nci', 'NCI analysis not available'),
                formatted.get('transcript', 'Transcription not available'),
                formatted.get('confidence', 'Confidence scoring not available'),
                json_output, temp_file.name
            )

            try:
                from output.visualizations import (
                    create_confidence_gauge,
                    create_big_five_radar,
                    create_dark_triad_bars,
                    create_threat_matrix,
                    # NCI/Chase Hughes visualizations
                    create_bte_gauge,
                    create_blink_rate_chart,
                    create_fate_radar,
                    create_nci_deception_summary
                )
                # Create charts from actual analysis data
                confidence_data = result.get('confidence', {})
                analyses = result.get('analyses', {})

                # Use dedicated personality/threat synthesis for scores, fallback to FBI synthesis
                personality_text = analyses.get('personality_synthesis', '') or analyses.get('fbi_behavioral_synthesis', '')
                threat_text = analyses.get('threat_synthesis', '') or analyses.get('fbi_behavioral_synthesis', '')

                # Combine all analysis text for NCI extraction
                all_analysis_text = analyses.get('fbi_behavioral_synthesis', '')
                for key, value in analyses.items():
                    if isinstance(value, str):
                        all_analysis_text += "\n" + value

                # Core visualizations
                viz_confidence = create_confidence_gauge(confidence_data)
                viz_big_five = create_big_five_radar(personality_text)
                viz_dark_triad = create_dark_triad_bars(personality_text)
                viz_threat = create_threat_matrix(threat_text)
                viz_mbti = create_mbti_chart(all_analysis_text)

                # NCI/Chase Hughes visualizations
                viz_bte = create_bte_gauge(all_analysis_text)
                viz_blink = create_blink_rate_chart(all_analysis_text)
                viz_fate = create_fate_radar(all_analysis_text)
                viz_nci = create_nci_deception_summary(all_analysis_text)

                core_charts = sum(1 for v in [viz_confidence, viz_big_five, viz_dark_triad, viz_threat, viz_mbti] if v is not None)
                nci_charts = sum(1 for v in [viz_bte, viz_blink, viz_fate, viz_nci] if v is not None)
                total_charts = core_charts + nci_charts
                if total_charts > 0:
                    viz_status = f"\n📊 {total_charts} visualization(s) generated ({core_charts} core, {nci_charts} NCI)"
            except Exception as viz_err:
                logger.warning(f"Visualization generation failed: {viz_err}")
                viz_status = "\n⚠️ Visualizations unavailable"

        # Extract mugshot image from result (decode base64 to PIL Image)
        mugshot_pil = None
        subject_id_text = "Subject identification not available"
        try:
            mugshot_data = result.get('mugshot', {})
            if mugshot_data.get('available') and mugshot_data.get('base64'):
                import base64 as b64
                from io import BytesIO
                from PIL import Image
                img_bytes = b64.b64decode(mugshot_data['base64'])
                mugshot_pil = Image.open(BytesIO(img_bytes))
                pass  # Mugshot loaded
            
            # Get subject identification from analyses
            subject_id_text = result.get('analyses', {}).get('subject_identification', 'Subject identification not available')
        except Exception as mug_err:
            pass  # Mugshot display failed silently

        # Final status
        final_status = f"""✓ ANALYSIS COMPLETE

Case ID: {result.get('case_id', 'N/A')}
Processing Time: {result.get('processing_time_seconds', 0):.2f}s
Timestamp: {result.get('timestamp', 'N/A')}

All analyses generated successfully.{viz_status}
Download JSON report below.{saved_msg}"""

        yield build_yield(
            generate_progress_html(7), final_status,  # Step 7 = 100% complete
            formatted['essence'], formatted['multimodal'], formatted['audio'],
            formatted['liwc'], formatted['fbi_profile'],
            formatted.get('nci', 'NCI analysis not available'),
            formatted.get('transcript', 'Transcription not available'),
            formatted.get('confidence', 'Confidence scoring not available'),
            json_output, temp_file.name,
            mugshot_pil, subject_id_text,
            viz_confidence, viz_big_five, viz_dark_triad, viz_threat, viz_mbti,
            viz_bte, viz_blink, viz_fate, viz_nci
        )

    except Exception as e:
        error_status = f"""⚠️ ANALYSIS FAILED

Error: {str(e)}

Please check:
1. Video file is valid (.mp4, .mov, .avi, .webm)
2. Video duration is 10-300 seconds (5 minutes max)
3. File size is under 100MB
4. OPENROUTER_API_KEY is set in .env file
5. You have active API credits"""

        yield build_yield(
            generate_progress_html(0), error_status,
            f"ERROR: {str(e)}", f"ERROR: {str(e)}", f"ERROR: {str(e)}",
            f"ERROR: {str(e)}", f"ERROR: {str(e)}", f"ERROR: {str(e)}",
            f"ERROR: {str(e)}", f"ERROR: {str(e)}", json.dumps({"error": str(e)}, indent=2), None
        )


# Import visualizations module
try:
    from output.visualizations import (
        create_all_visualizations,
        create_confidence_gauge,
        create_big_five_radar,
        create_dark_triad_bars,
        create_threat_matrix,
        create_mbti_chart,
        # NCI/Chase Hughes visualizations
        create_bte_gauge,
        create_blink_rate_chart,
        create_fate_radar,
        create_nci_deception_summary,
        check_plotly_available
    )
    VISUALIZATIONS_AVAILABLE = check_plotly_available()
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


# Create Gradio interface
def create_interface():
    """Create and return the Gradio interface."""

    # Use FBI-themed Glass theme
    fbi_theme = create_fbi_theme()

    with gr.Blocks(css=MINIMAL_CSS, theme=fbi_theme, title="FBI Behavioral Profiler") as app:
        # Clean Header with Attribution
        gr.HTML("""
            <div class="app-header">
                <h1 class="app-title">Behavioral Profiling System</h1>
                <p class="app-subtitle">Multimodal AI-Powered Psychological Assessment</p>
                <p class="app-attribution">Created by <a href="https://x.com/LLMSherpa" target="_blank">@LLMSherpa</a></p>
            </div>
        """)

        # Settings Section
        with gr.Accordion("⚙️ Settings & Configuration", open=False):
            gr.Markdown("""
            ### API Key Configuration
            Enter your OpenRouter API key below. The key will be encrypted and saved securely.

            **Get your API key:** [https://openrouter.ai/keys](https://openrouter.ai/keys)
            """)

            with gr.Row():
                with gr.Column(scale=3):
                    api_key_input = gr.Textbox(
                        label="OpenRouter API Key",
                        placeholder="sk-or-v1-...",
                        type="password",
                        info="Your API key is encrypted before being saved"
                    )
                with gr.Column(scale=1):
                    api_status = gr.Textbox(
                        label="Status",
                        value="❌ Not Configured",
                        interactive=False,
                        max_lines=1
                    )

            with gr.Row():
                save_key_btn = gr.Button("💾 Save API Key", variant="primary", size="sm")
                test_key_btn = gr.Button("🔍 Test Connection", variant="secondary", size="sm")
                load_key_btn = gr.Button("📋 Load Saved Key", variant="secondary", size="sm")

            api_message = gr.Textbox(
                label="",
                value="",
                interactive=False,
                show_label=False,
                max_lines=2
            )

            gr.Markdown("""
            **Note:** The API key is stored locally in an encrypted format.
            Each analysis costs approximately $0.11-0.23 depending on video complexity.
            """)

            gr.Markdown("---")
            gr.Markdown("""
            ### Model Configuration
            Select which AI models to use for each analysis stage.
            **Note:** Audio/multimodal stages require Gemini models (only Gemini supports audio input).
            """)

            with gr.Row():
                with gr.Column():
                    essence_model_dropdown = gr.Dropdown(
                        choices=ESSENCE_MODEL_CHOICES,
                        value=get_default_model_for_stage("essence"),
                        label="Sam Christensen Analysis (Vision)",
                        info="Model for visual essence profiling"
                    )
                with gr.Column():
                    multimodal_model_dropdown = gr.Dropdown(
                        choices=MULTIMODAL_MODEL_CHOICES,
                        value=get_default_model_for_stage("multimodal"),
                        label="Multimodal Analysis (Vision + Audio)",
                        info="Must be Gemini for audio support"
                    )

            with gr.Row():
                with gr.Column():
                    audio_model_dropdown = gr.Dropdown(
                        choices=AUDIO_MODEL_CHOICES,
                        value=get_default_model_for_stage("audio"),
                        label="Audio/Voice Analysis",
                        info="Must be Gemini for audio support"
                    )
                with gr.Column():
                    liwc_model_dropdown = gr.Dropdown(
                        choices=LIWC_MODEL_CHOICES,
                        value=get_default_model_for_stage("liwc"),
                        label="LIWC Linguistic Analysis",
                        info="Must be Gemini for audio support"
                    )

            with gr.Row():
                with gr.Column():
                    synthesis_model_dropdown = gr.Dropdown(
                        choices=SYNTHESIS_MODEL_CHOICES,
                        value=get_default_model_for_stage("synthesis"),
                        label="FBI Behavioral Synthesis (Text)",
                        info="Model for final profile synthesis"
                    )
                with gr.Column():
                    gr.Markdown("""
                    **Cost Tiers:**
                    - Budget: Faster, cheaper
                    - Standard: Balanced
                    - Premium: Best quality
                    """)

            gr.Markdown("---")
            gr.Markdown("""
            ### Cache Settings
            Cache analysis results to avoid redundant API calls when re-analyzing the same video.
            """)

            with gr.Row():
                use_cache_checkbox = gr.Checkbox(
                    label="Enable Caching",
                    value=True,
                    info="Use cached results when same video+models are analyzed again"
                )
                clear_cache_btn = gr.Button("🗑️ Clear Cache", size="sm", variant="secondary")
                cache_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    show_label=False,
                    max_lines=1,
                    scale=2
                )

            def get_cache_stats_display():
                cache = get_cache()
                stats = cache.get_stats()
                return f"📦 {stats['total_entries']} cached | {stats['total_size_mb']:.1f} MB | {stats['total_hits']} hits"

            def clear_all_cache():
                cache = get_cache()
                count = cache.invalidate(all_entries=True)
                return f"✓ Cleared {count} cache entries"

            clear_cache_btn.click(
                fn=clear_all_cache,
                inputs=[],
                outputs=[cache_status]
            )

        # Input Section
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="section-header">Video Input</div>')

                with gr.Tabs():
                    with gr.Tab("Upload File"):
                        video_input = gr.File(
                            label="Subject Video File",
                            file_types=[".mp4", ".mov", ".avi", ".webm"],
                            type="filepath"
                        )

                    with gr.Tab("Import from URL"):
                        url_input = gr.Textbox(
                            label="Video URL",
                            placeholder="https://youtube.com/watch?v=... or direct video URL",
                            info="Supports YouTube, Vimeo, Twitter/X, TikTok, and direct video URLs"
                        )
                        with gr.Row():
                            fetch_url_btn = gr.Button("Fetch Video", variant="secondary", size="sm")
                            url_status = gr.Textbox(
                                label="",
                                value="",
                                interactive=False,
                                show_label=False,
                                max_lines=2
                            )

                gr.Markdown("""
                **Requirements:**
                - Formats: .mp4, .mov, .avi, .webm
                - Duration: 10-300 seconds (5 minutes max)
                - Max size: 100MB
                - Clear view of subject
                """)

                gr.HTML('<div class="section-header" style="margin-top: 15px;">Subject Information</div>')
                with gr.Row():
                    subject_name_input = gr.Textbox(
                        label="Subject Name",
                        placeholder="Enter name to save profile (optional)",
                        info="Name the person/entity to track multiple profiles over time",
                        scale=3
                    )
                    subject_notes_input = gr.Textbox(
                        label="Notes",
                        placeholder="Optional notes...",
                        scale=2
                    )

                analyze_button = gr.Button(
                    "INITIATE BEHAVIORAL ANALYSIS",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                gr.HTML('<div class="section-header">Video Preview</div>')
                video_preview = gr.Video(
                    label="Preview",
                    show_label=False,
                    height=300
                )
                video_metadata_display = gr.HTML(
                    value='<div style="color: #6b7280; text-align: center; padding: 20px;">Upload a video to see preview and metadata</div>'
                )

        # Connect file upload to video preview
        def update_video_preview(video_file):
            if video_file is None:
                return None, '<div style="color: #6b7280; text-align: center; padding: 20px;">Upload a video to see preview and metadata</div>'

            try:
                from media.frame_extractor import validate_video_file
                metadata = validate_video_file(video_file)

                duration = metadata.get('duration_seconds', 0)
                mins = int(duration // 60)
                secs = int(duration % 60)

                width, height = metadata.get('resolution', (0, 0))
                fps = metadata.get('fps', 0)
                size_mb = metadata.get('file_size_mb', 0)

                metadata_html = f'''
                <div style="background: #0a0e1a; border: 1px solid #2d3a5f; border-radius: 4px; padding: 12px; margin-top: 10px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-family: monospace; font-size: 12px;">
                        <div style="color: #6b7280;">Duration:</div>
                        <div style="color: #4a9eff;">{mins}:{secs:02d}</div>
                        <div style="color: #6b7280;">Resolution:</div>
                        <div style="color: #4a9eff;">{width}x{height}</div>
                        <div style="color: #6b7280;">Frame Rate:</div>
                        <div style="color: #4a9eff;">{fps:.1f} fps</div>
                        <div style="color: #6b7280;">File Size:</div>
                        <div style="color: #4a9eff;">{size_mb:.2f} MB</div>
                    </div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #2d3a5f;">
                        {"<span style='color: #22c55e;'>✓ Video meets requirements</span>" if 10 <= duration <= 300 and size_mb <= 100 else "<span style='color: #ef4444;'>⚠ Video may not meet requirements</span>"}
                    </div>
                </div>
                '''
                return video_file, metadata_html
            except Exception as e:
                return video_file, f'<div style="color: #ef4444; padding: 20px;">Error reading video: {str(e)}</div>'

        video_input.change(
            fn=update_video_preview,
            inputs=[video_input],
            outputs=[video_preview, video_metadata_display]
        )

        # URL fetch handler
        def fetch_video_from_url(url):
            if not url or not url.strip():
                return None, "⚠️ Please enter a URL", None, '<div style="color: #6b7280; text-align: center; padding: 20px;">Upload a video to see preview and metadata</div>'

            url = url.strip()
            if not is_valid_url(url):
                return None, "⚠️ Invalid URL format", None, '<div style="color: #ef4444; padding: 20px;">Invalid URL</div>'

            try:
                # Download the video
                file_path, metadata = download_video(
                    url,
                    max_duration=300,
                    max_filesize_mb=100
                )

                # Create metadata display
                duration = metadata.get('duration_seconds', 0)
                mins = int(duration // 60)
                secs = int(duration % 60)

                metadata_html = f'''
                <div style="background: #0a0e1a; border: 1px solid #2d3a5f; border-radius: 4px; padding: 12px; margin-top: 10px;">
                    <div style="font-weight: bold; color: #4a9eff; margin-bottom: 8px;">{metadata.get('title', 'Unknown')[:50]}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-family: monospace; font-size: 12px;">
                        <div style="color: #6b7280;">Platform:</div>
                        <div style="color: #4a9eff;">{metadata.get('platform', 'Unknown')}</div>
                        <div style="color: #6b7280;">Duration:</div>
                        <div style="color: #4a9eff;">{mins}:{secs:02d}</div>
                        <div style="color: #6b7280;">File Size:</div>
                        <div style="color: #4a9eff;">{metadata.get('file_size_mb', 0):.2f} MB</div>
                        <div style="color: #6b7280;">Uploader:</div>
                        <div style="color: #4a9eff;">{metadata.get('uploader', 'Unknown')[:20]}</div>
                    </div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #2d3a5f;">
                        <span style='color: #22c55e;'>✓ Video downloaded successfully</span>
                    </div>
                </div>
                '''

                return file_path, f"✓ Downloaded: {metadata.get('title', 'Video')[:30]}", file_path, metadata_html

            except ImportError as e:
                return None, "⚠️ yt-dlp not installed. Run: pip install yt-dlp", None, '<div style="color: #ef4444; padding: 20px;">yt-dlp required</div>'
            except ValueError as e:
                return None, f"⚠️ {str(e)}", None, f'<div style="color: #ef4444; padding: 20px;">{str(e)}</div>'
            except Exception as e:
                return None, f"⚠️ Download failed: {str(e)}", None, f'<div style="color: #ef4444; padding: 20px;">Error: {str(e)}</div>'

        fetch_url_btn.click(
            fn=fetch_video_from_url,
            inputs=[url_input],
            outputs=[video_input, url_status, video_preview, video_metadata_display]
        )

        # Processing Status
        gr.HTML('<div class="section-header">Processing Status</div>')
        progress_html = gr.HTML(
            value=generate_progress_html(0),
            visible=True
        )
        status_display = gr.Textbox(
            label="",
            value="⏳ Ready to begin analysis. Upload video and click the button above.",
            lines=4,
            max_lines=6,
            interactive=False,
            show_label=False
        )

        # Results Section - Analysis Results
        gr.HTML('<div class="section-header" style="margin-top: 30px;">Analysis Results</div>')

        with gr.Tabs():
            with gr.Tab("📂 Raw Analysis Data"):
                gr.Markdown("*Technical details - expand for raw outputs*")

                with gr.Accordion("Visual Essence", open=False):
                    essence_output = gr.Textbox(
                        label="Forensic Visual Essence Analysis",
                        value="Results will appear here after analysis...",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )

                with gr.Accordion("Multimodal Analysis", open=False):
                    multimodal_output = gr.Textbox(
                        label="Comprehensive Multimodal Analysis (Video + Audio)",
                        value="Results will appear here after analysis...",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )

                with gr.Accordion("Audio/Voice", open=False):
                    audio_output = gr.Textbox(
                        label="Voice Forensics & Paralinguistic Analysis",
                        value="Results will appear here after analysis...",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )

                with gr.Accordion("LIWC Metrics", open=True):
                    liwc_output = gr.Textbox(
                        label="Linguistic Inquiry & Word Count Analysis",
                        value="Results will appear here after analysis...",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )

                with gr.Accordion("Transcript", open=True):
                    transcript_output = gr.Textbox(
                        label="Audio Transcription",
                        value="Speech transcript will appear here after analysis...",
                        lines=15,
                        interactive=False,
                        show_copy_button=True
                    )

            with gr.Tab("🎯 FBI Behavioral Profile"):
                gr.Markdown("*WHO they are: Psychology, personality, and threat assessment*")

                # Subject Identification with Mugshot
                with gr.Row():
                    with gr.Column(scale=1):
                        mugshot_image = gr.Image(
                            label="Subject",
                            height=200,
                            width=200,
                            show_label=True,
                            type="pil",
                            interactive=False
                        )
                    with gr.Column(scale=3, elem_id="subject-id-container"):
                        subject_id_output = gr.Textbox(
                            label="Subject Identification",
                            value="Subject identification will appear here after analysis...",
                            lines=20,
                            max_lines=100,
                            autoscroll=False,
                            interactive=False,
                            show_copy_button=True,
                            elem_id="subject-id-output-box"
                        )

                # Spacer before visualizations
                gr.HTML('<div style="height: 20px;"></div>')

                # Visual Analytics Row (charts)
                if VISUALIZATIONS_AVAILABLE:
                    gr.Markdown("### Visual Analytics")
                    with gr.Row():
                        with gr.Column(scale=1):
                            confidence_gauge = gr.Plot(
                                label="Confidence",
                                show_label=False
                            )
                        with gr.Column(scale=2):
                            big_five_chart = gr.Plot(
                                label="Personality Profile",
                                show_label=False
                            )

                    gr.HTML('<div style="height: 15px;"></div>')
                    with gr.Row():
                        with gr.Column(scale=1):
                            dark_triad_chart = gr.Plot(
                                label="Dark Triad",
                                show_label=False
                            )
                        with gr.Column(scale=1):
                            threat_chart = gr.Plot(
                                label="Threat Assessment",
                                show_label=False
                            )

                    gr.HTML('<div style="height: 15px;"></div>')
                    # MBTI Chart Row
                    with gr.Row():
                        with gr.Column(scale=1):
                            mbti_chart = gr.Plot(
                                label="MBTI Profile",
                                show_label=False
                            )
                else:
                    # Placeholder variables if visualizations not available
                    confidence_gauge = gr.State(None)
                    big_five_chart = gr.State(None)
                    dark_triad_chart = gr.State(None)
                    threat_chart = gr.State(None)
                    mbti_chart = gr.State(None)

                with gr.Accordion("📋 FBI Behavioral Synthesis", open=True):
                    fbi_output = gr.Textbox(
                        label="FBI-Style Psychological Assessment & Recommendations",
                        value="Results will appear here after analysis...",
                        lines=30,
                        max_lines=100,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True,
                        elem_id="fbi-output-box"
                    )

                with gr.Accordion("📊 Confidence Details", open=False):
                    confidence_output = gr.Textbox(
                        label="Analysis Confidence Assessment",
                        value="Confidence assessment will appear here after analysis...",
                        lines=20,
                        max_lines=60,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True,
                        elem_id="confidence-output-box"
                    )

                with gr.Accordion("🔧 Complete JSON Data", open=False):
                    json_output = gr.Textbox(
                        label="Full Analysis Report (JSON Format)",
                        value="{}",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )

            with gr.Tab("🔍 Deception Detection"):
                gr.Markdown("*Are they LYING? Credibility and deception indicators*")

                # NCI Visual Analytics Row 1 (BTE and Blink Rate)
                if VISUALIZATIONS_AVAILABLE:
                    with gr.Row():
                        with gr.Column(scale=1):
                            bte_gauge = gr.Plot(
                                label="BTE Deception Score",
                                show_label=False
                            )
                        with gr.Column(scale=1):
                            blink_rate_chart = gr.Plot(
                                label="Blink Rate Analysis",
                                show_label=False
                            )

                    # NCI Visual Analytics Row 2 (FATE and Summary)
                    with gr.Row():
                        with gr.Column(scale=1):
                            fate_radar = gr.Plot(
                                label="FATE Motivational Profile",
                                show_label=False
                            )
                        with gr.Column(scale=1):
                            nci_deception_chart = gr.Plot(
                                label="NCI Deception Summary",
                                show_label=False
                            )
                else:
                    # Placeholder variables if visualizations not available
                    bte_gauge = gr.State(None)
                    blink_rate_chart = gr.State(None)
                    fate_radar = gr.State(None)
                    nci_deception_chart = gr.State(None)

                # NCI Analysis Text Output
                with gr.Accordion("📋 Full NCI Analysis Details", open=True):
                    nci_output = gr.Textbox(
                        label="NCI/Chase Hughes Behavioral Analysis",
                        value="Run analysis to see NCI deception indicators...",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )

                with gr.Accordion("ℹ️ NCI Methodology Reference", open=False):
                    gr.Markdown("""
**Behavioral Table of Elements (BTE)**: Score of 12+ suggests high deception probability

**Blink Rate**: Normal 17-25 BPM, Stressed 50+ BPM

**FATE Model**: Focus, Authority, Tribe, Emotion - primary psychological drivers

**Five C's Framework**: Change, Context, Clusters, Culture, Checklist

**Detail Mountain/Valley**: Truth-tellers add details over time, deceivers often reduce details

**Minimizing Language**: Words like "just", "only", "a little" may indicate deception

**Gestural Mismatch**: Incongruence between verbal content and body language
                    """)

        # Download Section
        gr.HTML('<div class="section-header">Export Report</div>')
        with gr.Row():
            download_button = gr.File(
                label="Download JSON Report",
                type="filepath"
            )
            pdf_download = gr.File(
                label="Download PDF Report",
                type="filepath"
            )

        with gr.Row():
            generate_pdf_btn = gr.Button(
                "📄 Generate PDF Report",
                variant="secondary",
                size="sm",
                interactive=REPORTLAB_AVAILABLE
            )
            pdf_status = gr.Textbox(
                label="",
                value="" if REPORTLAB_AVAILABLE else "⚠️ reportlab not installed",
                interactive=False,
                show_label=False,
                max_lines=1,
                scale=2
            )

        # Profile History Section
        with gr.Accordion("📁 Profile History", open=False):
            gr.Markdown("""
            ### Saved Profiles
            Browse previously analyzed subjects and their profile history.
            """)

            # Stats display
            history_stats = gr.HTML(value=get_database_stats())
            refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
            refresh_stats_btn.click(
                fn=get_database_stats,
                inputs=[],
                outputs=[history_stats]
            )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Select Subject**")
                    subject_dropdown = gr.Dropdown(
                        choices=get_subjects_list(),
                        label="Subject",
                        info="Select a subject to view their profiles",
                        interactive=True
                    )
                    refresh_subjects_btn = gr.Button("🔄 Refresh List", size="sm")

                with gr.Column(scale=2):
                    subject_summary = gr.Markdown(
                        value="Select a subject to view their profile history."
                    )

            with gr.Row():
                with gr.Column(scale=4):
                    profile_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Profile Report",
                        info="Choose a specific analysis to view",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    delete_profile_btn = gr.Button(
                        "🗑️ Delete Profile",
                        variant="stop",
                        size="sm"
                    )

            # Profile details display
            with gr.Accordion("Profile Details", open=True):
                history_meta = gr.Markdown(value="*Select a profile to view details*")

                with gr.Tabs():
                    with gr.Tab("Essence"):
                        history_essence = gr.Textbox(
                            label="Sam Christensen Essence",
                            value="",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True
                        )
                    with gr.Tab("Multimodal"):
                        history_multimodal = gr.Textbox(
                            label="Multimodal Analysis",
                            value="",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True
                        )
                    with gr.Tab("Audio"):
                        history_audio = gr.Textbox(
                            label="Audio/Voice Analysis",
                            value="",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True
                        )
                    with gr.Tab("LIWC"):
                        history_liwc = gr.Textbox(
                            label="LIWC Analysis",
                            value="",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True
                        )
                    with gr.Tab("FBI Profile"):
                        history_fbi = gr.Textbox(
                            label="FBI Behavioral Synthesis",
                            value="",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True
                        )

            # Event handlers for history browsing
            def update_subject_selection(subject_id):
                summary, choices, default = load_subject_profiles(subject_id)
                return summary, gr.Dropdown(choices=choices, value=default)

            subject_dropdown.change(
                fn=update_subject_selection,
                inputs=[subject_dropdown],
                outputs=[subject_summary, profile_dropdown]
            )

            profile_dropdown.change(
                fn=load_profile_details,
                inputs=[profile_dropdown],
                outputs=[
                    history_meta,
                    history_essence,
                    history_multimodal,
                    history_audio,
                    history_liwc,
                    history_fbi
                ]
            )

            def do_refresh_subjects():
                choices = get_subjects_list()
                return gr.Dropdown(choices=choices, value=None)

            refresh_subjects_btn.click(
                fn=do_refresh_subjects,
                inputs=[],
                outputs=[subject_dropdown]
            )

            # Delete profile handler
            def delete_selected_profile(profile_id, subject_id):
                """Delete the selected profile and refresh the list."""
                if not profile_id:
                    return (
                        gr.Dropdown(choices=[], value=None),
                        "*No profile selected*",
                        "", "", "", "", ""
                    )

                # profile_id is already the database ID (from dropdown value)
                db = get_database()
                db.delete_profile(int(profile_id))

                # Refresh the profile list
                if subject_id:
                    summary, choices, default = load_subject_profiles(subject_id)
                    return (
                        gr.Dropdown(choices=choices, value=None),
                        "*Profile deleted. Select another profile.*",
                        "", "", "", "", ""
                    )
                return (
                    gr.Dropdown(choices=[], value=None),
                    "*Profile deleted.*",
                    "", "", "", "", ""
                )

            delete_profile_btn.click(
                fn=delete_selected_profile,
                inputs=[profile_dropdown, subject_dropdown],
                outputs=[
                    profile_dropdown,
                    history_meta,
                    history_essence,
                    history_multimodal,
                    history_audio,
                    history_liwc,
                    history_fbi
                ]
            )

        # ==================================================================================
        # DEVELOPER META-ANALYSIS SECTION - TO REMOVE BEFORE PRODUCTION
        # ==================================================================================
        with gr.Accordion("🛠️ [DEV] Meta-Analysis (REMOVE BEFORE UPLOAD)", open=False):
            gr.Markdown("""
            ### Developer Meta-Analysis
            **⚠️ This section is for DEVELOPMENT ONLY and should be REMOVED before production upload.**

            This sends the complete profiling report to Gemini 3 Pro for analysis of:
            - Profiler system improvements
            - Workflow optimization suggestions
            - Behavioral analysis quality feedback
            - Prompt engineering recommendations
            """)

            with gr.Row():
                dev_meta_model = gr.Dropdown(
                    choices=[(f"{m.name} ({m.provider})", m.id) for m in DEV_META_MODELS],
                    value=DEFAULT_DEV_META_MODEL,
                    label="Meta-Analysis Model",
                    info="Model to use for meta-analysis"
                )
                run_meta_btn = gr.Button(
                    "🔍 Run Meta-Analysis",
                    variant="secondary",
                    size="sm"
                )

            dev_meta_status = gr.Textbox(
                label="",
                value="Click 'Run Meta-Analysis' after completing an analysis to get feedback.",
                interactive=False,
                show_label=False,
                max_lines=2
            )

            dev_meta_output = gr.Textbox(
                label="Meta-Analysis Feedback",
                value="",
                lines=30,
                interactive=False,
                show_copy_button=True
            )

            # Meta-analysis handler
            def run_meta_analysis_handler(json_str, model_id):
                """Run developer meta-analysis on the current result."""
                if not json_str or json_str == "{}" or json_str.startswith('{"error"'):
                    return "⚠️ No analysis results available. Run analysis first.", ""

                try:
                    import json as json_module
                    result = json_module.loads(json_str)

                    # Run meta-analysis
                    meta_feedback = run_dev_meta_analysis(
                        result=result,
                        model=model_id
                    )

                    return "✓ Meta-analysis complete", meta_feedback

                except Exception as e:
                    return f"⚠️ Meta-analysis failed: {str(e)}", ""

            run_meta_btn.click(
                fn=run_meta_analysis_handler,
                inputs=[json_output, dev_meta_model],
                outputs=[dev_meta_status, dev_meta_output]
            )

        # Footer
        gr.HTML("""
            <div class="disclaimer-box">
                <p class="disclaimer-text">
                    ⚠️ Research & Educational Use Only • Subject Consent Required • No Unauthorized Surveillance
                </p>
            </div>
        """)

        gr.Markdown("""
        ---
        **Processing Pipeline:** Keyframe Extraction → Audio Analysis → Visual Essence → Multimodal Behavioral → Linguistic Analysis → FBI Synthesis

        *Powered by OpenRouter API • ~1-2 min processing time*
        """)

        # Event handlers for Settings
        save_key_btn.click(
            fn=save_api_key,
            inputs=[api_key_input],
            outputs=[api_message, api_status]
        )

        test_key_btn.click(
            fn=test_api_key,
            inputs=[api_key_input],
            outputs=[api_message]
        )

        load_key_btn.click(
            fn=load_saved_api_key,
            inputs=[],
            outputs=[api_key_input, api_status]
        )

        # Event handler for Analysis - build outputs list
        analysis_outputs = [
            progress_html,
            status_display,
            essence_output,
            multimodal_output,
            audio_output,
            liwc_output,
            fbi_output,
            nci_output,
            transcript_output,
            confidence_output,
            json_output,
            download_button,
            mugshot_image,
            subject_id_output
        ]

        # Add visualization outputs if available (core + NCI charts)
        if VISUALIZATIONS_AVAILABLE:
            analysis_outputs.extend([
                confidence_gauge,
                big_five_chart,
                dark_triad_chart,
                threat_chart,
                mbti_chart,
                # NCI/Chase Hughes charts
                bte_gauge,
                blink_rate_chart,
                fate_radar,
                nci_deception_chart
            ])

        analyze_button.click(
            fn=run_profiling_analysis,
            inputs=[
                video_input,
                essence_model_dropdown,
                multimodal_model_dropdown,
                audio_model_dropdown,
                liwc_model_dropdown,
                synthesis_model_dropdown,
                subject_name_input,
                subject_notes_input,
                use_cache_checkbox
            ],
            outputs=analysis_outputs
        )

        # PDF generation handler
        def generate_pdf_from_results(json_str, subject_name):
            """Generate PDF from the JSON results."""
            if not REPORTLAB_AVAILABLE:
                return None, "⚠️ reportlab not installed. Run: pip install reportlab"

            if not json_str or json_str == "{}" or json_str.startswith('{"error"'):
                return None, "⚠️ No analysis results to export. Run analysis first."

            try:
                import json as json_module
                result = json_module.loads(json_str)

                pdf_path = generate_pdf_report(
                    result=result,
                    subject_name=subject_name if subject_name else None
                )

                return pdf_path, "✓ PDF generated successfully"
            except Exception as e:
                return None, f"⚠️ PDF generation failed: {str(e)}"

        generate_pdf_btn.click(
            fn=generate_pdf_from_results,
            inputs=[json_output, subject_name_input],
            outputs=[pdf_download, pdf_status]
        )

        # Load API key status on startup
        app.load(
            fn=lambda: load_saved_api_key()[1],
            inputs=[],
            outputs=[api_status]
        )

    return app


if __name__ == "__main__":
    # Check for API key
    config = ConfigManager()
    if not config.has_api_key():
        print("\n" + "="*70)
        print("⚠️  API KEY NOT CONFIGURED")
        print("="*70)
        print("\nConfigure your OpenRouter API key via the web interface:")
        print("1. Launch the app (it will open in your browser)")
        print("2. Click '⚙️ Settings & Configuration'")
        print("3. Enter your API key and click 'Save'")
        print("4. Get your key from: https://openrouter.ai/keys")
        print("\nAlternatively, edit .env file manually.")
        print("\n" + "="*70 + "\n")

    # Create and launch interface
    app = create_interface()

    print("\n" + "="*70)
    print("FBI-STYLE BEHAVIORAL PROFILING SYSTEM")
    print("="*70)
    print("\n🚀 Launching Gradio interface...")
    print("📊 System ready for multimodal analysis")
    if config.has_api_key():
        print("✅ API key configured")
    else:
        print("⚠️  Configure API key in Settings panel")
    print("\n" + "="*70 + "\n")

    # Find an available port starting from 7861
    import socket

    def find_available_port(start_port=7861, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return start_port  # Fallback

    port = find_available_port()
    print(f"🌐 Starting on port {port}")

    app.launch(
        server_name="localhost",
        server_port=port,
        share=False,
        show_error=True
    )
