"""
Main profiling orchestration module.
Coordinates the 6-stage behavioral analysis pipeline.
"""

import json
import time
import logging
from typing import Dict, Callable, Optional
from datetime import datetime
from dataclasses import dataclass

from logger import AnalysisLogger
from audio_extractor import extract_audio_from_video
from api_client import OpenRouterClient
from prompts import (
    SAM_CHRISTENSEN_PROMPT,
    GEMINI_COMPREHENSIVE_PROMPT,
    FBI_SYNTHESIS_PROMPT,
    AUDIO_ANALYSIS_PROMPT,
    LIWC_ANALYSIS_PROMPT
)
from models_config import (
    StageModelConfig,
    DEFAULT_MODEL_CONFIG,
    get_model_info,
    validate_model_for_stage
)
from cache_manager import check_cache, store_in_cache, get_cache
from prompt_templates import get_template_manager, get_prompt_for_stage
from transcription import transcribe_audio, format_transcript_for_display, TranscriptionResult
from confidence_scoring import (
    calculate_analysis_confidence,
    add_confidence_to_result,
    format_confidence_for_display
)
from parallel_executor import run_stage_parallel, AnalysisTask

# Import modular executor for focused sub-analyses
try:
    from modular_executor import ModularAnalysisExecutor, format_modular_results
    MODULAR_AVAILABLE = True
except ImportError:
    MODULAR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelSelection:
    """Model selection for each pipeline stage."""
    essence_model: str = "google/gemini-3-pro-preview"
    multimodal_model: str = "google/gemini-3-pro-preview"
    audio_model: str = "google/gemini-3-pro-preview"
    liwc_model: str = "google/gemini-3-pro-preview"
    synthesis_model: str = "google/gemini-3-pro-preview"


@dataclass
class CustomPrompts:
    """Optional custom prompts to override defaults for each stage."""
    essence_prompt: Optional[str] = None
    multimodal_prompt: Optional[str] = None
    audio_prompt: Optional[str] = None
    liwc_prompt: Optional[str] = None
    synthesis_prompt: Optional[str] = None


class BehavioralProfiler:
    """
    Orchestrates the complete FBI-style behavioral profiling pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_config: Optional[ModelSelection] = None,
        custom_prompts: Optional[CustomPrompts] = None,
        use_template_prompts: bool = True
    ):
        """
        Initialize the profiler with OpenRouter client.

        Args:
            api_key: Optional OpenRouter API key (uses env var if not provided)
            model_config: Optional model selection for each stage
            custom_prompts: Optional custom prompts to override defaults
            use_template_prompts: If True, use prompts from template manager (default)
        """
        self.client = OpenRouterClient(api_key=api_key)
        self.model_config = model_config or ModelSelection()
        self.custom_prompts = custom_prompts
        self.use_template_prompts = use_template_prompts
        self.analysis_logger = AnalysisLogger()

    def _get_prompt(self, stage: str) -> str:
        """
        Get the prompt for a given stage.

        Priority:
        1. Custom prompts passed directly (if provided)
        2. Template manager active prompts (if use_template_prompts=True)
        3. Default prompts from prompts.py

        Args:
            stage: The analysis stage (essence, multimodal, audio, liwc, synthesis)

        Returns:
            The prompt text to use
        """
        # Check for direct custom prompts first
        if self.custom_prompts:
            prompt_map = {
                'essence': self.custom_prompts.essence_prompt,
                'multimodal': self.custom_prompts.multimodal_prompt,
                'audio': self.custom_prompts.audio_prompt,
                'liwc': self.custom_prompts.liwc_prompt,
                'synthesis': self.custom_prompts.synthesis_prompt
            }
            if prompt_map.get(stage):
                return prompt_map[stage]

        # Use template manager if enabled
        if self.use_template_prompts:
            return get_prompt_for_stage(stage)

        # Fall back to defaults
        default_map = {
            'essence': SAM_CHRISTENSEN_PROMPT,
            'multimodal': GEMINI_COMPREHENSIVE_PROMPT,
            'audio': AUDIO_ANALYSIS_PROMPT,
            'liwc': LIWC_ANALYSIS_PROMPT,
            'synthesis': FBI_SYNTHESIS_PROMPT
        }
        return default_map.get(stage, "")

    def profile_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        results_callback: Optional[Callable[[str, str], None]] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Run complete behavioral profiling pipeline on video.

        Uses modular sub-analysis architecture with native Gemini video handling.
        Each analysis aspect runs as a separate, parallel call for better accuracy.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback function(status_message, step_number)
            results_callback: Optional callback for streaming partial results(stage_name, result_text)
            use_cache: Whether to check/use cached results

        Returns:
            Dictionary containing all analyses and metadata

        Raises:
            Exception: If any stage of the pipeline fails
        """
        start_time = time.time()
        case_id = self._generate_case_id()

        # Initialize analysis logging with case ID
        self.analysis_logger.set_case_id(case_id)
        self.analysis_logger.analysis_start(video_path)

        # Build models config dict for cache key
        models_config = {
            'essence': self.model_config.essence_model,
            'multimodal': self.model_config.multimodal_model,
            'audio': self.model_config.audio_model,
            'liwc': self.model_config.liwc_model,
            'synthesis': self.model_config.synthesis_model
        }

        # Check cache first
        if use_cache:
            self._update_progress(
                progress_callback,
                "🔍 Checking cache for existing analysis...",
                0
            )
            cache_hit, cached_result = check_cache(video_path, models_config)
            if cache_hit and cached_result:
                self._update_progress(
                    progress_callback,
                    "✓ CACHE HIT - Returning cached analysis",
                    6
                )
                # Update timestamp and add cache info
                cached_result['retrieved_from_cache'] = True
                cached_result['cache_timestamp'] = cached_result.get('timestamp', 'unknown')
                cached_result['timestamp'] = datetime.now().isoformat()
                cached_result['processing_time_seconds'] = round(time.time() - start_time, 2)
                logger.info(f"Returning cached result for {video_path}")
                return cached_result

        try:
            # STEP 1: Load video and get metadata
            self._update_progress(
                progress_callback,
                "⚙️ STEP 1/6: Loading video for native processing...",
                1
            )

            # Get video metadata
            import cv2
            temp_cap = cv2.VideoCapture(video_path)
            total_video_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
            video_duration = total_video_frames / video_fps if video_fps > 0 else 0
            video_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            temp_cap.release()

            # Read video file as base64 for native Gemini processing
            import base64
            with open(video_path, 'rb') as f:
                base64_video = base64.b64encode(f.read()).decode('utf-8')

            video_metadata = {
                'duration_seconds': video_duration,
                'fps': video_fps,
                'total_frames': total_video_frames,
                'resolution': (video_width, video_height),
                'native_video_processing': True
            }

            logger.info(f"Video loaded: {video_duration:.1f}s, {video_width}x{video_height}, {len(base64_video) / 1024 / 1024:.1f}MB base64")

            self._update_progress(
                progress_callback,
                f"✓ STEP 1/6: Video loaded ({video_duration:.1f}s, native processing)",
                1
            )

            # STEP 2: Extract audio (do this once, reuse for all audio-based analyses)
            self._update_progress(
                progress_callback,
                "🎵 STEP 2/6: Extracting audio from video...",
                2
            )

            base64_audio = None
            audio_metadata = {}
            transcription_result = None
            try:
                base64_audio, audio_metadata = extract_audio_from_video(video_path)
                self._update_progress(
                    progress_callback,
                    "✓ STEP 2/6: Audio extracted, transcribing speech...",
                    2
                )

                # Transcribe the audio
                try:
                    transcription_result = transcribe_audio(
                        base64_audio=base64_audio,
                        api_client=self.client,
                        model=self.model_config.audio_model,
                        timeout=180
                    )
                    if transcription_result.success:
                        self._update_progress(
                            progress_callback,
                            f"✓ STEP 2/6: Audio transcribed ({transcription_result.word_count} words)",
                            2
                        )
                        # Stream transcript result
                        transcript_text = f"TRANSCRIPT:\n{transcription_result.transcript}\n\nSUMMARY:\n{transcription_result.summary}"
                        self._send_result(results_callback, 'transcript', transcript_text)
                    else:
                        logger.warning(f"Transcription failed: {transcription_result.error}")
                except Exception as trans_err:
                    logger.warning(f"Transcription error: {trans_err}")

            except Exception as e:
                self._update_progress(
                    progress_callback,
                    f"⚠️ STEP 2/6: Audio extraction failed ({str(e)}), continuing without audio",
                    2
                )

            # Get model names for display
            essence_model_name = get_model_info(self.model_config.essence_model)
            essence_display = essence_model_name.name if essence_model_name else self.model_config.essence_model
            multimodal_model_name = get_model_info(self.model_config.multimodal_model)
            multimodal_display = multimodal_model_name.name if multimodal_model_name else self.model_config.multimodal_model
            audio_model_name = get_model_info(self.model_config.audio_model)
            audio_display = audio_model_name.name if audio_model_name else self.model_config.audio_model
            synthesis_model_name = get_model_info(self.model_config.synthesis_model)
            synthesis_display = synthesis_model_name.name if synthesis_model_name else self.model_config.synthesis_model

            # MODULAR EXECUTION: Use focused sub-analyses with native video
            if MODULAR_AVAILABLE:
                self._update_progress(
                    progress_callback,
                    "⚡ Running modular analysis pipeline (native video)...",
                    3
                )

                executor = ModularAnalysisExecutor(
                    api_client=self.client,
                    max_workers=4,
                    max_tokens_sub=2000,
                    max_tokens_synthesis=3000,
                    temperature=0.7
                )

                # Run the full modular pipeline with native video
                modular_results = executor.run_full_pipeline(
                    video=base64_video,
                    audio=base64_audio,
                    visual_model=self.model_config.essence_model,
                    multimodal_model=self.model_config.multimodal_model,
                    audio_model=self.model_config.audio_model,
                    synthesis_model=self.model_config.synthesis_model,
                    progress_callback=progress_callback,
                    results_callback=results_callback
                )

                # Format results for compatibility with existing structure
                formatted = format_modular_results(modular_results)

                # Map to existing analysis structure
                essence_analysis = formatted.get('essence', 'Analysis unavailable')
                multimodal_analysis = formatted.get('multimodal', 'Analysis unavailable')
                audio_analysis = formatted.get('audio', 'Analysis unavailable')
                liwc_analysis = formatted.get('audio', 'Integrated into audio sub-analyses')
                fbi_profile = formatted.get('fbi_profile', 'Synthesis unavailable')

                # Get sub-analysis data for visualizations
                personality_analysis = formatted.get('personality', '')
                threat_analysis = formatted.get('threat', '')

                # Calculate execution times from modular results
                total_modular_time = sum(r.execution_time for r in modular_results.values())
                logger.info(f"Modular pipeline completed in {total_modular_time:.2f}s")

            else:
                # FALLBACK: Legacy parallel execution if modular unavailable
                logger.warning("Modular executor unavailable, using legacy parallel execution")
                self._update_progress(
                    progress_callback,
                    f"⚡ STEPS 3-5/6: Running analyses in parallel (legacy mode)...",
                    3
                )

                # Define analysis functions
                def run_essence():
                    return self.client.analyze_with_vision(
                        prompt=self._get_prompt('essence'),
                        base64_images=base64_frames,
                        model=self.model_config.essence_model,
                        max_tokens=3000,
                        temperature=0.7,
                        timeout=120
                    )

                def run_multimodal():
                    return self.client.analyze_with_multimodal(
                        prompt=self._get_prompt('multimodal'),
                        base64_images=base64_frames,
                        base64_audio=base64_audio,
                        model=self.model_config.multimodal_model,
                        max_tokens=4000,
                        temperature=0.7,
                        timeout=180
                    )

                def run_audio():
                    if not base64_audio:
                        return "Audio analysis unavailable: No audio could be extracted"
                    return self.client.analyze_audio(
                        prompt=self._get_prompt('audio'),
                        base64_audio=base64_audio,
                        model=self.model_config.audio_model,
                        max_tokens=3000,
                        temperature=0.7,
                        timeout=120
                    )

                def run_liwc():
                    if not base64_audio:
                        return "LIWC linguistic analysis unavailable: No audio could be extracted"
                    return self.client.analyze_audio(
                        prompt=self._get_prompt('liwc'),
                        base64_audio=base64_audio,
                        model=self.model_config.liwc_model,
                        max_tokens=4000,
                        temperature=0.7,
                        timeout=180
                    )

                # Callback for streaming results as they complete
                def on_stage_complete(stage, result):
                    self._send_result(results_callback, stage, result)
                    logger.info(f"Parallel stage '{stage}' completed")

                # Run all analyses in parallel
                parallel_results = run_stage_parallel(
                    essence_func=run_essence,
                    multimodal_func=run_multimodal,
                    audio_func=run_audio,
                    liwc_func=run_liwc,
                    on_complete=on_stage_complete
                )

                # Extract results
                essence_analysis = parallel_results['essence'].result
                multimodal_analysis = parallel_results['multimodal'].result
                audio_analysis = parallel_results['audio'].result
                liwc_analysis = parallel_results['liwc'].result

                # Calculate total parallel execution time
                parallel_time = max(r.execution_time for r in parallel_results.values())
                logger.info(f"Parallel execution completed in {parallel_time:.2f}s")

                self._update_progress(
                    progress_callback,
                    f"✓ STEPS 3-5/6: All parallel analyses complete ({parallel_time:.1f}s)",
                    5
                )

                # Legacy mode needs FBI synthesis step
                self._update_progress(
                    progress_callback,
                    f"🎯 STEP 6/6: Synthesizing FBI behavioral profile...",
                    6
                )

                combined_analyses = f"""ANALYSIS 1: SAM CHRISTENSEN VISUAL ESSENCE PROFILE
{essence_analysis}

ANALYSIS 2: COMPREHENSIVE MULTIMODAL BEHAVIORAL ANALYSIS
{multimodal_analysis}

ANALYSIS 3: AUDIO/VOICE ANALYSIS
{audio_analysis}

ANALYSIS 4: LIWC-STYLE LINGUISTIC ANALYSIS
{liwc_analysis}"""

                fbi_profile = self.client.synthesize_text(
                    prompt=self._get_prompt('synthesis'),
                    previous_analyses=combined_analyses,
                    model=self.model_config.synthesis_model,
                    max_tokens=4000,
                    temperature=0.7,
                    timeout=120
                )
                self._send_result(results_callback, 'synthesis', fbi_profile)

            # Calculate total processing time
            processing_time = time.time() - start_time

            # Compile complete result
            result = {
                'case_id': case_id,
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'video_metadata': video_metadata,
                'audio_metadata': audio_metadata,
                'models_used': {
                    'essence': self.model_config.essence_model,
                    'multimodal': self.model_config.multimodal_model,
                    'audio': self.model_config.audio_model,
                    'liwc': self.model_config.liwc_model,
                    'synthesis': self.model_config.synthesis_model
                },
                'analyses': {
                    'sam_christensen_essence': essence_analysis,
                    'multimodal_behavioral': multimodal_analysis,
                    'audio_voice_analysis': audio_analysis,
                    'liwc_linguistic_analysis': liwc_analysis,
                    'fbi_behavioral_synthesis': fbi_profile,
                    'personality_synthesis': personality_analysis if MODULAR_AVAILABLE else '',
                    'threat_synthesis': threat_analysis if MODULAR_AVAILABLE else ''
                },
                'transcription': {
                    'transcript': transcription_result.transcript if transcription_result and transcription_result.success else '',
                    'summary': transcription_result.summary if transcription_result and transcription_result.success else '',
                    'speakers': transcription_result.speakers if transcription_result and transcription_result.success else [],
                    'word_count': transcription_result.word_count if transcription_result and transcription_result.success else 0,
                    'audio_quality': transcription_result.audio_quality if transcription_result and transcription_result.success else '',
                    'available': transcription_result.success if transcription_result else False
                },
                'status': 'completed'
            }

            # Add confidence scoring
            result = add_confidence_to_result(result)
            logger.info(f"Confidence score: {result.get('confidence', {}).get('overall', 'N/A')}")

            self._update_progress(
                progress_callback,
                f"✓ ALL STEPS COMPLETE - Processing time: {processing_time:.1f}s",
                7  # Step 7 = 100% complete
            )

            # Log analysis completion
            self.analysis_logger.analysis_complete(processing_time)

            # Store result in cache
            if use_cache:
                try:
                    store_in_cache(video_path, models_config, result)
                    logger.info(f"Cached analysis result for {video_path}")
                except Exception as cache_err:
                    logger.warning(f"Failed to cache result: {cache_err}")

            return result

        except Exception as e:
            # Return error result
            error_result = {
                'case_id': case_id,
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(time.time() - start_time, 2),
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }

            self._update_progress(
                progress_callback,
                f"⚠️ ANALYSIS FAILED: {str(e)}",
                0
            )

            # Log analysis failure
            self.analysis_logger.analysis_failed(str(e))

            raise Exception(f"Profiling failed: {str(e)}")

    def _generate_case_id(self) -> str:
        """
        Generate unique case ID for this analysis.

        Returns:
            Case ID string in format: PROF-YYYYMMDD-HHMMSS-XXX
        """
        timestamp = datetime.now()
        random_suffix = str(int(time.time() * 1000))[-3:]
        case_id = f"PROF-{timestamp.strftime('%Y%m%d-%H%M%S')}-{random_suffix}"
        return case_id

    def _update_progress(
        self,
        callback: Optional[Callable[[str, int], None]],
        message: str,
        step: int
    ):
        """
        Send progress update via callback if provided.

        Args:
            callback: Progress callback function
            message: Status message
            step: Current step number (0-6)
        """
        if callback:
            callback(message, step)

    def _send_result(
        self,
        callback: Optional[Callable[[str, str], None]],
        stage: str,
        result: str
    ):
        """
        Send partial result via callback for streaming updates.

        Args:
            callback: Results callback function
            stage: Analysis stage name (essence, multimodal, audio, liwc, synthesis, transcript)
            result: The analysis result text
        """
        if callback:
            try:
                callback(stage, result)
            except Exception as e:
                logger.warning(f"Failed to send result for {stage}: {e}")

    @staticmethod
    def export_to_json(result: Dict, output_path: str):
        """
        Export profiling result to JSON file.

        Args:
            result: Profiling result dictionary
            output_path: Path for output JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    @staticmethod
    def format_result_for_display(result: Dict) -> Dict[str, str]:
        """
        Format result dictionary for Gradio display.
        Extracts main sections into separate display strings.

        Args:
            result: Complete profiling result

        Returns:
            Dictionary with formatted display strings for each tab
        """
        if result.get('status') == 'failed':
            error_msg = f"""⚠️ ANALYSIS FAILED

Error Type: {result.get('error_type', 'Unknown')}
Error Message: {result.get('error', 'No details available')}

Case ID: {result.get('case_id', 'N/A')}
Timestamp: {result.get('timestamp', 'N/A')}
"""
            return {
                'essence': error_msg,
                'multimodal': error_msg,
                'fbi_profile': error_msg,
                'transcript': error_msg,
                'confidence': error_msg,
                'json': json.dumps(result, indent=2)
            }

        analyses = result.get('analyses', {})

        # Format metadata header
        metadata = result.get('video_metadata', {})
        header = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CASE ID: {result.get('case_id', 'N/A')}
TIMESTAMP: {result.get('timestamp', 'N/A')}
PROCESSING TIME: {result.get('processing_time_seconds', 0):.2f}s
VIDEO DURATION: {metadata.get('duration_seconds', 0):.1f}s
FRAMES ANALYZED: {metadata.get('frames_extracted', 0)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

        # Format transcription section
        transcription_data = result.get('transcription', {})
        if transcription_data.get('available'):
            transcript_text = f"""AUDIO TRANSCRIPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audio Quality: {transcription_data.get('audio_quality', 'Unknown')}
Word Count: ~{transcription_data.get('word_count', 0)}
Speakers: {len(transcription_data.get('speakers', []))}

TRANSCRIPT:
{transcription_data.get('transcript', 'No transcript available')}

SUMMARY:
{transcription_data.get('summary', 'No summary available')}
"""
        else:
            transcript_text = "Transcription not available for this analysis."

        # Format confidence section
        confidence_data = result.get('confidence', {})
        if confidence_data:
            confidence_text = f"""ANALYSIS CONFIDENCE ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERALL CONFIDENCE: {confidence_data.get('overall', 0):.0%} ({confidence_data.get('level', 'unknown').upper()})
DATA QUALITY SCORE: {confidence_data.get('data_quality', 0):.0%}

COMPONENT SCORES:
"""
            for component in confidence_data.get('components', []):
                bar_len = int(component.get('score', 0) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                confidence_text += f"\n  {component.get('category', 'Unknown')}:\n"
                confidence_text += f"    [{bar}] {component.get('score', 0):.0%} ({component.get('level', 'unknown')})\n"
                if component.get('reasoning'):
                    confidence_text += f"    └─ {component.get('reasoning')}\n"

            if confidence_data.get('warnings'):
                confidence_text += "\nWARNINGS:\n"
                for warning in confidence_data.get('warnings', []):
                    confidence_text += f"  ⚠ {warning}\n"

            confidence_text += """
INTERPRETATION GUIDE:
  Very High (80-100%): Strong data, consistent indicators
  High (60-80%): Good data, reliable analysis
  Moderate (30-60%): Limited data or mixed signals
  Low (0-30%): Insufficient data, use with caution
"""
        else:
            confidence_text = "Confidence scoring not available for this analysis."

        return {
            'essence': header + analyses.get('sam_christensen_essence', 'No data'),
            'multimodal': header + analyses.get('multimodal_behavioral', 'No data'),
            'audio': header + analyses.get('audio_voice_analysis', 'No data'),
            'liwc': header + analyses.get('liwc_linguistic_analysis', 'No data'),
            'fbi_profile': header + analyses.get('fbi_behavioral_synthesis', 'No data'),
            'transcript': header + transcript_text,
            'confidence': header + confidence_text,
            'json': json.dumps(result, indent=2)
        }


# ==================================================================================
# DEVELOPER META-ANALYSIS - TO REMOVE BEFORE PRODUCTION
# ==================================================================================
def run_dev_meta_analysis(
    result: Dict,
    api_key: Optional[str] = None,
    model: str = "google/gemini-3-pro-preview"
) -> str:
    """
    Run developer meta-analysis on a completed profiling result.
    Sends the full report to an AI for suggestions on improving the profiler.

    NOTE: This function is for DEVELOPMENT ONLY and should be REMOVED before production.

    Args:
        result: Complete profiling result dictionary
        api_key: Optional OpenRouter API key (uses env var if not provided)
        model: Model to use for meta-analysis (default: Gemini 3 Pro)

    Returns:
        Meta-analysis feedback string
    """
    from prompts import DEV_META_ANALYSIS_PROMPT

    # Build full report content
    analyses = result.get('analyses', {})

    full_report = f"""
=== PROFILING RESULT ===

CASE ID: {result.get('case_id', 'N/A')}
PROCESSING TIME: {result.get('processing_time_seconds', 0):.2f} seconds
STATUS: {result.get('status', 'unknown')}

MODELS USED:
{json.dumps(result.get('models_used', {}), indent=2)}

VIDEO METADATA:
{json.dumps(result.get('video_metadata', {}), indent=2)}

=== SAM CHRISTENSEN ESSENCE ANALYSIS ===
{analyses.get('sam_christensen_essence', 'Not available')}

=== MULTIMODAL BEHAVIORAL ANALYSIS ===
{analyses.get('multimodal_behavioral', 'Not available')}

=== AUDIO/VOICE ANALYSIS ===
{analyses.get('audio_voice_analysis', 'Not available')}

=== LIWC LINGUISTIC ANALYSIS ===
{analyses.get('liwc_linguistic_analysis', 'Not available')}

=== FBI BEHAVIORAL SYNTHESIS ===
{analyses.get('fbi_behavioral_synthesis', 'Not available')}
"""

    try:
        # Create API client
        from api_client import OpenRouterClient
        from config_manager import ConfigManager

        if not api_key:
            config = ConfigManager()
            api_key = config.load_api_key()

        client = OpenRouterClient(api_key=api_key)

        # Call the meta-analysis model
        logger.info(f"[DEV] Running meta-analysis with model: {model}")

        meta_analysis = client.synthesize_text(
            prompt=DEV_META_ANALYSIS_PROMPT,
            previous_analyses=full_report,
            model=model,
            max_tokens=8000,  # Longer for comprehensive feedback
            temperature=0.7,
            timeout=180  # Allow more time for deep analysis
        )

        logger.info("[DEV] Meta-analysis complete")
        return meta_analysis

    except Exception as e:
        error_msg = f"Meta-analysis failed: {str(e)}"
        logger.error(f"[DEV] {error_msg}")
        return error_msg


# Convenience function for quick profiling
def profile_video_file(
    video_path: str,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> Dict:
    """
    Quick function to profile a video file.

    Args:
        video_path: Path to video file
        api_key: Optional OpenRouter API key
        progress_callback: Optional progress callback

    Returns:
        Complete profiling result dictionary
    """
    profiler = BehavioralProfiler(api_key=api_key)
    return profiler.profile_video(video_path, progress_callback)
