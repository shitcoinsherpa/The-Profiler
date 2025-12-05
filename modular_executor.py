"""
Modular execution engine for parallel sub-analysis processing.
Handles data flow between dependent analysis stages.
"""

import logging
import time
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from modular_prompts import (
    VISUAL_PROMPTS,
    MULTIMODAL_PROMPTS,
    AUDIO_PROMPTS,
    SYNTHESIS_PROMPTS,
)

logger = logging.getLogger(__name__)


@dataclass
class SubAnalysisResult:
    """Result from a single sub-analysis."""
    name: str
    stage: str
    result: str
    execution_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class StageResult:
    """Aggregated results from a complete stage."""
    stage: str
    sub_results: Dict[str, SubAnalysisResult]
    combined_text: str
    execution_time: float
    success: bool


class ModularAnalysisExecutor:
    """
    Executes modular sub-analyses in parallel with proper data flow.

    Execution flow:
    1. Visual sub-analyses (parallel) - needs: images
    2. Multimodal sub-analyses (parallel) - needs: images + audio
    3. Audio sub-analyses (parallel) - needs: audio
    4. Synthesis sub-analyses (parallel) - needs: all previous results
    5. Final integration - needs: all previous + synthesis results
    """

    def __init__(
        self,
        api_client,
        max_workers: int = 4,
        max_tokens_sub: int = 8000,
        max_tokens_synthesis: int = 16000,
        temperature: float = 0.7
    ):
        """
        Initialize the modular executor.

        Args:
            api_client: OpenRouter API client instance
            max_workers: Max parallel API calls per stage
            max_tokens_sub: Max tokens for sub-analysis responses
            max_tokens_synthesis: Max tokens for synthesis responses
            temperature: Model temperature setting
        """
        self.client = api_client
        self.max_workers = max_workers
        self.max_tokens_sub = max_tokens_sub
        self.max_tokens_synthesis = max_tokens_synthesis
        self.temperature = temperature

    def _run_sub_analysis(
        self,
        name: str,
        stage: str,
        prompt: str,
        model: str,
        video: str = None,
        audio: str = None,
        timeout: int = 90,
        max_retries: int = 3
    ) -> SubAnalysisResult:
        """Run a single sub-analysis with retry logic to ensure completion."""
        start_time = time.time()
        response_format = None  # Not using structured output
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 * attempt  # Exponential backoff
                    logger.info(f"Retry {attempt}/{max_retries} for '{name}' sub-analysis (waiting {wait_time}s)")
                    time.sleep(wait_time)

                if video and audio:
                    # Full multimodal call with native video + audio
                    result = self.client.analyze_with_multimodal(
                        prompt=prompt,
                        base64_video=video,
                        base64_audio=audio,
                        model=model,
                        max_tokens=self.max_tokens_sub,
                        temperature=self.temperature,
                        timeout=timeout,
                        response_format=response_format
                    )
                elif video:
                    # Video-only call (Gemini native video handling)
                    result = self.client.analyze_with_multimodal(
                        prompt=prompt,
                        base64_video=video,
                        model=model,
                        max_tokens=self.max_tokens_sub,
                        temperature=self.temperature,
                        timeout=timeout,
                        response_format=response_format
                    )
                elif audio:
                    # Audio call
                    result = self.client.analyze_audio(
                        prompt=prompt,
                        base64_audio=audio,
                        model=model,
                        max_tokens=self.max_tokens_sub,
                        temperature=self.temperature,
                        timeout=timeout,
                        response_format=response_format
                    )
                else:
                    # Text-only call (synthesis)
                    result = self.client.synthesize_text(
                        prompt=prompt,
                        previous_analyses="",  # Context already in prompt
                        model=model,
                        max_tokens=self.max_tokens_synthesis,
                        temperature=self.temperature,
                        timeout=timeout,
                        response_format=response_format
                    )

                # Success - return result
                execution_time = time.time() - start_time
                logger.info(f"Sub-analysis '{name}' completed in {execution_time:.2f}s")

                return SubAnalysisResult(
                    name=name,
                    stage=stage,
                    result=result,
                    execution_time=execution_time,
                    success=True
                )

            except Exception as e:
                last_error = e
                logger.warning(f"Sub-analysis '{name}' attempt {attempt + 1}/{max_retries} failed: {e}")

        # All retries exhausted
        execution_time = time.time() - start_time
        logger.error(f"Sub-analysis '{name}' failed after {max_retries} attempts: {last_error}")

        return SubAnalysisResult(
            name=name,
            stage=stage,
            result=f"ERROR after {max_retries} retries: {str(last_error)}",
            execution_time=execution_time,
            success=False,
            error=str(last_error)
        )

    def _run_parallel_sub_analyses(
        self,
        prompts: Dict[str, str],
        stage: str,
        model: str,
        video: str = None,
        audio: str = None,
        context: str = None,
        on_complete: Callable[[str, str], None] = None
    ) -> StageResult:
        """
        Run multiple sub-analyses in parallel.

        Args:
            prompts: Dict of {name: prompt_text}
            stage: Stage identifier
            model: Model ID to use
            video: Optional base64 video (native Gemini video handling)
            audio: Optional base64 audio
            context: Optional context to inject into prompts
            on_complete: Callback when each sub-analysis completes

        Returns:
            StageResult with all sub-results
        """
        start_time = time.time()
        sub_results = {}

        # Prepare prompts with context injection if needed
        prepared_prompts = {}
        for name, prompt in prompts.items():
            if context and '{previous_analyses}' in prompt:
                prepared_prompts[name] = prompt.format(previous_analyses=context)
            elif context and '{synthesis_results}' in prompt:
                # For final integration, context contains previous_analyses
                # We'll handle synthesis_results separately
                prepared_prompts[name] = prompt
            else:
                prepared_prompts[name] = prompt

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for name, prompt in prepared_prompts.items():
                future = executor.submit(
                    self._run_sub_analysis,
                    name=name,
                    stage=stage,
                    prompt=prompt,
                    model=model,
                    video=video,
                    audio=audio
                )
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    sub_results[name] = result

                    if on_complete and result.success:
                        on_complete(name, result.result)

                except Exception as e:
                    logger.error(f"Future failed for {name}: {e}")
                    sub_results[name] = SubAnalysisResult(
                        name=name,
                        stage=stage,
                        result=f"ERROR: {str(e)}",
                        execution_time=0,
                        success=False,
                        error=str(e)
                    )

        # Combine successful results
        combined_parts = []
        for name, result in sub_results.items():
            if result.success:
                combined_parts.append(f"=== {name.upper().replace('_', ' ')} ===\n{result.result}")

        combined_text = "\n\n".join(combined_parts)
        execution_time = time.time() - start_time
        success = any(r.success for r in sub_results.values())

        logger.info(f"Stage '{stage}' completed in {execution_time:.2f}s with {sum(1 for r in sub_results.values() if r.success)}/{len(sub_results)} successful")

        return StageResult(
            stage=stage,
            sub_results=sub_results,
            combined_text=combined_text,
            execution_time=execution_time,
            success=success
        )

    def run_visual_analysis(
        self,
        video: str,
        model: str,
        on_complete: Callable[[str, str], None] = None
    ) -> StageResult:
        """
        Run visual sub-analyses in parallel using native video.

        Args:
            video: Base64-encoded video (Gemini native handling)
            model: Vision model ID
            on_complete: Callback for each completed sub-analysis

        Returns:
            StageResult with visual analysis
        """
        logger.info(f"Starting visual analysis with {len(VISUAL_PROMPTS)} sub-analyses (native video)")

        return self._run_parallel_sub_analyses(
            prompts=VISUAL_PROMPTS,
            stage='visual',
            model=model,
            video=video,
            on_complete=on_complete
        )

    def run_multimodal_analysis(
        self,
        video: str,
        audio: str,
        model: str,
        on_complete: Callable[[str, str], None] = None
    ) -> StageResult:
        """
        Run multimodal sub-analyses in parallel using native video.

        Args:
            video: Base64-encoded video (Gemini native handling)
            audio: Base64-encoded audio
            model: Multimodal model ID
            on_complete: Callback for each completed sub-analysis

        Returns:
            StageResult with multimodal analysis
        """
        logger.info(f"Starting multimodal analysis with {len(MULTIMODAL_PROMPTS)} sub-analyses (native video)")

        return self._run_parallel_sub_analyses(
            prompts=MULTIMODAL_PROMPTS,
            stage='multimodal',
            model=model,
            video=video,
            audio=audio,
            on_complete=on_complete
        )

    def run_audio_analysis(
        self,
        audio: str,
        model: str,
        on_complete: Callable[[str, str], None] = None
    ) -> StageResult:
        """
        Run audio sub-analyses in parallel.

        Args:
            audio: Base64-encoded audio
            model: Audio-capable model ID
            on_complete: Callback for each completed sub-analysis

        Returns:
            StageResult with audio analysis
        """
        logger.info(f"Starting audio analysis with {len(AUDIO_PROMPTS)} sub-analyses")

        return self._run_parallel_sub_analyses(
            prompts=AUDIO_PROMPTS,
            stage='audio',
            model=model,
            audio=audio,
            on_complete=on_complete
        )

    def run_synthesis(
        self,
        previous_analyses: str,
        model: str,
        on_complete: Callable[[str, str], None] = None
    ) -> StageResult:
        """
        Run synthesis sub-analyses in parallel, then final integration.

        Args:
            previous_analyses: Combined text from all previous stages
            model: Synthesis model ID
            on_complete: Callback for each completed sub-analysis

        Returns:
            StageResult with synthesis
        """
        logger.info(f"Starting synthesis with {len(SYNTHESIS_PROMPTS)} sub-analyses")

        # First, run parallel synthesis sub-analyses (personality, threat, etc.)
        # Exclude 'final' which needs all synthesis results
        parallel_prompts = {k: v for k, v in SYNTHESIS_PROMPTS.items() if k != 'final'}

        parallel_result = self._run_parallel_sub_analyses(
            prompts=parallel_prompts,
            stage='synthesis_parallel',
            model=model,
            context=previous_analyses,
            on_complete=on_complete
        )

        # Then run final integration with all data
        synthesis_text = parallel_result.combined_text

        final_prompt = SYNTHESIS_PROMPTS['final'].format(
            previous_analyses=previous_analyses,
            synthesis_results=synthesis_text
        )

        logger.info("Running final integration")
        final_result = self._run_sub_analysis(
            name='final_integration',
            stage='synthesis_final',
            prompt=final_prompt,
            model=model
        )

        if on_complete and final_result.success:
            on_complete('final_integration', final_result.result)

        # Combine everything
        all_sub_results = dict(parallel_result.sub_results)
        all_sub_results['final_integration'] = final_result

        full_combined = f"{synthesis_text}\n\n=== FINAL INTEGRATION ===\n{final_result.result}"

        return StageResult(
            stage='synthesis',
            sub_results=all_sub_results,
            combined_text=full_combined,
            execution_time=parallel_result.execution_time + final_result.execution_time,
            success=final_result.success
        )

    def run_full_pipeline(
        self,
        video: str,
        audio: Optional[str],
        visual_model: str,
        multimodal_model: str,
        audio_model: str,
        synthesis_model: str,
        progress_callback: Callable[[str, int], None] = None,
        results_callback: Callable[[str, str], None] = None
    ) -> Dict[str, StageResult]:
        """
        Run the complete modular analysis pipeline using native video.

        Args:
            video: Base64-encoded video (Gemini native handling)
            audio: Optional base64-encoded audio
            visual_model: Model for visual analysis
            multimodal_model: Model for multimodal analysis
            audio_model: Model for audio analysis
            synthesis_model: Model for synthesis
            progress_callback: Progress update callback
            results_callback: Results streaming callback

        Returns:
            Dict of stage name to StageResult
        """
        all_results = {}

        def update_progress(msg, step):
            if progress_callback:
                progress_callback(msg, step)

        # Stage 1: Visual Analysis (parallel sub-analyses with native video)
        update_progress("🔍 Running visual sub-analyses (FACS, archetype, body language, deception)...", 3)
        visual_result = self.run_visual_analysis(
            video=video,
            model=visual_model,
            on_complete=results_callback
        )
        all_results['visual'] = visual_result
        update_progress(f"✓ Visual analysis complete ({len([r for r in visual_result.sub_results.values() if r.success])}/4 sub-analyses)", 3)

        # Stage 2: Multimodal Analysis (parallel sub-analyses with native video)
        if audio:
            update_progress("📊 Running multimodal sub-analyses (timeline, sync, environment, awareness)...", 4)
            multimodal_result = self.run_multimodal_analysis(
                video=video,
                audio=audio,
                model=multimodal_model,
                on_complete=results_callback
            )
            all_results['multimodal'] = multimodal_result
            update_progress(f"✓ Multimodal analysis complete ({len([r for r in multimodal_result.sub_results.values() if r.success])}/4 sub-analyses)", 4)
        else:
            all_results['multimodal'] = StageResult(
                stage='multimodal',
                sub_results={},
                combined_text="Audio unavailable - multimodal analysis skipped",
                execution_time=0,
                success=False
            )

        # Stage 3: Audio Analysis (parallel sub-analyses)
        if audio:
            update_progress("🎤 Running audio sub-analyses (voice, sociolinguistic, deception)...", 5)
            audio_result = self.run_audio_analysis(
                audio=audio,
                model=audio_model,
                on_complete=results_callback
            )
            all_results['audio'] = audio_result
            update_progress(f"✓ Audio analysis complete ({len([r for r in audio_result.sub_results.values() if r.success])}/3 sub-analyses)", 5)
        else:
            all_results['audio'] = StageResult(
                stage='audio',
                sub_results={},
                combined_text="Audio unavailable - audio analysis skipped",
                execution_time=0,
                success=False
            )

        # Combine all previous analyses for synthesis
        previous_analyses = f"""
=== VISUAL ANALYSIS ===
{visual_result.combined_text}

=== MULTIMODAL ANALYSIS ===
{all_results['multimodal'].combined_text}

=== AUDIO ANALYSIS ===
{all_results['audio'].combined_text}
"""

        # Stage 4: Synthesis (parallel sub-analyses + final integration)
        update_progress("🎯 Running synthesis (personality, threat, differential, contradictions, red team)...", 6)
        synthesis_result = self.run_synthesis(
            previous_analyses=previous_analyses,
            model=synthesis_model,
            on_complete=results_callback
        )
        all_results['synthesis'] = synthesis_result
        update_progress(f"✓ Synthesis complete ({len([r for r in synthesis_result.sub_results.values() if r.success])}/6 sub-analyses)", 6)

        return all_results


def format_modular_results(results: Dict[str, StageResult]) -> Dict[str, str]:
    """
    Format modular results for display.

    Args:
        results: Dict of stage results from run_full_pipeline

    Returns:
        Dict with formatted text for each display section
    """
    formatted = {}

    # Visual/Essence section
    if 'visual' in results:
        visual = results['visual']
        formatted['essence'] = visual.combined_text

        # Extract NCI visual sub-analyses for PDF
        for key in ['blink_rate', 'bte_scoring', 'facial_etching', 'gestural_mismatch', 'stress_clusters']:
            if key in visual.sub_results and visual.sub_results[key].success:
                formatted[key] = visual.sub_results[key].result

    # Multimodal section
    if 'multimodal' in results:
        multimodal = results['multimodal']
        formatted['multimodal'] = multimodal.combined_text

        # Extract NCI multimodal sub-analyses for PDF
        for key in ['five_cs', 'baseline_deviation']:
            if key in multimodal.sub_results and multimodal.sub_results[key].success:
                formatted[key] = multimodal.sub_results[key].result

    # Audio section
    if 'audio' in results:
        audio = results['audio']

        # Build audio output EXCLUDING liwc (which goes in its own section)
        audio_parts = []
        for name, result in audio.sub_results.items():
            if name != 'liwc' and result.success:  # Exclude LIWC from audio accordion
                audio_parts.append(f"=== {name.upper().replace('_', ' ')} ===
{result.result}")
        formatted['audio'] = "

".join(audio_parts) if audio_parts else audio.combined_text

        # Extract NCI audio sub-analyses for PDF
        for key in ['detail_mountain_valley', 'minimizing_language', 'linguistic_harvesting']:
            if key in audio.sub_results and audio.sub_results[key].success:
                formatted[key] = audio.sub_results[key].result

        # Use actual LIWC quantitative analysis result (separate accordion)
        if 'liwc' in audio.sub_results and audio.sub_results['liwc'].success:
            formatted['liwc'] = audio.sub_results['liwc'].result
        else:
            formatted['liwc'] = "LIWC analysis not available. Audio analysis may have failed or been skipped."

    # Synthesis/FBI Profile section
    if 'synthesis' in results:
        synthesis = results['synthesis']
        # Get the final integration specifically
        if 'final_integration' in synthesis.sub_results:
            formatted['fbi_profile'] = synthesis.sub_results['final_integration'].result
        else:
            formatted['fbi_profile'] = synthesis.combined_text

        # Also provide individual synthesis components
        formatted['personality'] = synthesis.sub_results.get('personality', SubAnalysisResult('', '', 'N/A', 0, False)).result
        formatted['threat'] = synthesis.sub_results.get('threat', SubAnalysisResult('', '', 'N/A', 0, False)).result
        formatted['differential'] = synthesis.sub_results.get('differential', SubAnalysisResult('', '', 'N/A', 0, False)).result
        formatted['contradictions'] = synthesis.sub_results.get('contradictions', SubAnalysisResult('', '', 'N/A', 0, False)).result
        formatted['red_team'] = synthesis.sub_results.get('red_team', SubAnalysisResult('', '', 'N/A', 0, False)).result

        # Extract NCI synthesis sub-analyses for PDF
        for key in ['fate_model', 'nci_deception_summary']:
            if key in synthesis.sub_results and synthesis.sub_results[key].success:
                formatted[key] = synthesis.sub_results[key].result

    return formatted
