"""

Modular execution engine for parallel sub-analysis processing.

Handles data flow between dependent analysis stages.

"""



import logging

import time

import random

from typing import Dict, List, Optional, Callable, Any

from concurrent.futures import ThreadPoolExecutor, as_completed

from dataclasses import dataclass



from prompts.modular_prompts import (

    STAGE_ZERO_PROMPTS,

    VISUAL_PROMPTS,

    MULTIMODAL_PROMPTS,

    AUDIO_PROMPTS,

    SYNTHESIS_PROMPTS,

)

from core.signal_collapsing import collapse_analysis_outputs



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



    # Retry configuration for API resilience
    DEFAULT_MAX_RETRIES = 10  # Increased from 3 for API outages
    SERVER_ERROR_CODES = ['500', '502', '503', '504', 'cloudflare', 'timeout', 'rate_limit']
    BASE_BACKOFF_SECONDS = 2.0
    MAX_BACKOFF_SECONDS = 120.0  # Cap at 2 minutes between retries
    SERVER_ERROR_BACKOFF_MULTIPLIER = 3.0  # Longer waits for server errors

    def __init__(

        self,

        api_client,

        max_workers: int = 4,

        max_tokens_sub: int = 8000,

        max_tokens_synthesis: int = 16000,

        temperature: float = 0.7,

        max_retries: int = None,

        persistent_retry: bool = True

    ):

        """

        Initialize the modular executor.



        Args:

            api_client: OpenRouter API client instance

            max_workers: Max parallel API calls per stage

            max_tokens_sub: Max tokens for sub-analysis responses

            max_tokens_synthesis: Max tokens for synthesis responses

            temperature: Model temperature setting

            max_retries: Max retry attempts per sub-analysis (default: 10)

            persistent_retry: If True, retry failed sub-analyses at stage level

        """

        self.client = api_client

        self.max_workers = max_workers

        self.max_tokens_sub = max_tokens_sub

        self.max_tokens_synthesis = max_tokens_synthesis

        self.temperature = temperature

        self.max_retries = max_retries or self.DEFAULT_MAX_RETRIES

        self.persistent_retry = persistent_retry



    def _is_server_error(self, error: Exception) -> bool:
        """Check if error is a server-side issue that warrants extended retry."""
        error_str = str(error).lower()
        for code in self.SERVER_ERROR_CODES:
            if code in error_str:
                return True
        return False

    def _calculate_backoff(self, attempt: int, is_server_error: bool) -> float:
        """Calculate backoff time with exponential increase and jitter."""
        base = self.BASE_BACKOFF_SECONDS
        if is_server_error:
            base *= self.SERVER_ERROR_BACKOFF_MULTIPLIER

        # Exponential backoff: base * 2^attempt
        backoff = base * (2 ** attempt)

        # Add jitter (±25%) to prevent thundering herd
        jitter = backoff * 0.25 * (2 * random.random() - 1)
        backoff += jitter

        # Cap at maximum
        return min(backoff, self.MAX_BACKOFF_SECONDS)

    def _run_sub_analysis(

        self,

        name: str,

        stage: str,

        prompt: str,

        model: str,

        video: str = None,

        audio: str = None,

        timeout: int = 120,

        max_retries: int = None

    ) -> SubAnalysisResult:

        """Run a single sub-analysis with robust retry logic for API resilience."""

        start_time = time.time()

        response_format = None  # Not using structured output

        last_error = None

        retries = max_retries or self.max_retries

        consecutive_server_errors = 0



        for attempt in range(retries):

            try:

                if attempt > 0:

                    is_server_error = self._is_server_error(last_error) if last_error else False

                    if is_server_error:
                        consecutive_server_errors += 1
                    else:
                        consecutive_server_errors = 0

                    wait_time = self._calculate_backoff(attempt, is_server_error)

                    # Log with appropriate severity
                    if consecutive_server_errors >= 3:
                        logger.warning(
                            f"API outage detected ({consecutive_server_errors} consecutive server errors). "
                            f"Retry {attempt}/{retries} for '{name}' (waiting {wait_time:.1f}s)"
                        )
                    else:
                        logger.info(f"Retry {attempt}/{retries} for '{name}' sub-analysis (waiting {wait_time:.1f}s)")

                    time.sleep(wait_time)



                if video and audio:

                    # Full multimodal call with native video + audio
                    logger.info(f"[API DEBUG] {name}: Calling analyze_with_multimodal (video+audio)")
                    logger.info(f"[API DEBUG] {name}: model={model}, prompt_len={len(prompt)}, video_len={len(video)}, audio_len={len(audio)}")

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
                    logger.info(f"[API DEBUG] {name}: Calling analyze_with_multimodal (video-only)")
                    logger.info(f"[API DEBUG] {name}: model={model}, prompt_len={len(prompt)}, video_len={len(video)}")

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
                    logger.info(f"[API DEBUG] {name}: Calling analyze_audio (audio-only)")
                    logger.info(f"[API DEBUG] {name}: model={model}, prompt_len={len(prompt)}, audio_len={len(audio)}")

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
                    logger.info(f"[API DEBUG] {name}: Calling synthesize_text (text-only)")
                    logger.info(f"[API DEBUG] {name}: model={model}, prompt_len={len(prompt)}")

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

                is_server_err = self._is_server_error(e)

                err_type = "SERVER ERROR" if is_server_err else "Error"

                logger.warning(f"Sub-analysis '{name}' attempt {attempt + 1}/{retries} - {err_type}: {e}")



        # All retries exhausted

        execution_time = time.time() - start_time

        logger.error(
            f"Sub-analysis '{name}' FAILED after {retries} attempts over {execution_time:.1f}s. "
            f"Last error: {last_error}"
        )



        return SubAnalysisResult(

            name=name,

            stage=stage,

            result=f"ERROR after {retries} retries ({execution_time:.0f}s): {str(last_error)}",

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

        on_complete: Callable[[str, str], None] = None,

        stage_retry_rounds: int = 3

    ) -> StageResult:

        """

        Run multiple sub-analyses in parallel with stage-level retry.



        Args:

            prompts: Dict of {name: prompt_text}

            stage: Stage identifier

            model: Model ID to use

            video: Optional base64 video (native Gemini video handling)

            audio: Optional base64 audio

            context: Optional context to inject into prompts

            on_complete: Callback when each sub-analysis completes

            stage_retry_rounds: Number of stage-level retry rounds for failed analyses



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



        # Stage-level retry for failed sub-analyses (persistent retry mode)
        if self.persistent_retry:
            for retry_round in range(stage_retry_rounds):
                # Check for failed sub-analyses
                failed_names = [name for name, result in sub_results.items() if not result.success]

                if not failed_names:
                    break  # All succeeded

                logger.warning(
                    f"Stage '{stage}' retry round {retry_round + 1}/{stage_retry_rounds}: "
                    f"Retrying {len(failed_names)} failed sub-analyses: {failed_names}"
                )

                # Wait before stage-level retry (longer wait to let API recover)
                stage_wait = self._calculate_backoff(retry_round + 2, is_server_error=True)
                logger.info(f"Waiting {stage_wait:.1f}s before stage retry round...")
                time.sleep(stage_wait)

                # Retry failed analyses in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as retry_executor:
                    retry_futures = {}

                    for name in failed_names:
                        prompt = prepared_prompts[name]
                        future = retry_executor.submit(
                            self._run_sub_analysis,
                            name=name,
                            stage=stage,
                            prompt=prompt,
                            model=model,
                            video=video,
                            audio=audio
                        )
                        retry_futures[future] = name

                    for future in as_completed(retry_futures):
                        name = retry_futures[future]
                        try:
                            result = future.result()
                            sub_results[name] = result

                            if on_complete and result.success:
                                on_complete(name, result.result)
                                logger.info(f"Stage retry: '{name}' succeeded on round {retry_round + 1}")

                        except Exception as e:
                            logger.error(f"Stage retry future failed for {name}: {e}")
                            sub_results[name] = SubAnalysisResult(
                                name=name,
                                stage=stage,
                                result=f"ERROR (stage retry {retry_round + 1}): {str(e)}",
                                execution_time=0,
                                success=False,
                                error=str(e)
                            )

        # Final status check
        failed_count = sum(1 for r in sub_results.values() if not r.success)
        if failed_count > 0:
            logger.error(
                f"Stage '{stage}' completed with {failed_count} FAILED sub-analyses "
                f"after all retry attempts"
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



    def run_stage_zero(

        self,

        video: str,

        audio: Optional[str],

        model: str,

        on_complete: Callable[[str, str], None] = None,

        interview_instructions: str = ""

    ) -> StageResult:

        """

        Run Stage 0: Subject ID, Baseline, and Deepfake Detection.

        This MUST run before other analyses to establish context.



        Args:

            video: Base64-encoded video

            audio: Optional base64-encoded audio (for deepfake audio analysis)

            model: Vision model ID

            on_complete: Callback for each completed sub-analysis



        Returns:

            StageResult with Stage 0 analyses

        """

        logger.info(f"Starting Stage 0 (Subject ID, Baseline, Deepfake) with {len(STAGE_ZERO_PROMPTS)} sub-analyses")
        logger.info(f"[STAGE0 DEBUG] interview_instructions truthy={bool(interview_instructions)}, len={len(interview_instructions)}")



        # Inject interview instructions into prompts if provided
        prompts = STAGE_ZERO_PROMPTS
        if interview_instructions:
            prompts = {key: interview_instructions + "\n\n" + prompt for key, prompt in STAGE_ZERO_PROMPTS.items()}
            logger.info(f"[STAGE0 DEBUG] Injected interview instructions into {len(prompts)} prompts")
        else:
            logger.info(f"[STAGE0 DEBUG] Using original prompts (no interview instructions)")

        for name, prompt in prompts.items():
            logger.info(f"[STAGE0 DEBUG] Prompt '{name}' length: {len(prompt)} chars")

        logger.info(f"[STAGE0 DEBUG] video param truthy={bool(video)}, audio param truthy={bool(audio)}")

        return self._run_parallel_sub_analyses(

            prompts=prompts,

            stage='stage_zero',

            model=model,

            video=video,

            audio=audio,

            on_complete=on_complete

        )



    def run_visual_analysis(

        self,

        video: str,

        model: str,

        blink_validation: Optional[Dict] = None,

        baseline_context: Optional[str] = None,

        on_complete: Callable[[str, str], None] = None,

        interview_instructions: str = ""

    ) -> StageResult:

        """

        Run visual sub-analyses in parallel using native video.



        Args:

            video: Base64-encoded video (Gemini native handling)

            model: Vision model ID

            blink_validation: CV blink detection results to inject into blink_rate prompt

            baseline_context: Baseline establishment results from Stage 0

            on_complete: Callback for each completed sub-analysis



        Returns:

            StageResult with visual analysis

        """

        logger.info(f"Starting visual analysis with {len(VISUAL_PROMPTS)} sub-analyses (native video)")


        # Prepare prompts, injecting baseline context and CV blink data
        visual_prompts = dict(VISUAL_PROMPTS)  # Copy to avoid modifying original

        # Inject baseline context into kinesic event log prompt
        if 'kinesic_log' in visual_prompts and '{baseline_context}' in visual_prompts['kinesic_log']:
            if baseline_context:
                visual_prompts['kinesic_log'] = visual_prompts['kinesic_log'].format(
                    baseline_context=baseline_context
                )
                logger.info("Baseline context injected into kinesic event log prompt")
            else:
                visual_prompts['kinesic_log'] = visual_prompts['kinesic_log'].format(
                    baseline_context="BASELINE NOT AVAILABLE - Establish baseline from opening 30 seconds."
                )

        # Inject CV blink data into blink_rate prompt
        if blink_validation and blink_validation.get('available', False):
            cv_blink_text = blink_validation.get('formatted_text', 'CV blink data unavailable')
            if 'blink_rate' in visual_prompts and '{cv_blink_data}' in visual_prompts['blink_rate']:
                visual_prompts['blink_rate'] = visual_prompts['blink_rate'].format(cv_blink_data=cv_blink_text)
                logger.info("CV blink data injected into blink_rate prompt for LLM interpretation")
        else:
            if 'blink_rate' in visual_prompts and '{cv_blink_data}' in visual_prompts['blink_rate']:
                fallback_msg = """CV BLINK DETECTION NOT AVAILABLE
Please estimate blink rates from the video, but note that LLM estimates
are less accurate than CV measurements. Be conservative with your estimates."""
                visual_prompts['blink_rate'] = visual_prompts['blink_rate'].format(cv_blink_data=fallback_msg)

        # Inject interview instructions into all prompts if provided
        if interview_instructions:
            visual_prompts = {key: interview_instructions + "\n\n" + prompt for key, prompt in visual_prompts.items()}
            logger.info("Interview mode instructions injected into visual prompts")

        return self._run_parallel_sub_analyses(

            prompts=visual_prompts,

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

        on_complete: Callable[[str, str], None] = None,

        interview_instructions: str = ""

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

        # Inject interview instructions into prompts if provided
        prompts = MULTIMODAL_PROMPTS
        if interview_instructions:
            prompts = {key: interview_instructions + "\n\n" + prompt for key, prompt in MULTIMODAL_PROMPTS.items()}
            logger.info("Interview mode instructions injected into multimodal prompts")



        return self._run_parallel_sub_analyses(

            prompts=prompts,

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

        transcript: Optional[str] = None,

        visual_context: Optional[str] = None,

        on_complete: Callable[[str, str], None] = None,

        interview_instructions: str = ""

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

        # Inject transcript into LIWC prompt if available
        audio_prompts = dict(AUDIO_PROMPTS)
        if transcript and 'liwc' in audio_prompts:
            audio_prompts['liwc'] = audio_prompts['liwc'].replace(
                '{transcript}',
                transcript
            )
            logger.info(f"Injected transcript ({len(transcript)} chars) into LIWC prompt")
        elif 'liwc' in audio_prompts:
            audio_prompts['liwc'] = audio_prompts['liwc'].replace(
                '{transcript}',
                '[Transcript unavailable - analyze speech patterns from audio]'
            )

        # Inject visual context into voice and deception prompts for cross-modal correlation
        if visual_context:
            context_injection = f"""

=== CROSS-MODAL CONTEXT (from visual analysis) ===
{visual_context}

IMPORTANT: Correlate vocal patterns with the visual indicators above.
Flag any timestamps where vocal stress and visual stress DIVERGE or CONVERGE.
=================================================
"""
            # Inject into voice characteristics and credibility prompts
            for key in ['voice_characteristics', 'credibility', 'sociolinguistic']:
                if key in audio_prompts:
                    audio_prompts[key] = audio_prompts[key] + context_injection
            logger.info(f"Injected visual context into audio prompts for cross-modal analysis")

        # Inject interview instructions into all prompts if provided
        if interview_instructions:
            audio_prompts = {key: interview_instructions + "\n\n" + prompt for key, prompt in audio_prompts.items()}
            logger.info("Interview mode instructions injected into audio prompts")

        return self._run_parallel_sub_analyses(

            prompts=audio_prompts,

            stage='audio',

            model=model,

            audio=audio,

            on_complete=on_complete

        )



    def run_synthesis(

        self,

        previous_analyses: str,

        model: str,

        on_complete: Callable[[str, str], None] = None,

        interview_instructions: str = ""

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

        # Inject interview instructions into prompts if provided
        synthesis_prompts = SYNTHESIS_PROMPTS
        if interview_instructions:
            synthesis_prompts = {key: interview_instructions + "\n\n" + prompt for key, prompt in SYNTHESIS_PROMPTS.items()}
            logger.info("Interview mode instructions injected into synthesis prompts")



        # First, run parallel synthesis sub-analyses (personality, threat, etc.)

        # Exclude 'final' which needs all synthesis results

        parallel_prompts = {k: v for k, v in synthesis_prompts.items() if k != 'final'}



        parallel_result = self._run_parallel_sub_analyses(

            prompts=parallel_prompts,

            stage='synthesis_parallel',

            model=model,

            context=previous_analyses,

            on_complete=on_complete

        )



        # Then run final integration with all data

        synthesis_text = parallel_result.combined_text



        final_prompt = synthesis_prompts['final'].format(

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


    def _generate_interview_instructions(self, interview_context: Dict) -> str:
        """
        Generate interview mode instructions to inject into prompts.

        These instructions ensure the LLM focuses analysis on the suspect only
        and uses interviewer questions as context without analyzing the interviewer.

        Args:
            interview_context: Dict with 'enabled', 'suspect_position', 'suspect_speaker', 'transcript'

        Returns:
            Formatted instruction string to prepend to prompts
        """
        suspect_position = interview_context.get('suspect_position', 'auto')
        suspect_speaker = interview_context.get('suspect_speaker', 'auto')

        # Determine position description
        if suspect_position == 'left':
            position_desc = "the person on the LEFT side of the frame"
        elif suspect_position == 'right':
            position_desc = "the person on the RIGHT side of the frame"
        elif suspect_position == 'fullscreen':
            position_desc = "the single person visible when the frame shows only one person"
        else:
            position_desc = "the person being interviewed/questioned (the interviewee, NOT the interviewer)"

        # Determine speaker description
        if suspect_speaker and suspect_speaker != 'auto':
            speaker_desc = f'In the transcript, the suspect is labeled as "{suspect_speaker}".'
        else:
            speaker_desc = "In the transcript, identify the suspect as the person ANSWERING questions, not asking them."

        instructions = f"""
═══════════════════════════════════════════════════════════════════════════════
INTERVIEW MODE ACTIVE - CRITICAL ANALYSIS CONSTRAINTS
═══════════════════════════════════════════════════════════════════════════════

This video shows an INTERVIEW or INTERROGATION between two people.

**THE SUSPECT/SUBJECT (ANALYZE THIS PERSON ONLY):**
- {position_desc}
- This is the person being questioned/interviewed
- ALL behavioral observations must be about THIS person
- ALL voice/speech analysis targets THIS person's statements
- ALL facial expressions, body language, micro-expressions = THIS person

**THE INTERVIEWER (DO NOT ANALYZE):**
- The other person in the frame
- Use their QUESTIONS only as CONTEXT for understanding responses
- Do NOT analyze their behavior, expressions, voice, or body language
- Do NOT include their speech patterns in linguistic analysis
- Do NOT assess their credibility or deception indicators

**TRANSCRIPT FILTERING:**
{speaker_desc}
- When analyzing speech content, focus ONLY on suspect's responses
- Interviewer questions provide CONTEXT but are NOT analyzed

**CRITICAL REMINDERS:**
- If you observe a behavior, VERIFY it belongs to the SUSPECT before including it
- In split-screen frames: Focus on the designated position ({suspect_position})
- In full-screen frames of one person: That person is likely the suspect speaking at length
- Do NOT confuse interviewer reactions with suspect behaviors

═══════════════════════════════════════════════════════════════════════════════

"""
        return instructions


    def run_full_pipeline(

        self,

        video: str,

        audio: Optional[str],

        visual_model: str,

        multimodal_model: str,

        audio_model: str,

        synthesis_model: str,

        transcript: Optional[str] = None,

        blink_validation: Optional[Dict] = None,

        progress_callback: Callable[[str, int], None] = None,

        results_callback: Callable[[str, str], None] = None,

        interview_context: Optional[Dict] = None

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

            blink_validation: Optional CV-based blink detection results (ground truth)

            progress_callback: Progress update callback

            results_callback: Results streaming callback

            interview_context: Optional interview mode context with keys:
                - enabled: bool
                - suspect_position: str ("auto", "left", "right", "fullscreen")
                - suspect_speaker: str ("auto", "Speaker 1", etc.)
                - transcript: str (for Q&A parsing)



        Returns:

            Dict of stage name to StageResult

        """

        all_results = {}

        # Generate interview mode instructions if enabled
        interview_instructions = ""
        logger.info(f"[PIPELINE DEBUG] interview_context={interview_context}")
        logger.info(f"[PIPELINE DEBUG] interview_context type={type(interview_context)}")
        if interview_context and interview_context.get('enabled'):
            interview_instructions = self._generate_interview_instructions(interview_context)
            logger.info(f"[PIPELINE DEBUG] Generated interview_instructions ({len(interview_instructions)} chars)")
        else:
            logger.info(f"[PIPELINE DEBUG] interview_instructions is empty string (interview mode disabled)")
        logger.info(f"[PIPELINE DEBUG] video size: {len(video) if video else 0} chars")
        logger.info(f"[PIPELINE DEBUG] audio size: {len(audio) if audio else 0} chars")



        def update_progress(msg, step):

            if progress_callback:

                progress_callback(msg, step)



        # Stage 0: Subject ID, Baseline & Deepfake Detection (MUST RUN FIRST)

        update_progress("🎯 Running Stage 0 (Subject ID, Baseline, Deepfake Detection)...", 2)

        stage_zero_result = self.run_stage_zero(

            video=video,

            audio=audio,

            model=visual_model,

            on_complete=results_callback,

            interview_instructions=interview_instructions

        )

        all_results['stage_zero'] = stage_zero_result



        # Extract baseline context for downstream analyses

        baseline_context = None

        if stage_zero_result.success and 'baseline_establishment' in stage_zero_result.sub_results:

            baseline_result = stage_zero_result.sub_results['baseline_establishment']

            if baseline_result.success:

                baseline_context = baseline_result.result

                logger.info(f"Baseline established ({len(baseline_context)} chars) - will inform subsequent analyses")



        # Check deepfake detection result

        if stage_zero_result.success and 'deepfake_detection' in stage_zero_result.sub_results:

            deepfake_result = stage_zero_result.sub_results['deepfake_detection']

            if deepfake_result.success and 'LIKELY SYNTHETIC' in deepfake_result.result:

                logger.warning("DEEPFAKE DETECTED - flagging for review")

                # Could abort here, but we continue with warning



        update_progress(f"✓ Stage 0 complete ({len([r for r in stage_zero_result.sub_results.values() if r.success])}/3 sub-analyses)", 2)



        # Stage 1: Visual Analysis (parallel sub-analyses with native video)

        update_progress("🔍 Running visual sub-analyses (unified behavioral coding, archetype, congruence)...", 3)

        visual_result = self.run_visual_analysis(

            video=video,

            model=visual_model,

            blink_validation=blink_validation,

            baseline_context=baseline_context,

            on_complete=results_callback,

            interview_instructions=interview_instructions

        )

        all_results['visual'] = visual_result

        # Extract visual context for cross-pollination to audio stage
        visual_context = None
        if visual_result.success:
            congruence_result = visual_result.sub_results.get('congruence')
            stress_result = visual_result.sub_results.get('stress_clusters')
            visual_parts = []
            if congruence_result and congruence_result.success:
                visual_parts.append("VISUAL CONGRUENCE INDICATORS:\n" + congruence_result.result[:2000])
            if stress_result and stress_result.success:
                visual_parts.append("STRESS CLUSTERS:\n" + stress_result.result[:1000])
            if visual_parts:
                visual_context = "\n\n".join(visual_parts)
                logger.info(f"Extracted visual context ({len(visual_context)} chars) for audio cross-pollination")

        update_progress(f"✓ Visual analysis complete ({len([r for r in visual_result.sub_results.values() if r.success])}/4 sub-analyses)", 3)



        # Stage 2: Multimodal Analysis (parallel sub-analyses with native video)

        if audio:

            update_progress("📊 Running multimodal sub-analyses (timeline, sync, environment, awareness)...", 4)

            multimodal_result = self.run_multimodal_analysis(

                video=video,

                audio=audio,

                model=multimodal_model,

                on_complete=results_callback,

                interview_instructions=interview_instructions

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

                transcript=transcript,

                visual_context=visual_context,

                on_complete=results_callback,

                interview_instructions=interview_instructions

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



        # Combine all previous analyses for synthesis (including Stage 0)

        stage_zero_text = stage_zero_result.combined_text if stage_zero_result.success else "Stage 0 analysis unavailable"

        previous_analyses = f"""

=== STAGE 0: SUBJECT ID, BASELINE, AUTHENTICATION ===

{stage_zero_text}



=== VISUAL ANALYSIS ===

{visual_result.combined_text}



=== MULTIMODAL ANALYSIS ===

{all_results['multimodal'].combined_text}



=== AUDIO ANALYSIS ===

{all_results['audio'].combined_text}

"""

        # Apply Signal Collapsing Layer to deduplicate timestamped events
        try:
            stage_texts = {
                'visual': visual_result.combined_text,
                'multimodal': all_results['multimodal'].combined_text,
                'audio': all_results['audio'].combined_text
            }
            collapsed_summary, collapsed_events = collapse_analysis_outputs(stage_texts)
            if collapsed_events:
                previous_analyses = collapsed_summary + "\n\n" + previous_analyses
                logger.info(f"Signal Collapsing: {len(collapsed_events)} events collapsed")
        except Exception as sc_err:
            logger.warning(f"Signal collapsing failed: {sc_err}")

        # Inject CV blink validation data (GROUND TRUTH) into synthesis context
        # This prevents hallucinated LLM blink rates from propagating
        if blink_validation and blink_validation.get('available', False):
            cv_blink_text = f"""
═══════════════════════════════════════════════════════════════
CV-VALIDATED BLINK DATA (GROUND TRUTH - USE THESE VALUES)
═══════════════════════════════════════════════════════════════
The following blink rates were measured by computer vision using
MediaPipe Face Mesh EAR (Eye Aspect Ratio) algorithm. These are
ACCURATE measurements of actual eye closures and should OVERRIDE
any LLM-estimated blink rates which may be hallucinated.

{blink_validation.get('formatted_text', 'CV data unavailable')}

CRITICAL: If LLM blink analysis claims rates significantly higher
than these CV-measured values, the LLM has hallucinated. Use CV data.
═══════════════════════════════════════════════════════════════
"""
            previous_analyses = cv_blink_text + "\n\n" + previous_analyses
            logger.info(f"CV blink data injected into synthesis (BPM={blink_validation.get('metrics', {}).get('bpm', 0):.1f})")

        # Inject interview mode instructions into synthesis context if enabled
        if interview_instructions:
            previous_analyses = interview_instructions + "\n\n" + previous_analyses
            logger.info("Interview mode instructions injected into synthesis context")



        # Stage 4: Synthesis (parallel sub-analyses + final integration)

        update_progress("🎯 Running synthesis (personality, threat, differential, contradictions, red team)...", 6)

        synthesis_result = self.run_synthesis(

            previous_analyses=previous_analyses,

            model=synthesis_model,

            on_complete=results_callback,

            interview_instructions=interview_instructions

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



    # Stage 0 results (Subject ID, Baseline, Deepfake)

    if 'stage_zero' in results:

        stage_zero = results['stage_zero']

        # Extract subject identification for case file

        if 'subject_identification' in stage_zero.sub_results and stage_zero.sub_results['subject_identification'].success:

            formatted['subject_identification'] = stage_zero.sub_results['subject_identification'].result

        # Extract baseline for reference

        if 'baseline_establishment' in stage_zero.sub_results and stage_zero.sub_results['baseline_establishment'].success:

            formatted['baseline'] = stage_zero.sub_results['baseline_establishment'].result

        # Extract deepfake detection for case file

        if 'deepfake_detection' in stage_zero.sub_results and stage_zero.sub_results['deepfake_detection'].success:

            formatted['deepfake_detection'] = stage_zero.sub_results['deepfake_detection'].result



    # Visual/Essence section

    if 'visual' in results:

        visual = results['visual']

        formatted['essence'] = visual.combined_text



        # Extract kinesic event log (single source of truth for behavioral observations)

        if 'kinesic_log' in visual.sub_results and visual.sub_results['kinesic_log'].success:

            formatted['kinesic_log'] = visual.sub_results['kinesic_log'].result



        # Extract blink rate analysis

        if 'blink_rate' in visual.sub_results and visual.sub_results['blink_rate'].success:

            formatted['blink_rate'] = visual.sub_results['blink_rate'].result



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

                audio_parts.append(f"=== {name.upper().replace('_', ' ')} ===\n{result.result}")
        formatted['audio'] = "\n\n".join(audio_parts) if audio_parts else audio.combined_text


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

