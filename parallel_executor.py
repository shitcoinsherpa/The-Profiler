"""
Parallel execution utilities for running analysis stages concurrently.
Uses ThreadPoolExecutor for non-blocking parallel API calls.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisTask:
    """Represents a single analysis task to run."""
    name: str
    func: Callable[..., str]
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class AnalysisResult:
    """Result of a parallel analysis task."""
    name: str
    result: str
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


def run_analyses_parallel(
    tasks: List[AnalysisTask],
    max_workers: int = 4,
    on_complete: Optional[Callable[[str, str], None]] = None
) -> Dict[str, AnalysisResult]:
    """
    Run multiple analysis tasks in parallel.

    Args:
        tasks: List of AnalysisTask objects to execute
        max_workers: Maximum number of parallel threads
        on_complete: Optional callback called when each task completes (name, result)

    Returns:
        Dictionary mapping task names to their AnalysisResult objects
    """
    import time

    results: Dict[str, AnalysisResult] = {}

    if not tasks:
        return results

    logger.info(f"Starting parallel execution of {len(tasks)} tasks with {max_workers} workers")

    def execute_task(task: AnalysisTask) -> Tuple[str, str, bool, Optional[str], float]:
        """Execute a single task and return results."""
        start_time = time.time()
        try:
            result = task.func(*task.args, **task.kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Task '{task.name}' completed in {execution_time:.2f}s")
            return (task.name, result, True, None, execution_time)
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task '{task.name}' failed: {str(e)}"
            logger.error(error_msg)
            return (task.name, f"Analysis failed: {str(e)}", False, str(e), execution_time)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(execute_task, task): task for task in tasks}

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                name, result, success, error, exec_time = future.result()
                results[name] = AnalysisResult(
                    name=name,
                    result=result,
                    success=success,
                    error=error,
                    execution_time=exec_time
                )

                # Call completion callback if provided
                if on_complete and success:
                    try:
                        on_complete(name, result)
                    except Exception as cb_err:
                        logger.warning(f"Completion callback failed for {name}: {cb_err}")

            except Exception as exc:
                logger.error(f"Task {task.name} generated an exception: {exc}")
                results[task.name] = AnalysisResult(
                    name=task.name,
                    result=f"Analysis failed: {str(exc)}",
                    success=False,
                    error=str(exc)
                )

    logger.info(f"Parallel execution complete. {sum(1 for r in results.values() if r.success)}/{len(results)} tasks succeeded")
    return results


def run_stage_parallel(
    essence_func: Callable[[], str],
    multimodal_func: Callable[[], str],
    audio_func: Callable[[], str],
    liwc_func: Callable[[], str],
    on_complete: Optional[Callable[[str, str], None]] = None
) -> Dict[str, AnalysisResult]:
    """
    Run the main analysis stages (3-5) in parallel.

    This runs essence, multimodal, audio, and LIWC analyses concurrently.

    Args:
        essence_func: Function to run essence analysis
        multimodal_func: Function to run multimodal analysis
        audio_func: Function to run audio analysis
        liwc_func: Function to run LIWC analysis
        on_complete: Optional callback when each stage completes

    Returns:
        Dictionary with results for each stage
    """
    tasks = [
        AnalysisTask(name="essence", func=essence_func),
        AnalysisTask(name="multimodal", func=multimodal_func),
        AnalysisTask(name="audio", func=audio_func),
        AnalysisTask(name="liwc", func=liwc_func)
    ]

    return run_analyses_parallel(tasks, max_workers=4, on_complete=on_complete)
