"""
FastAPI REST API server for the behavioral profiling system.
Provides programmatic access to the profiling pipeline via HTTP endpoints.
"""

import os
import json
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from profiler import BehavioralProfiler, ModelSelection
from config.config_manager import ConfigManager
from infra.database import get_database
from infra.cache_manager import get_cache
from output.pdf_generator import generate_pdf_report, REPORTLAB_AVAILABLE
from config.models_config import (
    VISION_MODELS,
    AUDIO_MODELS,
    SYNTHESIS_MODELS,
    get_default_model_for_stage
)

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class ModelConfig(BaseModel):
    """Model selection for analysis stages."""
    essence_model: str = Field(default="google/gemini-3-pro-preview", description="Model for Sam Christensen analysis")
    multimodal_model: str = Field(default="google/gemini-2.5-flash", description="Model for multimodal analysis")
    audio_model: str = Field(default="google/gemini-2.5-flash", description="Model for audio analysis")
    liwc_model: str = Field(default="google/gemini-2.5-flash", description="Model for LIWC analysis")
    synthesis_model: str = Field(default="google/gemini-3-pro-preview", description="Model for FBI synthesis")


class AnalysisRequest(BaseModel):
    """Request body for video analysis via URL."""
    video_url: Optional[str] = Field(None, description="URL to video (YouTube, etc.)")
    subject_name: Optional[str] = Field(None, description="Name of subject for profile storage")
    subject_notes: Optional[str] = Field(None, description="Notes about the subject")
    use_cache: bool = Field(True, description="Use cached results if available")
    models: Optional[ModelConfig] = Field(None, description="Custom model selection")


class AnalysisResponse(BaseModel):
    """Response from analysis endpoint."""
    case_id: str
    status: str
    timestamp: str
    processing_time_seconds: float
    video_metadata: Dict[str, Any]
    models_used: Dict[str, str]
    analyses: Dict[str, str]
    subject_name: Optional[str] = None
    profile_saved: bool = False


class JobStatus(BaseModel):
    """Status of an async analysis job."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress_step: int
    progress_message: str
    started_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SubjectInfo(BaseModel):
    """Subject information."""
    id: int
    name: str
    notes: str
    created_at: str
    updated_at: str
    profile_count: int


class ProfileSummary(BaseModel):
    """Summary of a profile."""
    id: int
    subject_id: Optional[int]
    subject_name: str
    case_id: str
    report_number: int
    timestamp: str
    processing_time: float
    status: str


class CacheStats(BaseModel):
    """Cache statistics."""
    total_entries: int
    total_hits: int
    total_size_kb: float
    total_size_mb: float
    cache_dir: str
    max_age_days: int


# ============================================================================
# In-memory job tracking
# ============================================================================

_jobs: Dict[str, JobStatus] = {}


def create_job(job_id: str) -> JobStatus:
    """Create a new job."""
    job = JobStatus(
        job_id=job_id,
        status="pending",
        progress_step=0,
        progress_message="Job created",
        started_at=datetime.now().isoformat()
    )
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[JobStatus]:
    """Get job status."""
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id in _jobs:
        job = _jobs[job_id]
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)


# ============================================================================
# Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting FBI Behavioral Profiler API Server")
    yield
    logger.info("Shutting down FBI Behavioral Profiler API Server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="FBI Behavioral Profiler API",
        description="""
REST API for the FBI-Style Behavioral Profiling System.

Provides programmatic access to:
- Video analysis and profiling
- Subject and profile management
- Prompt template management
- Cache management

**Authentication:** Uses the API key configured in the system settings.
        """,
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # Health & Status Endpoints
    # ========================================================================

    @app.get("/health", tags=["Status"])
    async def health_check():
        """Check if the API server is running."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/status", tags=["Status"])
    async def get_status():
        """Get system status and configuration."""
        config = ConfigManager()
        cache = get_cache()
        db = get_database()

        return {
            "api_key_configured": config.has_api_key(),
            "cache_stats": cache.get_stats(),
            "database_stats": db.get_stats(),
            "pdf_export_available": REPORTLAB_AVAILABLE
        }

    # ========================================================================
    # Analysis Endpoints
    # ========================================================================

    @app.post("/analyze/upload", response_model=AnalysisResponse, tags=["Analysis"])
    async def analyze_video_upload(
        video: UploadFile = File(..., description="Video file to analyze"),
        subject_name: Optional[str] = Form(None),
        subject_notes: Optional[str] = Form(None),
        use_cache: bool = Form(True),
        essence_model: str = Form(default="google/gemini-3-pro-preview"),
        multimodal_model: str = Form(default="google/gemini-2.5-flash"),
        audio_model: str = Form(default="google/gemini-2.5-flash"),
        liwc_model: str = Form(default="google/gemini-2.5-flash"),
        synthesis_model: str = Form(default="google/gemini-3-pro-preview")
    ):
        """
        Analyze an uploaded video file.

        Upload a video file and receive a complete behavioral profile.
        This is a synchronous endpoint - it will wait until analysis completes.
        """
        config = ConfigManager()
        if not config.has_api_key():
            raise HTTPException(status_code=400, detail="API key not configured")

        # Validate file type
        allowed_types = [".mp4", ".mov", ".avi", ".webm"]
        file_ext = Path(video.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )

        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, video.filename)

        try:
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(video.file, f)

            # Create model config
            model_config = ModelSelection(
                essence_model=essence_model,
                multimodal_model=multimodal_model,
                audio_model=audio_model,
                liwc_model=liwc_model,
                synthesis_model=synthesis_model
            )

            # Run analysis
            profiler = BehavioralProfiler(model_config=model_config)
            result = profiler.profile_video(temp_path, use_cache=use_cache)

            # Save to database if subject name provided
            profile_saved = False
            if subject_name and subject_name.strip():
                try:
                    db = get_database()
                    db.save_profile(
                        result=result,
                        subject_name=subject_name.strip(),
                        video_source=video.filename,
                        notes=subject_notes or ""
                    )
                    profile_saved = True
                except Exception as e:
                    logger.error(f"Failed to save profile: {e}")

            return AnalysisResponse(
                case_id=result.get("case_id", ""),
                status=result.get("status", "completed"),
                timestamp=result.get("timestamp", ""),
                processing_time_seconds=result.get("processing_time_seconds", 0),
                video_metadata=result.get("video_metadata", {}),
                models_used=result.get("models_used", {}),
                analyses=result.get("analyses", {}),
                subject_name=subject_name,
                profile_saved=profile_saved
            )

        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)

    @app.post("/analyze/url", response_model=AnalysisResponse, tags=["Analysis"])
    async def analyze_video_url(request: AnalysisRequest):
        """
        Analyze a video from URL.

        Provide a URL to a video (YouTube, Vimeo, etc.) and receive a complete behavioral profile.
        """
        if not request.video_url:
            raise HTTPException(status_code=400, detail="video_url is required")

        config = ConfigManager()
        if not config.has_api_key():
            raise HTTPException(status_code=400, detail="API key not configured")

        try:
            from media.video_downloader import download_video, is_valid_url

            if not is_valid_url(request.video_url):
                raise HTTPException(status_code=400, detail="Invalid URL format")

            # Download video
            file_path, metadata = download_video(
                request.video_url,
                max_duration=300,
                max_filesize_mb=100
            )

            try:
                # Create model config
                models = request.models or ModelConfig()
                model_config = ModelSelection(
                    essence_model=models.essence_model,
                    multimodal_model=models.multimodal_model,
                    audio_model=models.audio_model,
                    liwc_model=models.liwc_model,
                    synthesis_model=models.synthesis_model
                )

                # Run analysis
                profiler = BehavioralProfiler(model_config=model_config)
                result = profiler.profile_video(file_path, use_cache=request.use_cache)

                # Save to database if subject name provided
                profile_saved = False
                if request.subject_name and request.subject_name.strip():
                    try:
                        db = get_database()
                        db.save_profile(
                            result=result,
                            subject_name=request.subject_name.strip(),
                            video_source=request.video_url,
                            notes=request.subject_notes or ""
                        )
                        profile_saved = True
                    except Exception as e:
                        logger.error(f"Failed to save profile: {e}")

                return AnalysisResponse(
                    case_id=result.get("case_id", ""),
                    status=result.get("status", "completed"),
                    timestamp=result.get("timestamp", ""),
                    processing_time_seconds=result.get("processing_time_seconds", 0),
                    video_metadata=result.get("video_metadata", {}),
                    models_used=result.get("models_used", {}),
                    analyses=result.get("analyses", {}),
                    subject_name=request.subject_name,
                    profile_saved=profile_saved
                )

            finally:
                # Cleanup downloaded file
                if os.path.exists(file_path):
                    os.remove(file_path)

        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="URL download not available. Install yt-dlp: pip install yt-dlp"
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # ========================================================================
    # Subject & Profile Endpoints
    # ========================================================================

    @app.get("/subjects", response_model=List[SubjectInfo], tags=["Subjects"])
    async def list_subjects(search: Optional[str] = Query(None)):
        """List all subjects, optionally filtered by search term."""
        db = get_database()
        subjects = db.list_subjects(search=search)

        return [
            SubjectInfo(
                id=s.id,
                name=s.name,
                notes=s.notes,
                created_at=s.created_at,
                updated_at=s.updated_at,
                profile_count=s.profile_count
            )
            for s in subjects
        ]

    @app.get("/subjects/{subject_id}", response_model=SubjectInfo, tags=["Subjects"])
    async def get_subject(subject_id: int):
        """Get a specific subject by ID."""
        db = get_database()
        subject = db.get_subject(subject_id)

        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

        return SubjectInfo(
            id=subject.id,
            name=subject.name,
            notes=subject.notes,
            created_at=subject.created_at,
            updated_at=subject.updated_at,
            profile_count=subject.profile_count
        )

    @app.get("/subjects/{subject_id}/profiles", response_model=List[ProfileSummary], tags=["Subjects"])
    async def get_subject_profiles(subject_id: int):
        """Get all profiles for a specific subject."""
        db = get_database()
        profiles = db.get_profiles_for_subject(subject_id)

        return [
            ProfileSummary(
                id=p["id"],
                subject_id=p["subject_id"],
                subject_name=p["subject_name"],
                case_id=p["case_id"],
                report_number=p["report_number"],
                timestamp=p["timestamp"],
                processing_time=p["processing_time"],
                status=p["status"]
            )
            for p in profiles
        ]

    @app.get("/profiles", response_model=List[ProfileSummary], tags=["Profiles"])
    async def list_profiles(
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
        search: Optional[str] = Query(None)
    ):
        """List profiles with pagination."""
        db = get_database()
        profiles = db.list_profiles(limit=limit, offset=offset, search=search)

        return [
            ProfileSummary(
                id=p["id"],
                subject_id=p["subject_id"],
                subject_name=p["subject_name"],
                case_id=p["case_id"],
                report_number=p["report_number"],
                timestamp=p["timestamp"],
                processing_time=p["processing_time"],
                status=p["status"]
            )
            for p in profiles
        ]

    @app.get("/profiles/{profile_id}", tags=["Profiles"])
    async def get_profile(profile_id: int):
        """Get full profile details including all analyses."""
        db = get_database()
        profile = db.get_profile(profile_id=profile_id)

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        return profile

    @app.get("/profiles/{profile_id}/pdf", tags=["Profiles"])
    async def get_profile_pdf(profile_id: int, subject_name: Optional[str] = None):
        """Generate and download PDF report for a profile."""
        if not REPORTLAB_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail="PDF generation not available. Install reportlab: pip install reportlab"
            )

        db = get_database()
        profile = db.get_profile(profile_id=profile_id)

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        try:
            # Reconstruct result format expected by PDF generator
            result = profile.get("full_result", {})
            if not result:
                result = {
                    "case_id": profile["case_id"],
                    "timestamp": profile["timestamp"],
                    "analyses": profile["analyses"],
                    "video_metadata": profile.get("video_metadata", {}),
                    "models_used": profile.get("models_used", {})
                }

            pdf_path = generate_pdf_report(
                result=result,
                subject_name=subject_name or profile.get("subject_name")
            )

            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=f"profile_{profile['case_id']}.pdf"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    @app.delete("/profiles/{profile_id}", tags=["Profiles"])
    async def delete_profile(profile_id: int):
        """Delete a profile."""
        db = get_database()
        success = db.delete_profile(profile_id)

        if not success:
            raise HTTPException(status_code=404, detail="Profile not found")

        return {"status": "deleted", "profile_id": profile_id}

    # ========================================================================
    # Cache Endpoints
    # ========================================================================

    @app.get("/cache/stats", response_model=CacheStats, tags=["Cache"])
    async def get_cache_stats():
        """Get cache statistics."""
        cache = get_cache()
        stats = cache.get_stats()

        return CacheStats(**stats)

    @app.delete("/cache", tags=["Cache"])
    async def clear_cache():
        """Clear all cache entries."""
        cache = get_cache()
        count = cache.invalidate(all_entries=True)

        return {"status": "cleared", "entries_removed": count}

    @app.get("/cache/entries", tags=["Cache"])
    async def list_cache_entries(limit: int = Query(20, ge=1, le=100)):
        """List recent cache entries."""
        cache = get_cache()
        entries = cache.list_entries(limit=limit)

        return entries

    # ========================================================================
    # Models Endpoint
    # ========================================================================

    @app.get("/models", tags=["Models"])
    async def get_available_models():
        """Get available models for each stage."""
        return {
            "vision_models": [
                {"id": m.id, "name": m.name, "provider": m.provider, "cost_tier": m.cost_tier}
                for m in VISION_MODELS
            ],
            "audio_models": [
                {"id": m.id, "name": m.name, "provider": m.provider, "cost_tier": m.cost_tier}
                for m in AUDIO_MODELS
            ],
            "synthesis_models": [
                {"id": m.id, "name": m.name, "provider": m.provider, "cost_tier": m.cost_tier}
                for m in SYNTHESIS_MODELS
            ],
            "defaults": {
                "essence": get_default_model_for_stage("essence"),
                "multimodal": get_default_model_for_stage("multimodal"),
                "audio": get_default_model_for_stage("audio"),
                "liwc": get_default_model_for_stage("liwc"),
                "synthesis": get_default_model_for_stage("synthesis")
            }
        }

    return app


# ============================================================================
# Server Runner
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")
        return

    app = create_app()

    print("\n" + "=" * 70)
    print("FBI BEHAVIORAL PROFILER - REST API SERVER")
    print("=" * 70)
    print(f"\n Starting server at http://{host}:{port}")
    print(f" API Documentation: http://{host}:{port}/docs")
    print(f" Alternative docs:  http://{host}:{port}/redoc")
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FBI Behavioral Profiler REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
