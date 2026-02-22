"""
Background Jobs for Calendar Weather System.

═══════════════════════════════════════════════════════════════════════════
BACKGROUND TASKS
═══════════════════════════════════════════════════════════════════════════

This module provides background job scheduling for:

1. HISTORICAL DATA INGESTION
   - Bulk ingestion of multiple years of weather data
   - Run once per location (or when expanding date range)
   - Progress tracking via task IDs

2. DAILY FORECAST REFRESH
   - Update forecast cache daily for watched locations
   - Run at 00:00 UTC when new forecast data available
   - Clear stale forecast entries

3. DATABASE MAINTENANCE
   - Clean up old historical data if needed
   - Rebuild indexes periodically
   - Vacuum/analyze tables

═══════════════════════════════════════════════════════════════════════════
IMPLEMENTATION OPTIONS
═══════════════════════════════════════════════════════════════════════════

For production, consider these job queue options:

1. APScheduler (in-process)
   - Simple, no external deps
   - Runs in same process as FastAPI
   - Good for moderate workloads

2. Celery + Redis
   - Distributed task queue
   - Better for heavy workloads
   - Requires Redis (already have for caching)

3. FastAPI BackgroundTasks
   - Simplest option
   - Fire-and-forget
   - No job tracking

We use APScheduler for production-ready scheduling with job tracking.

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Job Status Model
# ═══════════════════════════════════════════════════════════════════════════

class JobStatus(str, Enum):
    """Status of a background job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress tracking for a background job."""
    task_id: str
    job_type: str
    status: JobStatus
    progress: float  # 0.0 to 1.0
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress": round(self.progress * 100, 1),
            "progress_pct": f"{self.progress * 100:.1f}%",
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_seconds": (
                (self.completed_at or datetime.now()) - self.started_at
            ).total_seconds(),
            "error": self.error,
            "result": self.result,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Job Manager
# ═══════════════════════════════════════════════════════════════════════════

class BackgroundJobManager:
    """
    Manages background job execution and tracking.
    
    Usage:
        manager = BackgroundJobManager()
        
        # Submit a job
        task_id = await manager.submit_ingestion_job(
            latitude=12.9,
            longitude=80.2,
            start_year=2020,
            end_year=2024,
        )
        
        # Check progress
        progress = manager.get_progress(task_id)
        print(f"Progress: {progress.progress_pct}")
        
        # List all jobs
        jobs = manager.list_jobs()
    """
    
    def __init__(self, max_workers: int = 4):
        self._jobs: Dict[str, JobProgress] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        return str(uuid.uuid4())[:8]
    
    def get_progress(self, task_id: str) -> Optional[JobProgress]:
        """Get progress for a job."""
        return self._jobs.get(task_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[JobProgress]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.started_at, reverse=True)
    
    async def submit_ingestion_job(
        self,
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int,
        db_connection_string: Optional[str] = None,
    ) -> str:
        """
        Submit a historical data ingestion job.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            start_year: First year to ingest
            end_year: Last year to ingest
            db_connection_string: Optional database URL
            
        Returns:
            Task ID for tracking progress
        """
        task_id = self._generate_task_id()
        
        progress = JobProgress(
            task_id=task_id,
            job_type="historical_ingestion",
            status=JobStatus.PENDING,
            progress=0.0,
            message=f"Queued ingestion for years {start_year}-{end_year}",
            started_at=datetime.now(),
        )
        self._jobs[task_id] = progress
        
        # Run the job in background
        async def run_job():
            try:
                await self._run_ingestion(
                    task_id, latitude, longitude, start_year, end_year, db_connection_string
                )
            except Exception as e:
                logger.exception("Ingestion job %s failed", task_id)
                progress.status = JobStatus.FAILED
                progress.error = str(e)
                progress.completed_at = datetime.now()
        
        task = asyncio.create_task(run_job())
        self._running_tasks[task_id] = task
        
        return task_id
    
    async def _run_ingestion(
        self,
        task_id: str,
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int,
        db_connection_string: Optional[str],
    ):
        """Execute historical data ingestion job."""
        from .historical_service import HistoricalWeatherService
        
        progress = self._jobs[task_id]
        progress.status = JobStatus.RUNNING
        progress.message = "Starting ingestion service..."
        
        service = HistoricalWeatherService(db_connection_string)
        
        try:
            total_years = end_year - start_year + 1
            ingested_count = 0
            failed_years = []
            
            for idx, year in enumerate(range(start_year, end_year + 1)):
                progress.progress = idx / total_years
                progress.message = f"Ingesting {year}..."
                
                try:
                    records = await service.fetch_year(latitude, longitude, year)
                    ingested_count += len(records)
                    logger.info("Ingested %d records for %d", len(records), year)
                except Exception as e:
                    logger.warning("Failed to ingest year %d: %s", year, e)
                    failed_years.append(year)
            
            progress.status = JobStatus.COMPLETED
            progress.progress = 1.0
            progress.completed_at = datetime.now()
            progress.message = f"Ingested {ingested_count} records"
            progress.result = {
                "total_records": ingested_count,
                "years_ingested": total_years - len(failed_years),
                "years_failed": failed_years,
                "latitude": latitude,
                "longitude": longitude,
            }
            
        finally:
            await service.close()
    
    async def submit_forecast_refresh_job(
        self,
        locations: List[Dict[str, float]],
    ) -> str:
        """
        Submit a forecast refresh job for multiple locations.
        
        Args:
            locations: List of {"latitude": float, "longitude": float}
            
        Returns:
            Task ID for tracking
        """
        task_id = self._generate_task_id()
        
        progress = JobProgress(
            task_id=task_id,
            job_type="forecast_refresh",
            status=JobStatus.PENDING,
            progress=0.0,
            message=f"Queued refresh for {len(locations)} locations",
            started_at=datetime.now(),
        )
        self._jobs[task_id] = progress
        
        async def run_job():
            try:
                await self._run_forecast_refresh(task_id, locations)
            except Exception as e:
                logger.exception("Forecast refresh %s failed", task_id)
                progress.status = JobStatus.FAILED
                progress.error = str(e)
                progress.completed_at = datetime.now()
        
        task = asyncio.create_task(run_job())
        self._running_tasks[task_id] = task
        
        return task_id
    
    async def _run_forecast_refresh(
        self,
        task_id: str,
        locations: List[Dict[str, float]],
    ):
        """Execute forecast refresh job."""
        from .forecast_service import ForecastService
        
        progress = self._jobs[task_id]
        progress.status = JobStatus.RUNNING
        
        service = ForecastService(cache_enabled=True)
        
        try:
            total = len(locations)
            refreshed = 0
            failed = 0
            
            for idx, loc in enumerate(locations):
                progress.progress = idx / total
                progress.message = f"Refreshing {idx+1}/{total}..."
                
                try:
                    await service.get_forecast(
                        loc["latitude"],
                        loc["longitude"],
                        force_refresh=True,
                    )
                    refreshed += 1
                    # Rate limit
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning("Failed to refresh forecast: %s", e)
                    failed += 1
            
            progress.status = JobStatus.COMPLETED
            progress.progress = 1.0
            progress.completed_at = datetime.now()
            progress.message = f"Refreshed {refreshed}/{total} forecasts"
            progress.result = {
                "refreshed": refreshed,
                "failed": failed,
                "total": total,
            }
            
        finally:
            await service.close()
    
    async def cancel_job(self, task_id: str) -> bool:
        """Cancel a running job."""
        task = self._running_tasks.get(task_id)
        progress = self._jobs.get(task_id)
        
        if not task or not progress:
            return False
        
        if progress.status != JobStatus.RUNNING:
            return False
        
        task.cancel()
        progress.status = JobStatus.CANCELLED
        progress.completed_at = datetime.now()
        progress.message = "Job cancelled by user"
        
        return True
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove old completed/failed jobs. Returns count removed."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        to_remove = []
        
        for task_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at and job.completed_at.timestamp() < cutoff:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._jobs[task_id]
            self._running_tasks.pop(task_id, None)
        
        return len(to_remove)


# ═══════════════════════════════════════════════════════════════════════════
# Scheduled Jobs
# ═══════════════════════════════════════════════════════════════════════════

class ScheduledJobRunner:
    """
    Runs scheduled background jobs.
    
    This can be integrated with APScheduler or run standalone.
    
    Usage:
        runner = ScheduledJobRunner()
        
        # Add watched locations
        runner.add_watched_location(12.9, 80.2, "Chennai")
        
        # Start scheduler
        await runner.start()
        
        # Stop scheduler 
        await runner.stop()
    """
    
    def __init__(self):
        self._watched_locations: List[Dict[str, Any]] = []
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._job_manager = BackgroundJobManager()
    
    def add_watched_location(
        self,
        latitude: float,
        longitude: float,
        name: str = "",
    ):
        """Add a location to be refreshed on schedule."""
        self._watched_locations.append({
            "latitude": latitude,
            "longitude": longitude,
            "name": name or f"{latitude:.2f},{longitude:.2f}",
        })
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("Scheduled job runner started")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduled job runner stopped")
    
    async def _run_scheduler(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Check if it's time for daily refresh (00:00-01:00 UTC)
                now = datetime.utcnow()
                if now.hour == 0 and now.minute < 5:
                    await self._daily_refresh()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Scheduler error: %s", e)
                await asyncio.sleep(60)
    
    async def _daily_refresh(self):
        """Run daily forecast refresh."""
        if not self._watched_locations:
            return
        
        logger.info("Starting daily forecast refresh for %d locations", len(self._watched_locations))
        
        task_id = await self._job_manager.submit_forecast_refresh_job(
            self._watched_locations
        )
        
        # Wait for completion
        while True:
            progress = self._job_manager.get_progress(task_id)
            if not progress or progress.status in (
                JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED
            ):
                break
            await asyncio.sleep(10)
        
        logger.info("Daily refresh completed: %s", progress.message if progress else "unknown")


# ═══════════════════════════════════════════════════════════════════════════
# Global Instance
# ═══════════════════════════════════════════════════════════════════════════

# Global job manager instance
_job_manager: Optional[BackgroundJobManager] = None


def get_job_manager() -> BackgroundJobManager:
    """Get or create global job manager."""
    global _job_manager
    if _job_manager is None:
        _job_manager = BackgroundJobManager()
    return _job_manager


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH APSCHEDULER (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════
#
# If using APScheduler for more robust scheduling:
#
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.triggers.cron import CronTrigger
#
# scheduler = AsyncIOScheduler()
#
# @scheduler.scheduled_job(CronTrigger(hour=0, minute=0))
# async def daily_forecast_refresh():
#     manager = get_job_manager()
#     locations = load_watched_locations_from_db()
#     await manager.submit_forecast_refresh_job(locations)
#
# scheduler.start()
#
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# CELERY INTEGRATION (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════
#
# If using Celery for distributed task processing:
#
# from celery import Celery
#
# celery_app = Celery(
#     "calendar_jobs",
#     broker="redis://localhost:6379/0",
#     backend="redis://localhost:6379/1",
# )
#
# @celery_app.task(bind=True)
# def ingest_historical_data(self, lat, lon, start_year, end_year):
#     from .historical_service import HistoricalWeatherService
#     import asyncio
#     
#     async def run():
#         service = HistoricalWeatherService()
#         try:
#             return await service.ingest_years(lat, lon, start_year, end_year)
#         finally:
#             await service.close()
#     
#     return asyncio.run(run())
#
# # Schedule:
# celery_app.conf.beat_schedule = {
#     'daily-forecast-refresh': {
#         'task': 'tasks.refresh_forecasts',
#         'schedule': crontab(hour=0, minute=0),
#     },
# }
#
# ═══════════════════════════════════════════════════════════════════════════
