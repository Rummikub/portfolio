import asyncio
import uuid
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import time

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobResult:
    """Job execution result"""
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class JobQueue(ABC):
    """Abstract base class for job queues"""
    
    @abstractmethod
    async def enqueue(self, job_type: str, data: Dict[str, Any], **kwargs) -> str:
        """Enqueue a job and return job ID"""
        pass
    
    @abstractmethod
    async def get_status(self, job_id: str) -> Optional[JobResult]:
        """Get job status"""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        pass

class InMemoryJobQueue(JobQueue):
    """In-memory job queue for development/testing"""
    
    def __init__(self):
        self.jobs: Dict[str, JobResult] = {}
        self.job_handlers: Dict[str, Callable] = {}
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def register_handler(self, job_type: str, handler: Callable):
        """Register a job handler function"""
        self.job_handlers[job_type] = handler
    
    async def enqueue(self, job_type: str, data: Dict[str, Any], **kwargs) -> str:
        """Enqueue a job"""
        job_id = str(uuid.uuid4())
        
        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING
        )
        
        # Store job data temporarily
        job_result._job_type = job_type
        job_result._job_data = data
        job_result._job_kwargs = kwargs
        
        self.jobs[job_id] = job_result
        print(f"Enqueued job {job_id} of type {job_type}")
        
        return job_id
    
    async def get_status(self, job_id: str) -> Optional[JobResult]:
        """Get job status"""
        return self.jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                return True
        return False
    
    def _worker(self):
        """Background worker thread"""
        while self._running:
            try:
                # Find pending jobs
                pending_jobs = [
                    (job_id, job) for job_id, job in self.jobs.items()
                    if job.status == JobStatus.PENDING
                ]
                
                for job_id, job in pending_jobs:
                    if not self._running:
                        break
                    
                    job_type = getattr(job, '_job_type', None)
                    job_data = getattr(job, '_job_data', {})
                    
                    if job_type in self.job_handlers:
                        self._execute_job(job_id, job, job_type, job_data)
                    else:
                        print(f"No handler registered for job type: {job_type}")
                        job.status = JobStatus.FAILED
                        job.error = f"No handler for job type: {job_type}"
                        job.completed_at = datetime.now(timezone.utc)
                
                time.sleep(1)  # Check for new jobs every second
                
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(5)
    
    def _execute_job(self, job_id: str, job: JobResult, job_type: str, job_data: Dict[str, Any]):
        """Execute a single job"""
        try:
            print(f"Executing job {job_id} of type {job_type}")
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            
            handler = self.job_handlers[job_type]
            
            # Execute handler (assuming it's synchronous for now)
            if asyncio.iscoroutinefunction(handler):
                # If handler is async, run it in the event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(handler(job_data))
                loop.close()
            else:
                result = handler(job_data)
            
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.now(timezone.utc)
            print(f"Job {job_id} completed successfully")
            
        except Exception as e:
            print(f"Job {job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
    
    def shutdown(self):
        """Shutdown the worker"""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

class CeleryJobQueue(JobQueue):
    """Celery job queue (stub implementation)"""
    
    def __init__(self, broker_url: str = "redis://localhost:6379/0"):
        self.broker_url = broker_url
        # TODO: Initialize Celery app
        print(f"Celery Job Queue initialized (stub) with broker: {broker_url}")
    
    async def enqueue(self, job_type: str, data: Dict[str, Any], **kwargs) -> str:
        """Enqueue a job with Celery (stub)"""
        job_id = str(uuid.uuid4())
        print(f"Would enqueue Celery job {job_id} of type {job_type}")
        # TODO: Use celery.send_task() to enqueue job
        return job_id
    
    async def get_status(self, job_id: str) -> Optional[JobResult]:
        """Get Celery job status (stub)"""
        print(f"Would check status of Celery job {job_id}")
        # TODO: Use celery.AsyncResult to get status
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel Celery job (stub)"""
        print(f"Would cancel Celery job {job_id}")
        # TODO: Use celery.control.revoke() to cancel job
        return True

# Global job queue instance
job_queue: Optional[JobQueue] = None

def init_job_queue(queue_type: str = "memory", **kwargs) -> JobQueue:
    """Initialize job queue"""
    global job_queue
    
    if queue_type == "memory":
        job_queue = InMemoryJobQueue()
    elif queue_type == "celery":
        job_queue = CeleryJobQueue(**kwargs)
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")
    
    # Register default job handlers
    _register_default_handlers()
    
    return job_queue

def get_job_queue() -> JobQueue:
    """Get the global job queue instance"""
    if job_queue is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")
    return job_queue

def _register_default_handlers():
    """Register default job handlers"""
    
    async def agent_run_handler(data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for agent run jobs"""
        agent_id = data.get('agent_id')
        config = data.get('config', {})
        
        # Simulate agent processing
        print(f"Running agent {agent_id} with config: {config}")
        await asyncio.sleep(2)  # Simulate work
        
        return {
            'agent_id': agent_id,
            'status': 'completed',
            'results': {
                'processed_items': 42,
                'insights_generated': 7,
                'recommendations': ['recommendation1', 'recommendation2']
            }
        }
    
    def market_research_handler(data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for market research jobs"""
        query = data.get('query')
        
        # Simulate market research
        print(f"Conducting market research for: {query}")
        time.sleep(3)  # Simulate work
        
        return {
            'query': query,
            'findings': ['finding1', 'finding2', 'finding3'],
            'confidence_score': 0.85
        }
    
    if isinstance(job_queue, InMemoryJobQueue):
        job_queue.register_handler('agent_run', agent_run_handler)
        job_queue.register_handler('market_research', market_research_handler)

# Convenience functions
async def enqueue_agent_run(agent_id: str, config: Dict[str, Any] = None) -> str:
    """Enqueue an agent run job"""
    return await get_job_queue().enqueue('agent_run', {
        'agent_id': agent_id,
        'config': config or {}
    })

async def enqueue_market_research(query: str) -> str:
    """Enqueue a market research job"""
    return await get_job_queue().enqueue('market_research', {
        'query': query
    })