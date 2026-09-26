"""Pipeline package for AICatch."""

from pipeline.decoder import VideoStreamDecoder
from pipeline.job_manager import Job, JobManager
try:
    from pipeline.gpu_scheduler import GPUScheduler
except Exception:
    GPUScheduler = None
from pipeline.journey_engine import JourneyEngine
from pipeline.matcher import ActiveQueryMatcher
from pipeline.dispatcher import ResultDispatcher
from pipeline.scene_indexer import SceneIndexer
from pipeline.stitch_engine import StitchEngine
from pipeline.stitch_models import StitchObservation, StitchSegment, StitchTransition, StitchResult, StitchProvenance, StitchContinuitySignals
from pipeline.stitch_validator import StitchValidator

