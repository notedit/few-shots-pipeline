"""Abstract base class for pipeline stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config import PipelineConfig
from ..models import PipelineContext

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    name: str = "base"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"fspipeline.{self.name}")

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this stage and return updated context."""
        ...

    def validate(self, ctx: PipelineContext) -> None:
        """Validate preconditions before running. Override in subclasses."""
        pass

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Validate and run the stage."""
        self.logger.info(f"Starting stage: {self.name}")
        self.validate(ctx)
        ctx = self.run(ctx)
        self.logger.info(f"Completed stage: {self.name}")
        return ctx
