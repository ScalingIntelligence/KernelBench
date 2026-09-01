"""Core abstraction for generated and direct-input operator execution."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from mate.execution_context import dry_run_context

WorkloadT = TypeVar("WorkloadT")
InputsT = TypeVar("InputsT")
OutputsT = TypeVar("OutputsT")
ReferenceT = TypeVar("ReferenceT")


class OpMode(Enum):
    NORMAL = "normal"
    DRY_RUN = "dry-run"
    DUMP = "dump"


class Operator(ABC, Generic[WorkloadT, InputsT, OutputsT, ReferenceT]):
    """Define how an operator is prepared, executed, and verified."""

    def __init__(self) -> None:
        self.mode = OpMode.NORMAL

    @abstractmethod
    def generate(self, workload: WorkloadT) -> InputsT:
        """Generate concrete operator inputs from a workload description."""
        raise NotImplementedError

    @abstractmethod
    def call(self, inputs: InputsT) -> OutputsT:
        """Execute the operator with concrete inputs."""
        raise NotImplementedError

    @abstractmethod
    def reference(self, inputs: InputsT) -> ReferenceT:
        """Produce the reference result for concrete inputs."""
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        inputs: InputsT,
        outputs: OutputsT,
        reference: ReferenceT,
    ) -> None:
        """Raise when operator outputs do not satisfy the reference result."""
        raise NotImplementedError

    def __call__(self, inputs: InputsT) -> OutputsT:
        """Execute the operator directly with concrete inputs."""
        return self.call(inputs)

    @contextmanager
    def use_mode(self, mode: OpMode):
        prev_mode = self.mode
        self.mode = mode
        try:
            if mode is OpMode.NORMAL:
                yield self
            else:
                with self._mode_context(mode):
                    yield self
        finally:
            self.mode = prev_mode

    def _mode_context(self, mode: OpMode):
        # OpMode.NORMAL will never enter here.
        match mode:
            case OpMode.DRY_RUN:
                return dry_run_context()
            case OpMode.DUMP:
                return nullcontext()
        return nullcontext()

    def dry_run(self):
        return self.use_mode(OpMode.DRY_RUN)

    def dump(self):
        return self.use_mode(OpMode.DUMP)
