"""Constant learning rate schedule."""

from dist_classicrl.schedules.base_schedules import BaseSchedule


class ConstantSchedule(BaseSchedule):
    """Constant learning rate schedule."""

    def __init__(self, value: float) -> None:
        super().__init__(value=value, min_value=value)

    def update(self, steps: int) -> None:
        """Learning rate remains constant."""
