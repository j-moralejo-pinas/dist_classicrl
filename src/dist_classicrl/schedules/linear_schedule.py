"""Linear learning rate schedule."""

from dist_classicrl.schedules.base_schedules import BaseSchedule


class LinearSchedule(BaseSchedule):
    """
    Linear learning rate schedule.

    Attributes
    ----------
    decay_rate : float
        The decay rate for the linear schedule.
    """

    decay_rate: float

    def __init__(self, value: float, decay_rate: float) -> None:
        super().__init__(value=value, min_value=-1e9)
        self.decay_rate = decay_rate

    def update(self, steps: int) -> None:
        """
        Update the learning rate based on the number of steps.

        Parameters
        ----------
        steps : int
            The number of steps that have been taken.
        """
        self.set_value(self.get_value() + steps * self.decay_rate)
