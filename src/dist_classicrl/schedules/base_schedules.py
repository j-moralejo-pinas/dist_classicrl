"""Base class for learning rate schedules."""

from abc import ABC
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized


class BaseSchedule(ABC):
    """
    Base class for learning rate schedules.

    Attributes
    ----------
    lr : float | Synchronized
        The learning rate.
    min_lr : float
        The minimum learning rate.
    """

    value: float | Synchronized
    min_value: float

    def __init__(self, value: float, min_value: float) -> None:
        self.value = value
        self.min_value = min_value

    def set_mp(self) -> None:
        """Set the learning rate as multiprocessing synchronized values."""
        assert isinstance(self.value, float), "Learning rate must be a float."
        self.value = Value("f", self.value)

    def set_value(self, value: float | Synchronized) -> None:
        """
        Set the learning rate.

        Parameters
        ----------
        value : float
            The new learning rate value.
        """
        if isinstance(value, Synchronized):
            assert isinstance(self.value, Synchronized), (
                "self.value must be a multiprocessing Value."
            )
            self.value = value
        elif isinstance(self.value, Synchronized):
            self.value.value = value
        else:
            self.value = value

    def get_value(self) -> float:
        """
        Get the current learning rate.

        Returns
        -------
        float
            The current learning rate.
        """
        if isinstance(self.value, Synchronized):
            return self.value.value
        return self.value

    def update(self, steps: int) -> None:
        """
        Update the learning rate based on the number of steps.

        Parameters
        ----------
        steps : int
            The number of steps that have been taken.
        """
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)
