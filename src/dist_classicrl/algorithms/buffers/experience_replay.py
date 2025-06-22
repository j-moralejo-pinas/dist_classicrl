import numpy as np


class ExperienceReplay:
    """
    Experience Replay buffer for storing and sampling experiences.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.state_buffer = np.empty((capacity,), dtype=int)
        self.action_buffer = np.empty((capacity,), dtype=int)
        self.reward_buffer = np.empty((capacity,), dtype=float)
        self.next_state_buffer = np.empty((capacity,), dtype=int)
        self.done_buffer = np.empty((capacity,), dtype=bool)
        self.position = 0
        self.full = False

    def push(self, experience):
        state, action, reward, next_state, done = experience
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def sample(self, batch_size: int):
        indices = np.random.choice(
            self.capacity if self.full else self.position, batch_size, replace=False
        )
        return (
            self.state_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_state_buffer[indices],
            self.done_buffer[indices],
        )

    def __len__(self):
        return self.capacity if self.full else self.position
