"""
Unit tests for
:class:`dist_classicrl.algorithms.base_algorithms.q_learning_optimal.OptimalQLearningBase`.
"""  # noqa: D205

from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from dist_classicrl.algorithms.base_algorithms.q_learning_optimal import (
    OptimalQLearningBase,
)


def test_init_q_table_shape_and_zeros() -> None:
    """Q-table is created with the expected shape and zeros."""
    ql = OptimalQLearningBase(state_size=3, action_size=4, discount_factor=0.9, seed=123)
    assert ql.q_table.shape == (3, 4)
    assert np.all(ql.q_table == 0)


def test_accepts_numpy_integer_sizes() -> None:
    """Constructor accepts NumPy integer types for sizes."""
    ql = OptimalQLearningBase(state_size=np.int32(2), action_size=np.int64(3))
    assert ql.q_table.shape == (2, 3)


def test_set_and_get_q_value() -> None:
    """Setting and getting a single Q-value works as expected."""
    ql = OptimalQLearningBase(3, 4)
    ql.set_q_value(1, 2, 0.5)
    assert ql.get_q_value(1, 2) == 0.5


def test_add_q_value() -> None:
    """Adding to a single Q-value accumulates correctly."""
    ql = OptimalQLearningBase(3, 4)
    ql.set_q_value(0, 0, 1.0)
    ql.add_q_value(0, 0, 0.25)
    assert ql.get_q_value(0, 0) == 1.25


def test_add_q_values_and_get_q_values() -> None:
    """Vectorized add/get operations update and retrieve point-wise values."""
    ql = OptimalQLearningBase(4, 5)
    states: npt.NDArray[np.int32] = np.array([0, 1, 2, 3], dtype=np.int32)
    actions: npt.NDArray[np.int32] = np.array([1, 2, 3, 4], dtype=np.int32)
    values: npt.NDArray[np.float64] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    ql.add_q_values(states, actions, values)

    # Verify point-wise
    for s, a, v in zip(states, actions, values, strict=True):
        assert ql.get_q_value(int(s), int(a)) == v

    # get_q_values should match provided values
    got = ql.get_q_values(states, actions)
    assert np.allclose(got, values)


def test_add_q_values_accumulates_duplicates() -> None:
    """add_q_values must accumulate when repeating the same (state, action)."""
    ql = OptimalQLearningBase(4, 5)
    states: npt.NDArray[np.int32] = np.array([2, 2, 3], dtype=np.int32)
    actions: npt.NDArray[np.int32] = np.array([4, 4, 1], dtype=np.int32)
    values: npt.NDArray[np.float64] = np.array([0.5, 0.3, 1.0], dtype=np.float64)

    ql.add_q_values(states, actions, values)

    # (2,4) receives two updates: 0.5 + 0.3 = 0.8
    assert ql.get_q_value(2, 4) == 0.8
    # (3,1) receives one update
    assert ql.get_q_value(3, 1) == 1.0

    # Ensure no unintended modifications elsewhere
    expected = np.zeros((4, 5), dtype=np.float64)
    expected[2, 4] = 0.8
    expected[3, 1] = 1.0
    assert np.array_equal(ql.q_table, expected)


def test_get_state_and_states_q_values() -> None:
    """Row slicing helpers for single and multiple states return expected values."""
    ql = OptimalQLearningBase(3, 4)
    # Fill with deterministic values 0..11
    ql.q_table = np.arange(12, dtype=np.float64).reshape(3, 4)

    # Single state row
    row1 = ql.get_state_q_values(1)
    assert np.array_equal(row1, np.array([4, 5, 6, 7], dtype=np.float64))

    # Multiple states rows in order
    rows = ql.get_states_q_values(np.array([2, 0], dtype=np.int32))
    expected = np.array([[8, 9, 10, 11], [0, 1, 2, 3]], dtype=np.float64)
    assert np.array_equal(rows, expected)


def test_get_action_and_actions_q_values() -> None:
    """Column slicing helpers for single and multiple actions return expected values."""
    ql = OptimalQLearningBase(3, 4)
    ql.q_table = np.arange(12, dtype=np.float64).reshape(3, 4)

    # Single action column
    col2 = ql.get_action_q_values(2)
    assert np.array_equal(col2, np.array([2, 6, 10], dtype=np.float64))

    # Multiple actions columns in order
    cols = ql.get_actions_q_values(np.array([3, 1], dtype=np.int32))
    expected = np.array([[3, 1], [7, 5], [11, 9]], dtype=np.float64)
    assert np.array_equal(cols, expected)


def test_save_roundtrip(tmp_path: Path) -> None:
    """Saving and loading the Q-table via NumPy round-trips equivalently."""
    ql = OptimalQLearningBase(2, 3)
    ql.q_table = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]], dtype=np.float64)

    out = tmp_path / "q_table.npy"
    ql.save(str(out))

    loaded = np.load(out)
    assert np.array_equal(loaded, ql.q_table)


def _invoke_learn(
    ql: OptimalQLearningBase,
    method_name: str,
    states: npt.NDArray[np.int32],
    actions: npt.NDArray[np.int32],
    rewards: npt.NDArray[np.float32],
    next_states: npt.NDArray[np.int32],
    terminated: npt.NDArray[np.bool_],
    *,
    lr: float = 1.0,
    next_action_masks: npt.NDArray[np.int32] | None = None,
) -> None:
    """Invoke a learn method variant with optional masks."""
    method = getattr(ql, method_name)
    if next_action_masks is None:
        method(states, actions, rewards, next_states, terminated, lr)
    else:
        method(states, actions, rewards, next_states, terminated, lr, next_action_masks)


@pytest.mark.parametrize("method_name", ["learn", "learn_iter", "learn_vec"])
def test_learn_single_without_mask(method_name: str) -> None:
    """Single-sample learning without mask uses the true next-state max."""
    # Setup
    gamma = 0.5
    ql = OptimalQLearningBase(state_size=4, action_size=3, discount_factor=gamma)
    # Next-state q-values for state 1
    ql.q_table[1] = np.array([1.0, 2.0, 0.5], dtype=np.float64)

    states = np.array([0], dtype=np.int32)
    actions = np.array([2], dtype=np.int32)
    rewards = np.array([1.0], dtype=np.float32)
    next_states = np.array([1], dtype=np.int32)
    terminated = np.array([False], dtype=bool)

    # Act
    _invoke_learn(ql, method_name, states, actions, rewards, next_states, terminated, lr=1.0)

    # Expect: target = r + gamma * max(q(next_state)) = 1 + 0.5*2.0 = 2.0
    assert ql.get_q_value(0, 2) == 2.0


@pytest.mark.parametrize("method_name", ["learn", "learn_iter", "learn_vec"])
def test_learn_single_with_mask_forces_suboptimal(method_name: str) -> None:
    """Masking the optimal next action forces a suboptimal target to be used."""
    # Setup
    gamma = 0.5
    ql = OptimalQLearningBase(state_size=4, action_size=3, discount_factor=gamma)
    # Next-state q-values for state 2: optimal would be action 1 (value 3.0)
    ql.q_table[2] = np.array([1.0, 3.0, 2.5], dtype=np.float64)

    states = np.array([0], dtype=np.int32)
    actions = np.array([0], dtype=np.int32)
    rewards = np.array([0.0], dtype=np.float32)
    next_states = np.array([2], dtype=np.int32)
    terminated = np.array([False], dtype=bool)
    # Mask out the optimal action 1, allow actions 0 and 2 only
    next_action_masks = np.array([[1, 0, 1]], dtype=np.int32)

    # Act
    _invoke_learn(
        ql,
        method_name,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        lr=1.0,
        next_action_masks=next_action_masks,
    )

    # Expect: forced max is 2.5 (action 2), target = 0 + 0.5*2.5 = 1.25
    assert ql.get_q_value(0, 0) == 1.25


@pytest.mark.parametrize("method_name", ["learn", "learn_iter", "learn_vec"])
def test_learn_multiple_without_mask_with_duplicate_update(method_name: str) -> None:
    """Vector vs iterative learning differ when (state, action) indices repeat."""
    gamma = 0.5
    ql = OptimalQLearningBase(state_size=5, action_size=3, discount_factor=gamma)
    # Define next-state q-values for states 2 and 3
    ql.q_table[2] = np.array([0.5, 1.5, 1.0], dtype=np.float64)
    ql.q_table[3] = np.array([1.0, 3.0, 2.5], dtype=np.float64)

    # Two distinct cells are updated: (0,2) once and (1,0) twice
    states = np.array([0, 1, 1], dtype=np.int32)
    actions = np.array([2, 0, 0], dtype=np.int32)
    rewards = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    next_states = np.array([2, 3, 3], dtype=np.int32)
    terminated = np.array([False, False, False], dtype=bool)

    # Act
    _invoke_learn(ql, method_name, states, actions, rewards, next_states, terminated, lr=1.0)

    # Expectations:
    # (0,2): target = 1 + 0.5*1.5 = 1.75
    assert ql.get_q_value(0, 2) == 1.75
    # (1,0): duplicate updates
    if method_name == "learn_vec":
        # vectorized path accumulates both deltas when indices repeat: 1.5 + 3.5 = 5.0
        assert ql.get_q_value(1, 0) == 5.0
    else:
        # iterative paths apply sequentially, ending at the last target 3.5
        assert ql.get_q_value(1, 0) == 3.5


@pytest.mark.parametrize("method_name", ["learn", "learn_iter", "learn_vec"])
def test_learn_multiple_with_mask_and_duplicate_update(method_name: str) -> None:
    """Masked multi-sample learning with repeated indices behaves as specified."""
    gamma = 0.5
    ql = OptimalQLearningBase(state_size=5, action_size=3, discount_factor=gamma)
    # Next-state q-values where optimal is masked out
    ql.q_table[2] = np.array([0.5, 1.5, 1.0], dtype=np.float64)  # optimal=1.5 at action 1
    ql.q_table[3] = np.array([1.0, 3.0, 2.5], dtype=np.float64)  # optimal=3.0 at action 1

    states = np.array([0, 1, 1], dtype=np.int32)
    actions = np.array([2, 0, 0], dtype=np.int32)
    rewards = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    next_states = np.array([2, 3, 3], dtype=np.int32)
    terminated = np.array([False, False, False], dtype=bool)

    # Mask out action 1 (the optimal) for all three samples
    next_action_masks = np.array(
        [
            [1, 0, 1],  # for next_state 2 -> forced max is 1.0 at action 2
            [1, 0, 1],  # for next_state 3 -> forced max is 2.5 at action 2
            [1, 0, 1],  # for next_state 3 -> forced max is 2.5 at action 2
        ],
        dtype=np.int32,
    )

    # Act
    _invoke_learn(
        ql,
        method_name,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        lr=1.0,
        next_action_masks=next_action_masks,
    )

    # Expectations with masked suboptimal choices:
    # (0,2): target = 1 + 0.5*1.0 = 1.5
    assert ql.get_q_value(0, 2) == 1.5
    # (1,0): duplicate updates with masked suboptimal value 2.5
    if method_name == "learn_vec":
        # accumulate both contributions: 1.25 + 3.25 = 4.5
        assert ql.get_q_value(1, 0) == 4.5
    else:
        # sequential final value matches the last target
        assert ql.get_q_value(1, 0) == 3.25


# --------------------
# Action selection tests
# --------------------


def _invoke_choose_single(
    ql: OptimalQLearningBase,
    method_name: str,
    state: int,
    *,
    exploration_rate: float,
    deterministic: bool,
    action_mask: list[int] | None = None,
) -> int:
    """Invoke a single-state action selection variant, optionally masked."""
    if method_name in ("choose_action", "choose_action_vec"):
        return getattr(ql, method_name)(state, exploration_rate, deterministic=deterministic)
    if method_name in ("choose_masked_action", "choose_masked_action_vec"):
        assert action_mask is not None
        return getattr(ql, method_name)(
            state, action_mask, exploration_rate, deterministic=deterministic
        )
    msg = f"Unsupported single-state method: {method_name}"
    raise ValueError(msg)


def _invoke_choose_multi(
    ql: OptimalQLearningBase,
    method_name: str,
    states: npt.NDArray[np.int32],
    *,
    exploration_rate: float,
    deterministic: bool,
    action_masks: npt.NDArray[np.int32] | None = None,
) -> npt.NDArray[np.int32]:
    """Invoke a batched action selection variant, optionally with masks."""
    if method_name in ("choose_actions", "choose_actions_iter", "choose_actions_vec_iter"):
        return getattr(ql, method_name)(
            states, exploration_rate, deterministic=deterministic, action_masks=action_masks
        )
    if method_name == "choose_actions_vec":
        assert action_masks is None
        return ql.choose_actions_vec(states, exploration_rate, deterministic=deterministic)
    if method_name == "choose_masked_actions_vec":
        assert action_masks is not None
        return ql.choose_masked_actions_vec(
            states, action_masks, exploration_rate, deterministic=deterministic
        )
    msg = f"Unsupported multi-state method: {method_name}"
    raise ValueError(msg)


@pytest.mark.parametrize("method_name", ["choose_action", "choose_action_vec"])
def test_choose_action_single_deterministic_unmasked_unique_max(method_name: str) -> None:
    """Deterministic selection picks the unique unmasked maximum."""
    ql = OptimalQLearningBase(state_size=2, action_size=3, discount_factor=0.9, seed=123)
    # Unique max at index 1
    ql.q_table[0] = np.array([0.1, 0.8, 0.2], dtype=np.float64)

    action = _invoke_choose_single(ql, method_name, 0, exploration_rate=0.0, deterministic=True)
    assert action == 1


@pytest.mark.parametrize("method_name", ["choose_masked_action", "choose_masked_action_vec"])
def test_choose_action_single_deterministic_masked_unique_max(method_name: str) -> None:
    """Deterministic selection respects masks and picks the masked maximum."""
    ql = OptimalQLearningBase(state_size=2, action_size=3, discount_factor=0.9, seed=123)
    # Global max at index 1 but mask hides it; among masked {0,2}, index 2 is max
    ql.q_table[0] = np.array([0.3, 1.0, 0.7], dtype=np.float64)
    mask = [1, 0, 1]

    action = _invoke_choose_single(
        ql, method_name, 0, exploration_rate=0.0, deterministic=True, action_mask=mask
    )
    assert action == 2


@pytest.mark.parametrize("method_name", ["choose_action", "choose_action_vec"])
def test_choose_action_single_nondeterministic_unmasked_explore(method_name: str) -> None:
    """Exploration path returns a random action when epsilon triggers."""
    ql = OptimalQLearningBase(state_size=2, action_size=3, discount_factor=0.9, seed=123)
    # Q-values irrelevant when exploring
    ql.q_table[0] = np.array([0.1, 0.8, 0.2], dtype=np.float64)

    if method_name == "choose_action":
        with (
            patch.object(ql._rng, "uniform", return_value=0.0),
            patch.object(ql._rng, "randint", return_value=2),
        ):
            action = _invoke_choose_single(
                ql, method_name, 0, exploration_rate=1.0, deterministic=False
            )
    else:  # choose_action_vec
        with (
            patch.object(ql._rng, "random", return_value=0.0),
            patch.object(ql._rng, "randint", return_value=2),
        ):
            action = _invoke_choose_single(
                ql, method_name, 0, exploration_rate=1.0, deterministic=False
            )
    assert action == 2


@pytest.mark.parametrize("method_name", ["choose_masked_action", "choose_masked_action_vec"])
def test_choose_action_single_nondeterministic_masked_single_option(method_name: str) -> None:
    """With a single allowed action, exploration still returns that action."""
    ql = OptimalQLearningBase(state_size=2, action_size=3, discount_factor=0.9, seed=123)
    ql.q_table[0] = np.array([0.3, 1.0, 0.7], dtype=np.float64)
    # Only a single allowed action -> deterministic outcome despite exploration
    mask = [0, 1, 0]
    if method_name == "choose_masked_action":
        with (
            patch.object(ql._rng, "uniform", return_value=0.0),
            patch.object(ql._rng, "choice", return_value=1),
        ):
            action = _invoke_choose_single(
                ql, method_name, 0, exploration_rate=1.0, deterministic=False, action_mask=mask
            )
    else:  # choose_masked_action_vec
        with (
            patch.object(ql._rng, "random", return_value=0.0),
            patch.object(ql._rng, "choice", return_value=1),
        ):
            action = _invoke_choose_single(
                ql, method_name, 0, exploration_rate=1.0, deterministic=False, action_mask=mask
            )
    assert action == 1


@pytest.mark.parametrize(
    "method_name",
    ["choose_actions", "choose_actions_iter", "choose_actions_vec_iter", "choose_actions_vec"],
)
def test_choose_actions_multiple_deterministic_unmasked_unique_max(method_name: str) -> None:
    """Deterministic batched selection picks unique maxima per state."""
    ql = OptimalQLearningBase(state_size=4, action_size=3, discount_factor=0.9, seed=123)
    # Build unique maxima for states 0,2
    ql.q_table[0] = np.array([0.2, 0.9, 0.1], dtype=np.float64)  # -> 1
    ql.q_table[2] = np.array([0.5, 0.4, 0.7], dtype=np.float64)  # -> 2
    states = np.array([0, 2], dtype=np.int32)

    actions = _invoke_choose_multi(
        ql, method_name, states, exploration_rate=0.0, deterministic=True
    )
    assert np.array_equal(actions, np.array([1, 2], dtype=np.int32))


@pytest.mark.parametrize(
    "method_name",
    [
        "choose_actions",
        "choose_actions_iter",
        "choose_actions_vec_iter",
        "choose_masked_actions_vec",
    ],
)
def test_choose_actions_multiple_deterministic_masked_unique_max(method_name: str) -> None:
    """Deterministic batched selection respects masks for maxima per state."""
    ql = OptimalQLearningBase(state_size=4, action_size=3, discount_factor=0.9, seed=123)
    # Global maxima masked out
    ql.q_table[0] = np.array([0.2, 0.9, 0.1], dtype=np.float64)  # mask out 1 -> pick 0
    ql.q_table[2] = np.array([0.5, 0.4, 0.7], dtype=np.float64)  # mask out 2 -> pick 0
    states = np.array([0, 2], dtype=np.int32)
    masks = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.int32)

    actions = _invoke_choose_multi(
        ql,
        method_name,
        states,
        exploration_rate=0.0,
        deterministic=True,
        action_masks=masks,
    )
    assert np.array_equal(actions, np.array([0, 0], dtype=np.int32))


@pytest.mark.parametrize(
    "method_name",
    ["choose_actions", "choose_actions_iter", "choose_actions_vec_iter", "choose_actions_vec"],
)
def test_choose_actions_multiple_nondeterministic_unmasked_explore(method_name: str) -> None:
    """Exploration in batched unmasked mode yields per-sample random actions."""
    # Force exploration for all via mocking
    ql = OptimalQLearningBase(state_size=4, action_size=3, discount_factor=0.9, seed=123)
    ql.q_table[:] = 0.0
    states = np.array([0, 1, 2], dtype=np.int32)

    if method_name == "choose_actions_vec":
        with patch.object(ql, "_np_rng") as np_rng_mock:
            # Return floats < exploration_rate to force exploration for all
            np_rng_mock.random.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            np_rng_mock.integers.return_value = np.array([0, 1, 2])
            actions = _invoke_choose_multi(
                ql, method_name, states, exploration_rate=1.0, deterministic=False
            )
    elif method_name in ("choose_actions", "choose_actions_iter"):
        with (
            patch.object(ql._rng, "uniform", return_value=0.0),
            patch.object(ql._rng, "randint", side_effect=[0, 1, 2]),
        ):
            actions = _invoke_choose_multi(
                ql, method_name, states, exploration_rate=1.0, deterministic=False
            )
    else:  # choose_actions_vec_iter uses choose_action_vec
        with (
            patch.object(ql._rng, "random", return_value=0.0),
            patch.object(ql._rng, "randint", side_effect=[0, 1, 2]),
        ):
            actions = _invoke_choose_multi(
                ql, method_name, states, exploration_rate=1.0, deterministic=False
            )
    assert actions.shape == (3,)
    assert np.array_equal(actions, np.array([0, 1, 2], dtype=np.int32))


@pytest.mark.parametrize(
    "method_name",
    [
        "choose_actions",
        "choose_actions_iter",
        "choose_actions_vec_iter",
        "choose_masked_actions_vec",
    ],
)
def test_choose_actions_multiple_nondeterministic_masked_explore(method_name: str) -> None:
    """Exploration in batched masked mode samples only from allowed actions."""
    # Force exploration; selections must be within mask allowed indices
    ql = OptimalQLearningBase(state_size=4, action_size=4, discount_factor=0.9, seed=123)
    ql.q_table[:] = 0.0
    states = np.array([0, 1, 2], dtype=np.int32)
    masks = np.array(
        [
            [1, 0, 0, 0],  # only 0 allowed -> deterministic 0
            [0, 1, 1, 0],  # 1 or 2 allowed
            [0, 0, 0, 1],  # only 3 allowed -> deterministic 3
        ],
        dtype=np.int32,
    )

    if method_name == "choose_masked_actions_vec":
        with (
            patch.object(ql, "_np_rng") as np_rng_mock,
            patch.object(ql._rng, "choice", side_effect=[0, 2, 3]),
        ):
            # Return floats < exploration_rate to force exploration for all
            np_rng_mock.random.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            actions = _invoke_choose_multi(
                ql,
                method_name,
                states,
                exploration_rate=1.0,
                deterministic=False,
                action_masks=masks,
            )
    elif method_name in ("choose_actions", "choose_actions_iter"):
        with (
            patch.object(ql._rng, "uniform", return_value=0.0),
            patch.object(ql._rng, "choice", side_effect=[0, 2, 3]),
        ):
            actions = _invoke_choose_multi(
                ql,
                method_name,
                states,
                exploration_rate=1.0,
                deterministic=False,
                action_masks=masks,
            )
    else:  # choose_actions_vec_iter uses choose_masked_action_vec -> _rng.random and _rng.choice
        with (
            patch.object(ql._rng, "random", return_value=0.0),
            patch.object(ql._rng, "choice", side_effect=[0, 2, 3]),
        ):
            actions = _invoke_choose_multi(
                ql,
                method_name,
                states,
                exploration_rate=1.0,
                deterministic=False,
                action_masks=masks,
            )
    assert actions.shape == (3,)
    assert np.array_equal(actions, np.array([0, 2, 3], dtype=np.int32))


@pytest.mark.parametrize("method_name", ["choose_action", "choose_action_vec"])
def test_choose_action_tie_breaking_mocked_two_choices(method_name: str) -> None:
    """When a tie occurs, RNG-based tie-breaking is used (mocked here)."""
    ql = OptimalQLearningBase(state_size=1, action_size=3, discount_factor=0.9, seed=123)
    # Tie between actions 0 and 1
    ql.q_table[0] = np.array([0.9, 0.9, 0.1], dtype=np.float64)

    with patch.object(ql._rng, "choice", side_effect=[0, 1]):
        a1 = _invoke_choose_single(ql, method_name, 0, exploration_rate=0.0, deterministic=True)
        a2 = _invoke_choose_single(ql, method_name, 0, exploration_rate=0.0, deterministic=True)

    assert a1 == 0
    assert a2 == 1


@pytest.mark.parametrize(
    "method_name",
    ["choose_actions", "choose_actions_iter", "choose_actions_vec_iter", "choose_actions_vec"],
)
def test_choose_actions_tie_breaking_mocked_array_unmasked(method_name: str) -> None:
    """Batched unmasked tie-breaking follows RNG (mock-controlled)."""
    ql = OptimalQLearningBase(state_size=3, action_size=3, discount_factor=0.9, seed=123)
    # Two states with ties among actions 0 and 1
    ql.q_table[0] = np.array([0.5, 0.5, 0.1], dtype=np.float64)
    ql.q_table[2] = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    states = np.array([0, 2], dtype=np.int32)

    with patch.object(ql._rng, "choice", side_effect=[0, 1]):
        actions = _invoke_choose_multi(
            ql, method_name, states, exploration_rate=0.0, deterministic=True
        )

    assert np.array_equal(actions, np.array([0, 1], dtype=np.int32))


@pytest.mark.parametrize(
    "method_name",
    [
        "choose_actions",
        "choose_actions_iter",
        "choose_actions_vec_iter",
        "choose_masked_actions_vec",
    ],
)
def test_choose_actions_tie_breaking_mocked_array_masked(method_name: str) -> None:
    """Batched masked tie-breaking follows RNG among allowed actions."""
    ql = OptimalQLearningBase(state_size=3, action_size=3, discount_factor=0.9, seed=123)
    # State 0: tie between 0 and 2 (mask hides 1)
    ql.q_table[0] = np.array([0.7, 0.7, 0.7], dtype=np.float64)
    # State 1: tie between 1 and 2 (mask hides 0)
    ql.q_table[1] = np.array([0.1, 0.9, 0.9], dtype=np.float64)
    states = np.array([0, 1], dtype=np.int32)
    masks = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32)

    with patch.object(ql._rng, "choice", side_effect=[2, 1]):
        actions = _invoke_choose_multi(
            ql,
            method_name,
            states,
            exploration_rate=0.0,
            deterministic=True,
            action_masks=masks,
        )

    assert np.array_equal(actions, np.array([2, 1], dtype=np.int32))
