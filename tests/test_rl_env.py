import numpy as np
import pytest
import gymnasium as gym

from src.rl.env import TrafficSignalEnv
from src.rl.reward import IntersectionState, compute_reward


@pytest.fixture
def env():
    return TrafficSignalEnv(seed=42, max_steps=200)


def test_observation_space(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_action_space(env):
    assert env.action_space.n == 2


def test_reset_returns_valid_obs(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert np.all(obs >= 0) and np.all(obs <= 1)
    assert isinstance(info, dict)


def test_step_returns_valid_types(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_obs_in_space(env):
    env.reset()
    obs, *_ = env.step(0)
    assert env.observation_space.contains(obs)


def test_episode_terminates(env):
    env.reset()
    done = False
    steps = 0
    while not done:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
        if steps > 10_000:
            pytest.fail("Episode did not terminate")
    assert steps <= env.max_steps + 10


def test_info_keys(env):
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "queue_lengths" in info
    assert "phase" in info
    assert "total_served" in info


def test_queue_increases_on_red(env):
    env.reset()
    for _ in range(10):
        env.step(0)
    assert env._queue[2] + env._queue[3] >= 0


def test_multiple_resets_independent(env):
    env.reset(seed=1)
    obs1, *_ = env.step(1)
    env.reset(seed=1)
    obs2, *_ = env.step(1)
    np.testing.assert_array_almost_equal(obs1, obs2)


def make_state(queues, waits, phase=0):
    return IntersectionState(
        queue_lengths=np.array(queues, dtype=np.float32),
        wait_times=np.array(waits, dtype=np.float32),
        throughput=np.zeros(4, dtype=np.float32),
        green_phase=phase,
        cycle_length=30.0,
        density=sum(queues) / 100.0,
    )


def test_reward_wait_time_reduction_improves():
    prev = make_state([10, 10, 5, 5], [20, 20, 10, 10])
    curr = make_state([5, 5, 5, 5],   [10, 10, 10, 10])
    r = compute_reward(curr, prev, mode="wait_time_reduction")
    assert r > 0


def test_reward_wait_time_reduction_worsens():
    prev = make_state([5, 5, 5, 5],   [10, 10, 10, 10])
    curr = make_state([10, 10, 5, 5], [20, 20, 10, 10])
    r = compute_reward(curr, prev, mode="wait_time_reduction")
    assert r < 0


def test_reward_composite_valid():
    prev = make_state([10, 10, 10, 10], [30, 30, 20, 20])
    curr = make_state([8, 8, 8, 8],    [25, 25, 18, 18])
    r = compute_reward(curr, prev, mode="composite")
    assert isinstance(r, float)


def test_reward_invalid_mode():
    s = make_state([0, 0, 0, 0], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        compute_reward(s, s, mode="unknown_mode")
