import gymnasium as gym
import numpy as np
from gymnasium import spaces

from eval_BSG import evaluate_sequence


class SequenceGeneratorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,  tau_values: list[float], seed: int | None = None,):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        self.tau_values = tau_values
        self.allowed_sequence_lengths = [4, 8, 10, 16, 32]
        self.sequence_length = None

        self.sequence = []
        self.current_step = 0

        self.action_space = spaces.Discrete(3)

        # [sequence length,
        #  current generation step,
        #  generation steps left
        #  previous action]
        self.observation_space = spaces.Box(
        low=np.array([1, 0, 0, -1], dtype=np.float32),
        high=np.array([32, 32, 32, 2], dtype=np.float32),
        shape=(4,),
        dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.sequence = []
        self.current_step = 0
        self.sequence_length = int(self.rng.choice(self.allowed_sequence_lengths))
        
        obs = self._get_obs()
        info = {}
        
        return obs, info

    def step(self, action: int):

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.sequence.append(action)
        self.current_step += 1

        terminated = self._sequence_complete()
        truncated = False

        reward = 0.0
        info = {
            "current_step": self.current_step,
            "sequence_so_far": list(self.sequence),
            "sequence_length": self.sequence_length,
        }

        if terminated:
            eval_result = self._evaluate_finished_sequence()

            reward = float(eval_result) * 100.0
            info["raw_score"] = float(eval_result)
            info["sequence_length"] = self.sequence_length

            info["sequence_tokens"] = self._decode_sequence(self.sequence)

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:

        sequence_length = self.sequence_length
        current_generation_step = self.current_step
        generation_steps_left = sequence_length - current_generation_step
        previous_action = self._get_previous_action()

        obs = np.array(
            [sequence_length, current_generation_step, generation_steps_left, previous_action],
            dtype=np.float32,
        )
        return obs

    def _get_previous_action(self) -> float:
        if len(self.sequence) == 0:
            return -1.0

        previous_action = self.sequence[-1]
        return float(previous_action)

    def _decode_sequence(self, sequence: list[int]) -> list[str]:
        token_map = {
            0: "I",
            1: "Xpi",
            2: "Ypi",
        }
        return [token_map[a] for a in sequence]

    def _sequence_complete(self) -> bool:
        return self.current_step >= self.sequence_length

    def _evaluate_finished_sequence(self) -> float:

        sequence = self.sequence

        result = evaluate_sequence(
            sequence=sequence,
            tau_values=self.tau_values,
        )
        return result