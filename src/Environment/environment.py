from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from .qubit import Qubit
    from . import qubit_gates
    from . import noise
except Exception:
    from qubit import Qubit
    import qubit_gates
    import noise



@dataclass
class EnvConfig:
    max_steps_range: Tuple[int, int] = (2, 50)                # Episode Length
    temperature_range_K: Tuple[float, float] = (1.0, 10.0)    # Temperature Range
    tau: float = 0.6e-6                                       # Time between Pulses ; Noise Window
    T1_0Kelvin: float = 50.0                                  # T1 Relaxation Rime at 0 Kelvin (Qubit Property)
    T_phi: float = 60e-6                                      # Pure Phase damping             (Qubit Property)
    gamma1_0Kelvin: float = 0.0
    omega_01: float = 0.0
    detuning_sigma_hz: float = 4e5                            # Quasi-Static Detuning Noise (Hz)

    reward_every: int = 2                                    # Reward shaping checkpoint period (steps); 0 disables



class General_DD_Env(gym.Env):
    """
    Open-loop DD environment:
      - Observation: [t_norm, last_action_norm]  (NO state given to agent)
      - Action: Discrete(3): 0 NONE, 1 Rx(pi), 2 Ry(pi)
      - Internal sim: density matrix rho (2x2 complex)
      - Each step: apply chosen pulse, then apply noise for cfg.tau
      - Reward: terminal-only fidelity to the episode's initial state (state preservation)
    """

    metadata = {"render_modes": ["human"]}


    def __init__(self, cfg: EnvConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        # Episode Step Range
        min_steps, max_steps = self.cfg.max_steps_range
        self._min_steps = int(min_steps)
        self._max_steps_cap = int(max_steps)
        self.episode_max_steps = self._max_steps_cap

        self.action_space = spaces.Discrete(3)                  # Action space      [None , q_Rx(pi) , q_Ry(pi)]
        self.observation_space = spaces.Box(                    # Observation space [t_now, t_max, last_action_norm]
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([float(self._max_steps_cap), float(self._max_steps_cap), 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Density matrix Rho
        self.rho: np.ndarray = np.eye(2, dtype=np.complex128) / 2.0
        self.rho0: np.ndarray = np.eye(2, dtype=np.complex128) / 2.0

        # Episode trackers
        self.t = 0
        self.last_action = 0
        self.temperature_K = 1.0

        # Quasi-static detuning noise
        self._detuning_rad_s: Optional[float] = None

        # Reward shaping tracker (checkpoint fidelity)
        self._last_checkpoint_fidelity: float = 1.0

        # TODO - add the other types of noise

    # Generate random state the agent will try to keep
    def generate_random_qbit_state(self) -> np.ndarray:
        u = float(self.np_random.random())
        v = float(self.np_random.random())
        theta = np.arccos(1.0 - 2.0 * u)
        phi = 2.0 * np.pi * v
        psi = np.array(
            [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
            dtype=np.complex128
        )
        return np.outer(psi, np.conjugate(psi))


    def get_observation(self) -> np.ndarray:
        t_now = self.t
        t_max = self.episode_max_steps
        last_norm = self.last_action / 2.0
        return np.array([t_now, t_max, last_norm], dtype=np.float32)

    def apply_action(self, action: int) -> None:
        if action == 0:
            return
        if action == 1:
            self.rho = qubit_gates.q_Rx(np.pi, self.rho)
            return
        if action == 2:
            self.rho = qubit_gates.q_Ry(np.pi, self.rho)
            return
        raise ValueError(f"Unknown action: {action}")


    # Apply different types of noise each step
    def _apply_noise(self) -> None:
        dt = float(self.cfg.tau)
        if dt <= 0.0:
            return

        # Quasi-static detuning
        if self.cfg.detuning_sigma_hz > 0.0:
            self.rho, self._detuning_rad_s = noise.quasi_static_detuning_step(
                self.rho,
                dt,
                float(self.cfg.detuning_sigma_hz),
                self.np_random,
                self._detuning_rad_s
            )

        # Phase Damp
        if self.cfg.T_phi > 0.0:
            self.rho = Qubit.phase_damp_step(self.rho, float(self.cfg.T_phi), dt)

        # GAD
        self.rho, _, _, _ = Qubit.GAD_step(
            self.rho,
            float(self.cfg.gamma1_0Kelvin),
            float(self.cfg.T1_0Kelvin),
            float(self.cfg.omega_01),
            float(self.temperature_K),
            dt,
            self.np_random
        )

    # Calulate Fidelity
    def _terminal_fidelity(self) -> float:
        F = float(np.real(np.trace(self.rho0 @ self.rho)))
        if not np.isfinite(F):
            return 0.0
        return float(np.clip(F, 0.0, 1.0))


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.t = 0
        self.last_action = 0
        self._detuning_rad_s = None

        lo, hi = self.cfg.temperature_range_K
        self.temperature_K = float(self.np_random.uniform(float(lo), float(hi)))

        lo_i = int(self._min_steps)
        hi_i = int(self._max_steps_cap)
        n = int(self.np_random.integers(lo_i, hi_i + 1))

        if n % 2 == 1:
            if n + 1 <= hi_i:
                n = n + 1
            elif n - 1 >= lo_i:
                n = n - 1

        self.episode_max_steps = int(n)

        # Start from a random state
        self.rho0 = self.generate_random_qbit_state()
        self.rho = self.rho0.copy()

        # checkpoint fidelity baseline
        self._last_checkpoint_fidelity = 1.0

        obs = self.get_observation()
        info = {"temperature_K": self.temperature_K}
        return obs, info



    def step(self, action):
        action = int(action)
        self.t += 1

        self.apply_action(action)
        self.last_action = action

        self._apply_noise()
        info = {"t": self.t, "temperature_K": self.temperature_K}

        terminated = False
        truncated = self.t >= self.episode_max_steps

        reward = 0.0
        fidelity = None

        # Reward shaping checkpoint every K steps (progress-based, weighted toward end)
        if (not truncated) and (self.cfg.reward_every > 0) and (self.t % int(self.cfg.reward_every) == 0):
            fidelity = self._terminal_fidelity()
            w = self.t / float(self.episode_max_steps)
            reward += 0.1 * float(w) * float(fidelity - self._last_checkpoint_fidelity)
            self._last_checkpoint_fidelity = float(fidelity)

        if terminated or truncated:
            fidelity = self._terminal_fidelity()
            info["fidelity"] = fidelity
            f = float(np.clip(fidelity, 0.0, 0.999))
            reward = ((-np.log(1.0 - f)) - 1)

            if fidelity >= 0.60:
                reward += 2.5
            if fidelity >= 0.65:
                reward += 5.5
            if fidelity >= 0.70:
                reward += 12.5
            if fidelity >= 0.85:
                reward += 2.5
            if fidelity >= 0.925:
                reward += 4.5
            if fidelity >= 0.95:
                reward += 5.5
            if fidelity >= 0.98:
                reward += 10.5

        obs = self.get_observation()
        return obs, float(reward), terminated, truncated, info


    def render(self):
        if self.render_mode != "human":
            return
        fidelity = self._terminal_fidelity() if self.t >= self.episode_max_steps else float("nan")
        print(f"t={self.t}/{self.episode_max_steps} last_action={self.last_action} T={self.temperature_K:.3f}K fidelity={fidelity}")

    def close(self):
        pass


def make_env(cfg: EnvConfig, rank: int, base_seed: int = 0) -> Callable[[], General_DD_Env]:
    def _init():
        env = General_DD_Env(cfg)
        env.reset(seed=base_seed + rank)
        return env
    return _init


