import os
from multiprocessing import freeze_support

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from environment_LASG import SequenceGeneratorEnv


LOG_DIR = "runs/sequence_generator"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------------------
# Training config
# ----------------------------
TAU_VALUES = [20e-6, 30e-6, 40e-6, 50e-6]

# "dummy" -> easiest for debugging
# "subproc" -> correct choice for real training when N_ENVS > 1
VEC_ENV_TYPE = "subproc"

N_ENVS = 4
TOTAL_TIMESTEPS = 200_000
BASE_SEED = 1000
RESUME_PATH = "models/generator_rppo_interrupt_safe"

RUN_NAME = "generator_rppo_run"
MONITOR_FILE = os.path.join(LOG_DIR, "vec_monitor.csv")

LEARNING_RATE = 4e-4
N_STEPS = 64
N_EPOCHS = 10

GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

LSTM_HIDDEN_SIZE = 128
NET_ARCH = dict(pi=[128, 128], vf=[128, 128])


def make_env(seed: int, tau_values: list[float]):
    def _init():
        return SequenceGeneratorEnv(
            tau_values=tau_values,
            seed=seed,
        )
    return _init


def build_vec_env(
    n_envs: int,
    vec_env_type: str,
    base_seed: int,
    tau_values: list[float],
):
    env_fns = [make_env(base_seed + i, tau_values) for i in range(n_envs)]

    if n_envs == 1:
        env = DummyVecEnv(env_fns)
    elif vec_env_type == "dummy":
        env = DummyVecEnv(env_fns)
    elif vec_env_type == "subproc":
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        raise ValueError(
            f"Unknown VEC_ENV_TYPE={vec_env_type!r}. Use 'dummy' or 'subproc'."
        )

    return VecMonitor(
        env,
        filename=MONITOR_FILE,
        info_keywords=("raw_score", "sequence_length"),
    )


def choose_batch_size(n_steps: int, n_envs: int, preferred: int = 128) -> int:
    rollout_size = n_steps * n_envs

    if preferred <= rollout_size and rollout_size % preferred == 0:
        return preferred

    return rollout_size


class EpisodeStatsTensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ep_info_buffer = self.model.ep_info_buffer

        if ep_info_buffer is None or len(ep_info_buffer) == 0:
            return

        raw_scores = [ep["raw_score"] for ep in ep_info_buffer if "raw_score" in ep]
        seq_lengths = [ep["sequence_length"] for ep in ep_info_buffer if "sequence_length" in ep]

        if raw_scores:
            self.logger.record("rollout/raw_score_mean", float(np.mean(raw_scores)))
            self.logger.record("rollout/raw_score_max", float(np.max(raw_scores)))
            self.logger.record("rollout/raw_score_min", float(np.min(raw_scores)))

        if seq_lengths:
            self.logger.record("rollout/sequence_length_mean", float(np.mean(seq_lengths)))


if __name__ == "__main__":
    freeze_support()

    batch_size = choose_batch_size(N_STEPS, N_ENVS, preferred=128)
    rollout_size = N_STEPS * N_ENVS

    print("=== Training config ===")
    print(f"VEC_ENV_TYPE   : {VEC_ENV_TYPE}")
    print(f"N_ENVS         : {N_ENVS}")
    print(f"TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}")
    print(f"TAU_VALUES     : {TAU_VALUES}")
    print(f"N_STEPS        : {N_STEPS}")
    print(f"ROLLOUT_SIZE   : {rollout_size}")
    print(f"BATCH_SIZE     : {batch_size}")
    print(f"RESUME_PATH    : {RESUME_PATH}")
    print(f"RUN_NAME       : {RUN_NAME}")
    print("=======================")

    env = build_vec_env(
        n_envs=N_ENVS,
        vec_env_type=VEC_ENV_TYPE,
        base_seed=BASE_SEED,
        tau_values=TAU_VALUES,
    )

    if RESUME_PATH is not None:
        model = RecurrentPPO.load(
            RESUME_PATH,
            env=env,
            tensorboard_log=LOG_DIR,
        )
        print(f"Resumed from: {RESUME_PATH}")
    else:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=batch_size,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            tensorboard_log=LOG_DIR,
            verbose=1,
            policy_kwargs=dict(
                lstm_hidden_size=LSTM_HIDDEN_SIZE,
                net_arch=NET_ARCH,
            ),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5_000 // N_ENVS, 1),
        save_path=MODEL_DIR,
        name_prefix="generator_rppo",
    )

    tensorboard_callback = EpisodeStatsTensorboardCallback()

    callbacks = CallbackList([
        checkpoint_callback,
        tensorboard_callback,
    ])

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            tb_log_name=RUN_NAME,
            reset_num_timesteps=False if RESUME_PATH is not None else True,
            log_interval=1,
            progress_bar=True,
        )
    finally:
        interrupt_path = os.path.join(MODEL_DIR, "generator_rppo_interrupt_safe")
        model.save(interrupt_path)
        print(f"Saved interrupt-safe model to: {interrupt_path}")

        env.close()

    final_path = os.path.join(MODEL_DIR, "generator_rppo_final")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")