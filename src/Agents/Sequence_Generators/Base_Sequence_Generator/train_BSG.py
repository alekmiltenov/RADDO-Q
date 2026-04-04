import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from environment_BSG import SequenceGeneratorEnv


LOG_DIR = "runs/sequence_generator"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(seed: int):
    def _init():
        env = SequenceGeneratorEnv(
            tau_values=[10e-6, 20e-6, 30e-6, 40e-6, 50e-6],
            max_pulses=8192,
            seed=seed,
        )
        return env
    return _init


if __name__ == "__main__":
    N_ENVS = 20
    TOTAL_TIMESTEPS = 200_000
    RESUME_PATH = None

    env = DummyVecEnv([make_env(1000 + i) for i in range(N_ENVS)])
    env = VecMonitor(env)

    if RESUME_PATH is not None:
        model = RecurrentPPO.load(RESUME_PATH, env=env, tensorboard_log=LOG_DIR)
        print(f"Resumed from: {RESUME_PATH}")
    else:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=3e-3,
            n_steps=64,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            verbose=1,
            policy_kwargs=dict(
                lstm_hidden_size=128,
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // N_ENVS, 1),
        save_path=MODEL_DIR,
        name_prefix="generator_rppo",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="generator_rppo_run",
            reset_num_timesteps=False if RESUME_PATH is not None else True,
            progress_bar=True,
        )
    finally:
        interrupt_path = os.path.join(MODEL_DIR, "generator_rppo_interrupt_safe")
        model.save(interrupt_path)
        print(f"Saved interrupt-safe model to: {interrupt_path}")

    final_path = os.path.join(MODEL_DIR, "generator_rppo_final")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

    env.close()