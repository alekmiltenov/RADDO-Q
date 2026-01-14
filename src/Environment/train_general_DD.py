import os
os.system("")

import re
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


from environment import EnvConfig, General_DD_Env

# Custom logs in tensor boardd
class FidelityTBCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._fids = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "fidelity" in info:
                self._fids.append(float(info["fidelity"]))
        return True

    def _on_rollout_end(self) -> None:
        if self._fids:
            self.logger.record("custom/mean_fidelity", float(np.mean(self._fids)))
            self._fids.clear()


def find_latest_checkpoint(folder: str, prefix: str) -> Path | None:
    ckpts = sorted(Path(folder).glob(f"{prefix}_*_steps.zip"))
    if not ckpts:
        return None

    def steps(p: Path) -> int:
        m = re.search(r"_([0-9]+)_steps\.zip$", p.name)
        return int(m.group(1)) if m else -1

    return max(ckpts, key=steps)


def main():
    cfg = EnvConfig(
        temperature_range_K=(1.0, 4.0),
        tau=35e-6,
        T1_0Kelvin=50.0,
        T_phi=20e-6,
        detuning_sigma_hz=2e5,
        max_steps_range=(2, 2)
    )
    check_env(General_DD_Env(cfg), warn=True)

    n_envs = 16                                                     # ! Number of envs
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    prefix = "ppo_dd"
    RUN_NAME = "PPO_1"

    def wrapped_make_env(rank: int):
        def _init():
            env = General_DD_Env(cfg)
            env = Monitor(env)
            env.reset(seed=123 + rank)
            return env
        return _init

    #
    vec_env = SubprocVecEnv([wrapped_make_env(i) for i in range(n_envs)], start_method="spawn")
    vec_env = VecMonitor(vec_env)


    checkpoint_every_timesteps = 100_000
    save_freq = max(checkpoint_every_timesteps // n_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(ckpt_dir),
        name_prefix=prefix
    )

    resume_path = find_latest_checkpoint(str(ckpt_dir), prefix)

    if resume_path is not None:
        print(f"Resuming from: {resume_path}")
        try:
            model = PPO.load(
                resume_path,
                env=vec_env,
                device="cpu",
                tensorboard_log="tb_logs",
            )
        except Exception as e:
            print(f"Failed to load checkpoint ({e}).")
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                device="cpu",
                tensorboard_log="tb_logs",
                ent_coef=0.1,
            )
    else:
        print("No checkpoint found!")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="cpu",
            tensorboard_log="tb_logs",
            ent_coef=0.1,
        )

    # Exploration rate
    model.ent_coef = 0.1

    try:
        fid_cb = FidelityTBCallback()
        callbacks = CallbackList([checkpoint_cb, fid_cb])

        model.learn(
            total_timesteps=2_000_000_000,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=RUN_NAME,
        )

    except KeyboardInterrupt:
        print("Interrupted — saving model...")

        # detach env reference before saving
        model.env = None
        type(model).save(model, str(ckpt_dir / f"{prefix}_interrupt_final"))

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
