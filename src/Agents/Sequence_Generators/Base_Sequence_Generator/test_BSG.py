import argparse
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO

from environment_BSG import SequenceGeneratorEnv
from eval_BSG import Sweep_Tau_T2_DD, _to_dd_sequence


TAU_VALUES_DEFAULT = [20e-6, 30e-6, 40e-6]
ALLOWED_LENGTHS = [4, 8, 16, 32]


BASELINE_DD_SEQUENCES = {
    "CPMG-Y": [("Y", 1)],
    "CPMG-X": [("X", 1)],
    "XY4": [("X", 1), ("Y", 1), ("X", 1), ("Y", 1)],
    "XY8": [("X", 1), ("Y", 1), ("X", 1), ("Y", 1), ("Y", 1), ("X", 1), ("Y", 1), ("X", 1)],
    "Custom - 3X 1Y": [("X", 1), ("X", 1), ("X", 1), ("Y", 1), ("X", 1), ("X", 1), ("X", 1), ("Y", 1)],
    "Custom - 3Y 1X": [("Y", 1), ("Y", 1), ("Y", 1), ("X", 1), ("Y", 1), ("Y", 1), ("Y", 1), ("X", 1)],
}

ACTION_TO_TOKEN = {
    0: "I",
    1: "Xpi",
    2: "Ypi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained RADDO-Q sequence generator, force a target motif length, "
            "generate one motif, and compare it against canonical CPMG / XY4 / XY8."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a saved RecurrentPPO model zip, e.g. models/generator_rppo_final.zip",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        required=True,
        choices=ALLOWED_LENGTHS,
        help="Target motif length to force during inference.",
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=TAU_VALUES_DEFAULT,
        help="Tau values in seconds, e.g. --taus 2e-5 3e-5 4e-5 5e-5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used for inference env reset.",
    )
    parser.add_argument(
        "--best-by",
        type=str,
        default="env",
        choices=["raw", "env", "fit"],
        help="Metric used to choose the best tau.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for offline tau sweeps.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of using deterministic inference.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots and only print results.",
    )
    return parser.parse_args()


class ForcedLengthSequenceEnv(SequenceGeneratorEnv):
    def __init__(self, tau_values: list[float], forced_length: int, seed: int | None = None):
        super().__init__(tau_values=tau_values, seed=seed)
        if forced_length not in self.allowed_sequence_lengths:
            raise ValueError(
                f"forced_length={forced_length} is not in allowed lengths {self.allowed_sequence_lengths}"
            )
        self.forced_length = int(forced_length)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.sequence = []
        self.current_step = 0
        self.sequence_length = self.forced_length
        return self._get_obs(), info


def dd_sequence_to_tokens(dd_sequence: list[tuple[str, int]]) -> list[str]:
    tokens: list[str] = []
    for axis, angle in dd_sequence:
        if axis == "I":
            tokens.append("I")
        elif angle == 1:
            tokens.append(f"{axis}pi")
        else:
            tokens.append(f"{axis}pi/{angle}")
    return tokens


def generate_sequence(
    model: RecurrentPPO,
    sequence_length: int,
    tau_values: list[float],
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    env = ForcedLengthSequenceEnv(
        tau_values=tau_values,
        forced_length=sequence_length,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)

    lstm_state = None
    episode_start = np.array([True], dtype=bool)

    terminated = False
    truncated = False
    reward = 0.0
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        action_int = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = env.step(action_int)
        episode_start = np.array([False], dtype=bool)

    sequence_indices = list(env.sequence)
    sequence_tokens = [ACTION_TO_TOKEN[a] for a in sequence_indices]
    dd_sequence = _to_dd_sequence(sequence_indices)

    return {
        "sequence_indices": sequence_indices,
        "sequence_tokens": sequence_tokens,
        "dd_sequence": dd_sequence,
        "reward": float(reward),
        "raw_score": float(info.get("raw_score", 0.0)),
        "sequence_length": int(info.get("sequence_length", sequence_length)),
    }


def evaluate_named_dd_sequence(
    name: str,
    dd_sequence: list[tuple[str, int]],
    tau_values: list[float],
    n_jobs: int,
    best_by: str,
) -> dict[str, Any]:
    results, best_tau, best_t2 = Sweep_Tau_T2_DD(
        taus=tau_values,
        dd_sequence=dd_sequence,
        n_jobs=n_jobs,
        best_by=best_by,
    )

    if best_by == "raw":
        metric_key = "t2_raw"
    elif best_by == "fit":
        metric_key = "t2_fit"
    else:
        metric_key = "t2_env"

    best_result = None
    if not math.isnan(best_t2):
        valid_results = [r for r in results if not np.isnan(r[metric_key])]
        if valid_results:
            best_result = max(valid_results, key=lambda r: r[metric_key])

    return {
        "name": name,
        "tokens": dd_sequence_to_tokens(dd_sequence),
        "dd_sequence": dd_sequence,
        "results": results,
        "best_tau": best_tau,
        "best_t2": best_t2,
        "best_metric_key": metric_key,
        "best_result": best_result,
    }


def print_generated_block(generated: dict[str, Any]) -> None:
    print("\n=== Generated motif ===")
    print(f"Length          : {generated['sequence_length']}")
    print(f"Action indices  : {generated['sequence_indices']}")
    print(f"Tokens          : {generated['sequence_tokens']}")
    print(f"DD sequence     : {generated['dd_sequence']}")
    print(f"Env raw score   : {generated['raw_score']:.6f}")
    print(f"Env reward      : {generated['reward']:.6f}")


def print_summary(evaluations: list[dict[str, Any]]) -> None:
    print("\n=== Motif comparison ===")
    header = (
        f"{'Name':<12} {'Motif len':>10} {'Best tau (us)':>14} "
        f"{'Best T2 (s)':>14} {'Pulses used':>12}"
    )
    print(header)
    print("-" * len(header))

    for ev in evaluations:
        best_result = ev["best_result"]
        best_tau_us = float(ev["best_tau"] * 1e6) if not math.isnan(ev["best_tau"]) else float("nan")
        best_t2 = float(ev["best_t2"]) if not math.isnan(ev["best_t2"]) else float("nan")
        pulses_used = best_result["n_pulses_used"] if best_result is not None else "-"

        print(
            f"{ev['name']:<12} {len(ev['dd_sequence']):>10} {best_tau_us:>14.3f} "
            f"{best_t2:>14.6f} {str(pulses_used):>12}"
        )
        print(f"  tokens: {ev['tokens']}")


def plot_best_curves(evaluations: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(10, 6))

    for ev in evaluations:
        best_result = ev["best_result"]
        if best_result is None:
            continue
        tau_us = best_result["tau"] * 1e6
        plt.plot(best_result["times"], best_result["coherences"], label=f"{ev['name']} @ {tau_us:.1f} us")

    plt.axhline(1 / np.e, linestyle="--", label="1/e")
    plt.xlabel("Time (s)")
    plt.ylabel("Coherence")
    plt.title("Best coherence curve per motif")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_best_t2_bar(evaluations: list[dict[str, Any]]) -> None:
    names = [ev["name"] for ev in evaluations]
    values = [0.0 if math.isnan(ev["best_t2"]) else float(ev["best_t2"]) for ev in evaluations]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Best extracted T2 (s)")
    plt.title("Best T2 by motif")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    model = RecurrentPPO.load(args.model_path)

    generated = generate_sequence(
        model=model,
        sequence_length=args.sequence_length,
        tau_values=args.taus,
        seed=args.seed,
        deterministic=not args.stochastic,
    )

    evaluations = [
        evaluate_named_dd_sequence(
            name="Generator",
            dd_sequence=generated["dd_sequence"],
            tau_values=args.taus,
            n_jobs=args.n_jobs,
            best_by=args.best_by,
        )
    ]

    for baseline_name, baseline_dd_sequence in BASELINE_DD_SEQUENCES.items():
        evaluations.append(
            evaluate_named_dd_sequence(
                name=baseline_name,
                dd_sequence=baseline_dd_sequence,
                tau_values=args.taus,
                n_jobs=args.n_jobs,
                best_by=args.best_by,
            )
        )

    print(f"\nLoaded model: {args.model_path}")
    print(f"Forced sequence length: {args.sequence_length}")
    print(f"Inference mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print_generated_block(generated)
    print_summary(evaluations)

    if not args.no_plot:
        plot_best_curves(evaluations)
        plot_best_t2_bar(evaluations)


if __name__ == "__main__":
    main()
