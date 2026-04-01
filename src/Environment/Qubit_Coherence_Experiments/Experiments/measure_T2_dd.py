import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt

from qubit import Qubit
from qubit_gates import q_Rx, q_Ry, q_init
from noise import Noise

# --- DD Sequence Selection ---
PI = np.pi

CPMG = [lambda rho: q_Ry(PI, rho)]
XY4  = [lambda rho: q_Rx(PI, rho),
         lambda rho: q_Ry(PI, rho),
         lambda rho: q_Rx(PI, rho),
         lambda rho: q_Ry(PI, rho)]

DD_SEQUENCE = CPMG                                              # <-- swap here

# --- Config ---
DT            = 1e-6
TEMPERATURE_K = 77.0
N_REPEATS     = 5

N_PULSES_LIST = [4096, 8192, 16384, 32768]
TOTAL_TIMES   = np.linspace(1e-5, 0.6, 20)

def dd_signal(total_time: float, n_pulses: int) -> float:
    tau        = total_time / n_pulses
    half_steps = max(1, int(round(tau / 2 / DT)))
    population_repeat_same_tau = []

    for _ in range(N_REPEATS):
        qubit = Qubit(rho=q_init(), dt=DT, temperature_Kelvin=TEMPERATURE_K)
        noise = Noise(dt=DT)

        qubit.rho = q_Rx(PI / 2, qubit.rho)

        for i in range(n_pulses):
            for _ in range(half_steps):
                qubit.rho = noise.apply_noise(qubit.rho)

            qubit.rho = DD_SEQUENCE[i % len(DD_SEQUENCE)](qubit.rho)   # cycle through sequence

            for _ in range(half_steps):
                qubit.rho = noise.apply_noise(qubit.rho)

        qubit.rho = q_Rx(PI / 2, qubit.rho)
        population_repeat_same_tau.append(float(np.real(qubit.rho[1, 1])))

    return float(np.mean(population_repeat_same_tau))


def extract_T2_dd(
                  total_times: np.ndarray,
                  signal_values: list[float],
                  dephased_baseline: float = 0.5
                  ) -> float:

    total_times   = np.asarray(total_times,   dtype=float)
    signal_values = np.asarray(signal_values, dtype=float)

    initial_contrast = signal_values[0] - dephased_baseline
    if initial_contrast <= 0:
        return np.nan

    normalized_contrast = (signal_values - dephased_baseline) / initial_contrast
    one_over_e_level    = np.exp(-1)

    crossing_indices = np.where(normalized_contrast <= one_over_e_level)[0]
    if len(crossing_indices) == 0:
        return np.nan

    crossing_index = crossing_indices[0]
    if crossing_index == 0:
        return total_times[0]

    tau_left       = total_times[crossing_index - 1]
    tau_right      = total_times[crossing_index]
    contrast_left  = normalized_contrast[crossing_index - 1]
    contrast_right = normalized_contrast[crossing_index]

    if contrast_right == contrast_left:
        return tau_right

    return tau_left + (
        (one_over_e_level - contrast_left)
        * (tau_right - tau_left)
        / (contrast_right - contrast_left)
    )


# --- Run ---
fig, ax = plt.subplots(figsize=(9, 5))
total_times_ms = TOTAL_TIMES * 1e3

for n_pulses in N_PULSES_LIST:
    print(f"Running N = {n_pulses}...")
    signal_values = Parallel(n_jobs=-1)(delayed(dd_signal)(t, n_pulses) for t in TOTAL_TIMES)
    T2_dd = extract_T2_dd(TOTAL_TIMES, signal_values)
    label = f"N={n_pulses}  T2_DD={T2_dd*1e3:.2f}ms" if np.isfinite(T2_dd) else f"N={n_pulses}"
    print(f"  T2_DD = {T2_dd*1e3:.3f} ms" if np.isfinite(T2_dd) else f"  T2_DD = not found")

    ax.plot(total_times_ms, signal_values, linewidth=1.5, label=label)

ax.set_xlabel("Total evolution time (ms)")
ax.set_ylabel("⟨ρ₀₀⟩")
ax.set_title("DD T2 decay 77K — CPMG")
ax.set_ylim(0.0, 1.0)
ax.set_xlim(total_times_ms[0], total_times_ms[-1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()