import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from qubit import Qubit
from qubit_gates import q_Rx, q_Ry, q_init
from noise import Noise

# --- DD Sequence Selection ---
PI   = np.pi
CPMG = [lambda rho: q_Ry(PI, rho)]
XY4  = [lambda rho: q_Rx(PI, rho),
        lambda rho: q_Ry(PI, rho),
        lambda rho: q_Rx(PI, rho),
        lambda rho: q_Ry(PI, rho)]

XY8  = [lambda rho: q_Rx(PI, rho),
        lambda rho: q_Ry(PI, rho),
        lambda rho: q_Rx(PI, rho),
        lambda rho: q_Ry(PI, rho),
        lambda rho: q_Ry(PI, rho),
        lambda rho: q_Rx(PI, rho),
        lambda rho: q_Ry(PI, rho),
        lambda rho: q_Rx(PI, rho)]

KDD = [lambda rho: q_Rx(PI, rho),
       lambda rho: q_Ry(PI/2, rho),
       lambda rho: q_Rx(PI, rho),
       lambda rho: q_Ry(PI/2, rho),
       lambda rho: q_Rx(PI, rho)]

DD_SEQUENCE = CPMG                                              # <-- swap here

# --- Config ---
DT                  = 1e-6
TEMPERATURE_K       = 77.0
N_REPEATS           = 20
COHERENCE_THRESHOLD = np.exp(-1)

TAUS         = np.logspace(-6, -3, 10)
N_PULSES_MAX = 32768


def single_repeat(tau: float, n_pulses: int, half_steps: int) -> float:
    qubit = Qubit(rho=q_init(), dt=DT, temperature_Kelvin=TEMPERATURE_K)
    noise = Noise(dt=DT)

    qubit.rho = q_Rx(PI / 2, qubit.rho)

    for i in range(n_pulses):
        for _ in range(half_steps):
            qubit.GAD()
            qubit.rho = noise.apply_noise(qubit.rho)

        qubit.rho = DD_SEQUENCE[i % len(DD_SEQUENCE)](qubit.rho)

        for _ in range(half_steps):
            qubit.GAD()
            qubit.rho = noise.apply_noise(qubit.rho)

    qubit.rho = q_Rx(PI / 2, qubit.rho)
    return float(np.real(qubit.rho[1, 1]))


def run_dd(tau: float, n_pulses: int) -> float:
    half_steps = max(1, int(round(tau / 2 / DT)))
    population_samples = Parallel(n_jobs=-1)(
        delayed(single_repeat)(tau, n_pulses, half_steps) for _ in range(N_REPEATS)
    )
    return float(np.mean(population_samples))


def normalized_contrast(signal: float, baseline: float = 0.5, initial_contrast: float = 0.5) -> float:
    return (signal - baseline) / initial_contrast


# --- Run ---
fig, ax = plt.subplots(figsize=(10, 5))
best_T2_dd = 0.0
best_tau   = None
cycle_len  = len(DD_SEQUENCE)

for tau in TAUS:
    n_pulses_list         = []
    contrast_list         = []
    total_time_at_failure = None

    n = cycle_len
    while n <= N_PULSES_MAX:
        signal   = run_dd(tau, n)
        contrast = normalized_contrast(signal)

        n_pulses_list.append(n)
        contrast_list.append(contrast)

        print(f"  tau={tau*1e6:.1f}us  N={n:6d}  total_time={n * tau * 1e3:.2f}ms  contrast={contrast:.3f}")

        if contrast <= COHERENCE_THRESHOLD:
            total_time_at_failure = n * tau
            break

        n = n * 2
        n = (n // cycle_len) * cycle_len

    total_times_ms = [n * tau * 1e3 for n in n_pulses_list]
    label = f"τ={tau*1e6:.1f}μs  T2_DD={total_time_at_failure*1e3:.1f}ms" if total_time_at_failure else f"τ={tau*1e6:.1f}μs  T2_DD>max"

    ax.plot(total_times_ms, contrast_list, marker='o', linewidth=1.5, label=label)

    if total_time_at_failure and total_time_at_failure > best_T2_dd:
        best_T2_dd = total_time_at_failure
        best_tau   = tau

print(f"\nBest T2_DD = {best_T2_dd*1e3:.2f}ms at tau = {best_tau*1e6:.2f}us")

ax.axhline(y=COHERENCE_THRESHOLD, color='k', linestyle='--', alpha=0.4, label='1/e threshold')
ax.set_xlabel("Total evolution time (ms)")
ax.set_ylabel("Normalized contrast")
ax.set_title(f"DD T2 decay 77K - {'CPMG' if DD_SEQUENCE is CPMG else 'XY4'}")
ax.set_ylim(0.0, 1.1)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()