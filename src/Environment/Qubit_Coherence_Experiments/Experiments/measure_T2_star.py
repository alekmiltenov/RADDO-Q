import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from qubit import Qubit
from qubit_gates import q_Rx, q_init
from noise import Noise

# Config
DT            = 1e-6                                            # Lower timestep because dt should be << T2*
TEMPERATURE_K = 77.0                                            # Tempearture in kelvin
N_REPEATS     = 2000                                            # repeat N_REPEATS times for smoother curve

tau_values = np.linspace(0, 1e-4, 1000)                         # initial tau values
tau_steps = np.unique(np.round(tau_values / DT).astype(int))    # filter unique step numbers
TAUS = tau_steps * DT                                           # final filtered tau values

def ramsey_signal(tau: float) -> float:
    steps = int(round(tau / DT))
    population_repeat_same_tau = []

    for _ in range(N_REPEATS):
        qubit = Qubit(rho=q_init(), dt=DT, temperature_Kelvin=TEMPERATURE_K)
        noise = Noise(dt=DT)

        qubit.rho = q_Rx(np.pi / 2, qubit.rho)

        for _ in range(steps):
            # qubit.GAD()
            qubit.rho = noise.apply_noise(qubit.rho)

        qubit.rho = q_Rx(np.pi / 2, qubit.rho)

        # Measure |1> state after pi/2
        population_repeat_same_tau.append(float(np.real(qubit.rho[1, 1])))

    return float(np.mean(population_repeat_same_tau))


def extract_T2_star(
                    tau_values: np.ndarray,
                    ramsey_population_values: list[float],
                    dephased_baseline: float = 0.5
                    ) -> float:
    
    tau_values = np.asarray(tau_values, dtype=float)
    ramsey_population_values = np.asarray(ramsey_population_values, dtype=float)

    # Convert population curve (about 1 -> 0.5) into normalized contrast (1 -> 0)
    initial_contrast = ramsey_population_values[0] - dephased_baseline
    if initial_contrast <= 0:
        return np.nan

    normalized_contrast = (
        ramsey_population_values - dephased_baseline
    ) / initial_contrast

    one_over_e_level = np.exp(-1)

    # Find the first point that drops below the 1/e contrast level
    crossing_indices = np.where(normalized_contrast <= one_over_e_level)[0]
    if len(crossing_indices) == 0:
        return np.nan

    crossing_index = crossing_indices[0]
    if crossing_index == 0:
        return tau_values[0]

    # Linearly interpolate between the two points around the crossing
    tau_left = tau_values[crossing_index - 1]
    tau_right = tau_values[crossing_index]
    contrast_left = normalized_contrast[crossing_index - 1]
    contrast_right = normalized_contrast[crossing_index]

    if contrast_right == contrast_left:
        return tau_right

    return tau_left + (
        (one_over_e_level - contrast_left)
        * (tau_right - tau_left)
        / (contrast_right - contrast_left)
    )



# Get populations for each tau
ramsey_population_values = Parallel(n_jobs=-2)(
    delayed(ramsey_signal)(tau) for tau in TAUS
)

# Extract T2* as a number
T2_star = extract_T2_star(TAUS, ramsey_population_values)


# Plot
fig, ax = plt.subplots(figsize=(8, 5))
taus_us = TAUS * 1e6
ax.plot(taus_us, ramsey_population_values, linewidth=2)
ax.set_xlabel("τ (μs)")
ax.set_ylabel("⟨ρ₁₁⟩")
ax.set_title("Ramsey T2* decay 77K")
ax.set_ylim(0.0, 1.0)
ax.set_xlim(taus_us[0], taus_us[-1])
ax.set_xticks(np.arange(0, taus_us[-1] + 1, 10))
ax.grid(True, alpha=0.3)

if np.isfinite(T2_star):
    ax.text(
        0.98, 0.95,
        f"T2* = {T2_star * 1e6:.2f} μs",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

plt.tight_layout()
plt.show()