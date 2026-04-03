import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from qubit import Qubit
from qubit_gates import q_init, q_Rx
from noise import Noise

NUM_RUNS = 100
TEMPERATURE_K = 77.0
DT = 1e-4
MAX_TIME = 30.0

first_relaxation_times = []

for run in range(NUM_RUNS):

    noise = Noise(dt=DT)
    rho0 = q_init()
    rho0 = q_Rx(np.pi, rho0, noise)
    qubit = Qubit(rho=rho0, dt=DT, temperature_Kelvin=TEMPERATURE_K)

    elapsed_time = 0.0
    gad_result = -1

    while elapsed_time < MAX_TIME:
        gad_result = qubit.GAD()
        elapsed_time += DT

        if gad_result == 0:
            first_relaxation_times.append(elapsed_time)
            break

    if gad_result == -1:
        print(f"Run {run}: no jump within {MAX_TIME}s")

    print(f"Run {run}/{NUM_RUNS} done...")


# Stats
relaxation_times = np.array(first_relaxation_times)
print(f"\n--- Results at {TEMPERATURE_K}K ---")
print(f"Relaxation jumps:  {len(relaxation_times)}, mean = {np.mean(relaxation_times):.4f}s")
print(f"Paper T1 at 77K:  ~ 10s")


# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

paper_T1_seconds = 10.0
time_axis = np.linspace(0, MAX_TIME, 1000)
paper_survival_curve = np.exp(-time_axis / paper_T1_seconds)

sorted_relaxation_times = np.sort(relaxation_times)
simulated_survival_curve = 1.0 - np.arange(1, len(sorted_relaxation_times) + 1) / len(sorted_relaxation_times)

axes[1].plot(
    sorted_relaxation_times,
    simulated_survival_curve,
    color="steelblue",
    linewidth=2,
    label="Relaxation simulation"
)
axes[1].plot(
    time_axis,
    paper_survival_curve,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Theory T1 ~ {paper_T1_seconds:.0f}s"
)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("P(no relaxation yet)")
axes[1].set_title(f"Survival Curve | T = {TEMPERATURE_K}K")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[0].hist(relaxation_times, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
axes[0].axvline(
    np.mean(relaxation_times),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {np.mean(relaxation_times):.2f}s"
)
axes[0].axvline(
    paper_T1_seconds,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Paper T1 ~ {paper_T1_seconds:.0f}s"
)
axes[0].set_xlabel("First relaxation time (s)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"T1 Relaxation | T = {TEMPERATURE_K}K, N = {NUM_RUNS}")
axes[0].legend()
axes[0].grid(alpha=0.3)

plt.tight_layout()
plt.show()