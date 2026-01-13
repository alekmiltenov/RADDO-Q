import os 
os.system("")

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import random
import matplotlib.pyplot as plt

from qubit import Qubit
from qubit_gates import q_init, q_excite


def nth_bose(omega_01: float, temperature_K: float) -> float:
    if omega_01 <= 0.0 or temperature_K <= 0.0:
        return 0.0
    hbar = 1.054571817e-34
    kB = 1.380649e-23
    x = (hbar * omega_01) / (kB * temperature_K)
    x = min(max(float(x), 1e-12), 700.0)
    return float(1.0 / (np.exp(x) - 1.0))


def extract_T1_from_data(times):
    return np.mean(times)


def predict_T1(T1, measurement_relative_error=0.1):
    eps = measurement_relative_error * (2 * random.random() - 1)
    return T1 * (1 + eps)


def measure_relaxation_experiment(T1_0Kelvin, omega_01, temperature_K, N=3000, dt=1.0, t_max=1000.0):
    times = []
    rng = np.random.default_rng()
    steps = int(round(t_max / dt))
    gamma1_0Kelvin = 0.0

    for _ in range(N):
        rho = np.zeros((2, 2), dtype=np.complex128)
        rho = q_init(rho)
        rho = q_excite(rho)

        for n in range(steps + 1):
            elapsed_time = n * dt
            rho, jumped, _ = Qubit.relaxation_step(rho, gamma1_0Kelvin, T1_0Kelvin, omega_01, temperature_K, dt, rng)
            if jumped:
                times.append(elapsed_time + dt)
                break

    data = np.array(times)
    extracted_T1 = extract_T1_from_data(data)
    return data, extracted_T1


def measure_excitation_experiment(T1_0Kelvin, omega_01, temperature_K, N=3000, dt=1.0, t_max=1000.0):
    times = []
    rng = np.random.default_rng()
    steps = int(round(t_max / dt))
    gamma1_0Kelvin = 0.0

    for _ in range(N):
        rho = np.zeros((2, 2), dtype=np.complex128)
        rho = q_init(rho)

        for n in range(steps + 1):
            elapsed_time = n * dt
            rho, jumped, _ = Qubit.excitation_step(rho, gamma1_0Kelvin, T1_0Kelvin, omega_01, temperature_K, dt, rng)
            if jumped:
                times.append(elapsed_time + dt)
                break

    data = np.array(times)
    extracted_T1 = extract_T1_from_data(data)
    return data, extracted_T1


def plot_jump_distribution(times, rate_true, rate_pred, title):
    plt.figure()

    plt.hist(
        times,
        bins=50,
        density=True,
        alpha=0.6,
        label="Simulated jump histogram"
    )

    time_axis = np.linspace(0, max(times), 500)

    plt.plot(
        time_axis,
        rate_true * np.exp(-rate_true * time_axis),
        "r",
        label="True"
    )

    plt.plot(
        time_axis,
        rate_pred * np.exp(-rate_pred * time_axis),
        "k--",
        label="Predicted"
    )

    plt.xlabel("Jump time")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    T1_0Kelvin = 100.0
    temperature_K = 1.0
    omega_01 = 2.0 * np.pi * 2.87e9

    dt = 1.0
    t_max = 1000.0
    N = 3000

    measurement_relative_error = 0.1
    T1_0Kelvin_pred = predict_T1(T1_0Kelvin, measurement_relative_error=measurement_relative_error)

    n_th = nth_bose(omega_01, temperature_K)
    gamma0_true = 0.0 if T1_0Kelvin <= 0.0 else 1.0 / T1_0Kelvin
    gamma0_pred = 0.0 if T1_0Kelvin_pred <= 0.0 else 1.0 / T1_0Kelvin_pred

    rate_relax_true = gamma0_true * (1.0 + n_th)
    rate_relax_pred = gamma0_pred * (1.0 + n_th)

    rate_excite_true = gamma0_true * n_th
    rate_excite_pred = gamma0_pred * n_th

    relaxation_times, relaxation_T1_hat = measure_relaxation_experiment(T1_0Kelvin, omega_01, temperature_K, N=N, dt=dt, t_max=t_max)

    print(f"[RELAXATION]")
    print(f"T1_0K true         : {T1_0Kelvin}")
    print(f"Temperature (K)              : {temperature_K}")
    print(f"n_th               : {n_th:.3e}")
    print(f"rate_down (true)   : {rate_relax_true:.3e}  1/s")
    print(f"T1_down (true)     : {(1.0/rate_relax_true) if rate_relax_true>0 else np.inf:.3f} s")
    print(f"Measured mean time : {relaxation_T1_hat:.3f} s")

    plot_jump_distribution(
        relaxation_times,
        rate_relax_true,
        rate_relax_pred,
        "Relaxation jump-time distribution (|1> -> |0|)"
    )

    excitation_times, excitation_T1_hat = measure_excitation_experiment(T1_0Kelvin, omega_01, temperature_K, N=N, dt=dt, t_max=t_max)

    print(f"\n[EXCITATION]")
    print(f"T1_0K true         : {T1_0Kelvin}")
    print(f"Temperature (K)              : {temperature_K}")
    print(f"n_th               : {n_th:.3e}")
    print(f"rate_up (true)     : {rate_excite_true:.3e}  1/s")
    print(f"T1_up (true)       : {(1.0/rate_excite_true) if rate_excite_true>0 else np.inf:.3f} s")
    print(f"Measured mean time : {excitation_T1_hat:.3f} s")

    plot_jump_distribution(
        excitation_times,
        rate_excite_true,
        rate_excite_pred,
        "Excitation jump-time distribution (|0> -> |1|)"
    )


if __name__ == "__main__":
    main()
