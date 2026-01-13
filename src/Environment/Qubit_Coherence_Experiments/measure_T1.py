import os 
os.system("")

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import random
import matplotlib.pyplot as plt

from qubit import Qubit
from qubit_gates import q_init, q_excite


# T1 Decay experiment
def measure_T1_experiment(T1_true, N=3000, dt=1.0, t_max=1000.0):
    times = []
    rng = np.random.default_rng()
    steps = int(round(t_max / dt))

    for _ in range(N):
        rho = np.zeros((2, 2), dtype=np.complex128)
        rho = q_init(rho)
        rho = q_excite(rho)

        for n in range(steps + 1):
            elapsed_time = n * dt
            rho, jumped, _ = Qubit.amplitude_damp_step(rho, T1_true, dt, rng)
            if jumped:
                times.append(elapsed_time + dt)
                break
    data = np.array(times)
    extracted_T1 = extract_T1_from_data(data)
    return np.array(data), extracted_T1


# Extract T1
def extract_T1_from_data(times):
    return np.mean(times)


# Predict T1
def predict_T1(T1_true, measurement_relative_error=0.1):
    eps = measurement_relative_error * (2 * random.random() - 1)
    return T1_true * (1 + eps)



# Graph
def main():
    T1_true = 100.0

    times, T1_hat = measure_T1_experiment(T1_true)
    measurement_relative_error = 0.1
    T1_pred = predict_T1(T1_true, measurement_relative_error=measurement_relative_error)


    print(f"True T1      : {T1_true}")
    print(f"Measured T1̂ : {T1_hat:.2f}")

    plt.figure()

    plt.hist(
        times,
        bins=50,
        density=True,
        alpha=0.6,
        label="Measured jump"
    )

    time_axis = np.linspace(0, max(times), 500)
    plt.plot(time_axis, (1 / T1_true) * np.exp(-time_axis / T1_true),
             "r", label="True T1 ")
    
    plt.plot(time_axis, (1 / T1_pred) * np.exp(-time_axis / T1_pred),
             "k--", label=f"Predicted (T1_pred={T1_pred:.1f})")

    plt.xlabel("Jump time")
    plt.ylabel("Probability density")
    plt.title("T1 Measurement Sanity Check")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
