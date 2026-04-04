import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from qubit import Qubit
from qubit_gates import q_Rx, q_Ry
from noise import Noise

DT = 1e-6
# --- DD Sequence ---
PI   = np.pi
CPMG = [('Y', 1)]
XY4  = ['X', 'Y', 'X', 'Y']
XY8  = ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X']

DD_SEQUENCE = CPMG

# --- Config ---
TEMPERATURE_K = 77.0
N_REPEATS     = 10
INCLUDE_GAD    = True

N_PULSES = 16384
N_PULSES = N_PULSES + ((-N_PULSES) % len(DD_SEQUENCE))

TAUS = np.linspace(5e-6, 75e-6, 20)

COHERENCE_THRESHOLD = np.exp(-1)


def Dynamic_Decoupling(n_pulses: int, tau: float, dd_sequence):
    # Init Qubit & Noise
    qubit = Qubit(rho= np.array([[0.5, 0.5],
                                   [0.5, 0.5]], dtype=np.complex128),
                                   dt=DT,
                                   temperature_Kelvin=TEMPERATURE_K)
    noise = Noise(dt=DT)

    dt_in_tau = max(1, int(round(tau / DT)))
    actual_tau = dt_in_tau * DT
    
    sampled_times = []
    sampled_rho01 = []

    # Initial tau/2
    for _ in range(int(dt_in_tau/2)):
        if INCLUDE_GAD:
            qubit.GAD()
        qubit.rho = noise.apply_noise(qubit.rho)

    # Execute sequence for total of N pulses
    for i in range(n_pulses):
        
        # PI pulse
        axis, angle = dd_sequence[i % len(dd_sequence)]
        if axis == 'X': qubit.rho = q_Rx((np.pi)/angle, qubit.rho, noise)
        elif axis == 'Y': qubit.rho = q_Ry((np.pi)/angle, qubit.rho, noise)

        # Tau/2 if last pulse
        if i == (n_pulses-1):
            for _ in range(int(dt_in_tau/2)):
                if INCLUDE_GAD:
                    qubit.GAD()
                qubit.rho = noise.apply_noise(qubit.rho)
        else:
            # Free evolution in tau
            for _ in range(dt_in_tau):
                if INCLUDE_GAD:
                    qubit.GAD()
                qubit.rho = noise.apply_noise(qubit.rho)

        if (i+1) % (10 * len(DD_SEQUENCE)) == 0:
            current_time = ((actual_tau/2) + ((i+1)*actual_tau)) 
            sampled_times.append(current_time)
            sampled_rho01.append(complex(qubit.rho[0, 1]))
            
    return sampled_times , sampled_rho01


def Average_DD(n_pulses: int, tau: float, dd_sequence):
    times, rho01_samples = Dynamic_Decoupling(n_pulses, tau, dd_sequence)
    all_rho01_samples = [rho01_samples]

    for _ in range(N_REPEATS - 1):
        _, rho01_samples = Dynamic_Decoupling(n_pulses, tau, dd_sequence)
        all_rho01_samples.append(rho01_samples)

    all_rho01_samples = np.array(all_rho01_samples, dtype=np.complex128)

    mean_rho01 = np.mean(all_rho01_samples, axis=0)
    coherences = 2 * np.abs(mean_rho01)

    return times, coherences



def first_drop_below_threshold(times, coherences, threshold=1/np.e):
    for i in range(len(coherences)):
        if coherences[i] <= threshold:
            return times[i]
    return np.nan


def exp_decay(t, A, T2):
    return A * np.exp(-t / T2)


def extract_T2_DD(n_pulses: int, tau: float, dd_sequence):
    times, coherences = Average_DD(n_pulses, tau, dd_sequence)

    times = np.array(times, dtype=float)
    coherences = np.array(coherences, dtype=float)

    if len(times) == 0:
        return np.nan, np.nan, np.nan

    if times[0] > 0:
        times = np.insert(times, 0, 0.0)
        coherences = np.insert(coherences, 0, 1.0)

    threshold = 1 / np.e

    # 1) raw first drop
    t2_raw = first_drop_below_threshold(times, coherences, threshold)

    # 2) envelope first drop
    coherences_env = np.minimum.accumulate(coherences)
    t2_env = first_drop_below_threshold(times, coherences_env, threshold)

    # 3) exponential fit
    try:
        p0 = [1.0, times[-1] / 2]
        bounds = ([0.0, 1e-12], [2.0, np.inf])
        popt, _ = curve_fit(exp_decay, times, coherences, p0=p0, bounds=bounds, maxfev=10000)
        t2_fit = popt[1]
    except Exception:
        t2_fit = np.nan

    return t2_raw, t2_env, t2_fit

##################################################################################
import time

def main():
    n_pulses = 16384
    tau = 100e-6   # 20 us
    dd_sequence = [('Y', 1)]   # example XY

    start = time.perf_counter()

    t2_raw, t2_env, t2_fit = extract_T2_DD(
        n_pulses=n_pulses,
        tau=tau,
        dd_sequence=dd_sequence
    )

    end = time.perf_counter()

    print(f"tau = {tau:.2e} s")
    print(f"T2_DD raw threshold   = {t2_raw}")
    print(f"T2_DD envelope        = {t2_env}")
    print(f"T2_DD exp fit         = {t2_fit}")
    print(f"Runtime = {end - start:.3f} s")


if __name__ == "__main__":
    main()