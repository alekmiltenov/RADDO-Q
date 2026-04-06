import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from qubit import Qubit
from qubit_gates import q_Rx, q_Ry
from noise import Noise

DT = 1e-5
# --- DD Sequence ---
PI   = np.pi
CPMG = [('Y', 1)]
XY4  = [('X', 1), ('Y', 1), ('X', 1), ('Y', 1)]
XY8  = [('X', 1), ('Y', 1), ('X', 1), ('Y', 1), ('Y', 1), ('X', 1), ('Y', 1), ('X', 1)]

DD_SEQUENCE = CPMG

# --- Config ---
TEMPERATURE_K = 77.0
N_REPEATS     = 1


TAUS = [20e-6 , 30e-6 , 40e-6, 50e-6 , 60e-6 , 70e-6]

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
        qubit.GAD()
        qubit.rho = noise.apply_noise(qubit.rho)

    # Execute sequence for total of N pulses
    for i in range(n_pulses):
        
        # PI pulse
        axis, angle = dd_sequence[i % len(dd_sequence)]
        if axis == 'X': qubit.rho = q_Rx((np.pi)/angle, qubit.rho)
        elif axis == 'Y': qubit.rho = q_Ry((np.pi)/angle, qubit.rho)
        
        

        # Tau/2 if last pulse
        if i == (n_pulses-1):
            for _ in range(int(dt_in_tau/2)):
                qubit.GAD()
                qubit.rho = noise.apply_noise(qubit.rho)
        else:
            # Free evolution in tau
            for _ in range(dt_in_tau):
                qubit.GAD()
                qubit.rho = noise.apply_noise(qubit.rho)

        if (i+1) % (10 * len(dd_sequence)) == 0:
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

def threshold_crossing_interpolated(times, coherences, threshold=1/np.e):
    times = np.array(times, dtype=float)
    coherences = np.array(coherences, dtype=float)

    if len(times) == 0:
        return np.nan

    if coherences[0] <= threshold:
        return times[0]

    for i in range(1, len(coherences)):
        c1 = coherences[i - 1]
        c2 = coherences[i]
        t1 = times[i - 1]
        t2 = times[i]

        if c1 > threshold and c2 <= threshold:
            if c2 == c1:
                return t2
            return t1 + (threshold - c1) * (t2 - t1) / (c2 - c1)

    return np.nan

def extract_T2_DD(times, coherences):
    times = np.array(times, dtype=float)
    coherences = np.array(coherences, dtype=float)

    if len(times) == 0:
        return np.nan, np.nan, np.nan

    if times[0] > 0:
        times = np.insert(times, 0, 0.0)
        coherences = np.insert(coherences, 0, 1.0)

    threshold = 1 / np.e

    # 1) raw threshold with interpolation
    t2_raw = threshold_crossing_interpolated(times, coherences, threshold)

    # 2) envelope threshold with interpolation
    coherences_env = np.minimum.accumulate(coherences)
    t2_env = threshold_crossing_interpolated(times, coherences_env, threshold)

    # 3) exponential fit
    try:
        p0 = [1.0, times[-1] / 2]
        bounds = ([0.0, 1e-12], [2.0, np.inf])
        popt, _ = curve_fit(exp_decay, times, coherences, p0=p0, bounds=bounds, maxfev=10000)
        t2_fit = popt[1]
    except Exception:
        t2_fit = np.nan

    return t2_raw, t2_env, t2_fit

def Sweep_Tau_T2_DD(taus, dd_sequence, n_jobs=-1, best_by="env", max_total_time=0.24):
    def one_tau(tau):
        dt_in_tau = max(1, int(round(tau / DT)))
        actual_tau = dt_in_tau * DT

        n_pulses_eff = int(max_total_time / actual_tau)

        # make divisible by sequence length
        n_pulses_eff += (-n_pulses_eff) % len(dd_sequence)

        n_pulses_eff = max(n_pulses_eff, 10 * len(dd_sequence))

        times, coherences = Average_DD(n_pulses_eff, tau, dd_sequence)
        t2_raw, t2_env, t2_fit = extract_T2_DD(times, coherences)

        return {
            "tau": tau,
            "actual_tau": actual_tau,
            "n_pulses_used": n_pulses_eff,
            "times": times,
            "coherences": coherences,
            "t2_raw": t2_raw,
            "t2_env": t2_env,
            "t2_fit": t2_fit
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(one_tau)(tau) for tau in taus
    )

    results.sort(key=lambda r: r["tau"])

    if best_by == "raw":
        key = "t2_raw"
    elif best_by == "fit":
        key = "t2_fit"
    else:
        key = "t2_env"

    valid_results = [r for r in results if not np.isnan(r[key])]

    if len(valid_results) == 0:
        best_tau = np.nan
        best_t2 = np.nan
    else:
        best_result = max(valid_results, key=lambda r: r[key])
        best_tau = best_result["tau"]
        best_t2 = best_result[key]

    return results, best_tau, best_t2


import time

def main():
    dd_sequence = CPMG
    taus = TAUS

    start = time.perf_counter()

    results, best_tau, best_t2 = Sweep_Tau_T2_DD(
        taus=taus,
        dd_sequence=dd_sequence,
        n_jobs=-1,
        best_by="env"
    )

    end = time.perf_counter()

    for r in results:
        print(f"tau = {r['tau']:.2e} s")
        print(f"T2_DD raw threshold   = {r['t2_raw']}")
        print(f"T2_DD envelope        = {r['t2_env']}")
        print(f"T2_DD exp fit         = {r['t2_fit']}")
        print()

    print(f"Best tau = {best_tau}")
    print(f"Best T2_DD = {best_t2}")
    print(f"Total runtime = {end - start:.3f} s")

    plt.figure(figsize=(10, 6))

    for r in results:
        plt.plot(r["times"], r["coherences"], label=f"tau = {r['tau'] * 1e6:.1f} us")

    plt.axhline(COHERENCE_THRESHOLD, linestyle='--', label='1/e')
    plt.xlabel("Time (s)")
    plt.ylabel("Coherence")
    plt.title("Coherence decay curves for different tau")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()