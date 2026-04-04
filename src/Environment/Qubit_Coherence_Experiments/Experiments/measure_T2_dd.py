import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from qubit import Qubit
from qubit_gates import q_Rx, q_Ry
from noise import Noise


# --- DD Sequence ---
PI   = np.pi
CPMG = ['Y']
XY4  = ['X', 'Y', 'X', 'Y']
XY8  = ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X']
KDD  = ['X', 'Y', 'X', 'Y', 'X']

DD_SEQUENCE = CPMG

# --- Config ---
DT            = 1e-6
TEMPERATURE_K = 77.0
N_REPEATS     = 10   # needs to be >= 100 for stable coherence estimates
INCLUDE_T1    = True

# Fixed number of pi-pulses; sweep tau to build the decay curve.
# total_time = N_PULSES * tau  (x-axis).
N_PULSES = 8192

# tau range: total_time should span from well above 1/e down to below it.
# T2_DD ~ 400-600ms → tau = 5us..75us gives total_time 41ms..614ms
TAUS = np.linspace(5e-6, 75e-6, 20)

COHERENCE_THRESHOLD = np.exp(-1)


# ---------------------------------------------------------------------------

def q_equatorial() -> np.ndarray:
    return np.array([[0.5, 0.5],
                     [0.5, 0.5]], dtype=np.complex128)


def single_shot(n_pulses: int, half_steps: int) -> complex:
    """Run one complete DD sequence; return rho01 (off-diagonal element)."""
    qubit = Qubit(rho=q_equatorial(), dt=DT, temperature_Kelvin=TEMPERATURE_K)
    noise = Noise(dt=DT)

    for i in range(n_pulses):
        # tau/2 free evolution
        for _ in range(half_steps):
            if INCLUDE_T1:
                qubit.GAD()
            qubit.rho = noise.apply_noise(qubit.rho)

        # pi-pulse
        axis = DD_SEQUENCE[i % len(DD_SEQUENCE)]
        if axis == 'X':
            qubit.rho = q_Rx(PI, qubit.rho, noise)
        else:
            qubit.rho = q_Ry(PI, qubit.rho, noise)

        # tau/2 free evolution
        for _ in range(half_steps):
            if INCLUDE_T1:
                qubit.GAD()
            qubit.rho = noise.apply_noise(qubit.rho)

    return complex(qubit.rho[0, 1])


def coherence_at_tau(tau: float, n_pulses: int) -> tuple[float, float]:
    """
    Run N_REPEATS shots at the given tau.
    Average the complex rho01 FIRST, then compute coherence = 2|mean(rho01)|.
    Averaging before abs is essential — dephasing cancels opposite phases.
    Returns (total_time, coherence).
    """
    half_steps = max(1, int(round(tau / 2.0 / DT)))
    actual_tau = 2 * half_steps * DT

    rho01_values = np.array(
        [single_shot(n_pulses, half_steps) for _ in range(N_REPEATS)],
        dtype=np.complex128
    )

    coherence  = float(min(2.0 * abs(np.mean(rho01_values)), 1.0))
    total_time = n_pulses * actual_tau

    return total_time, coherence


# ---------------------------------------------------------------------------

def stretched_exponential(t: np.ndarray, A: float, T2: float, p: float) -> np.ndarray:
    return A * np.exp(-(t / T2) ** p)


def threshold_crossing(total_times: np.ndarray, coherences: np.ndarray) -> tuple[float, str]:
    """
    Find 1/e crossing of the raw coherence curve.
    Uses raw (non-monotone) coherences so a single noisy dip does not
    permanently anchor the threshold at an artificially early time.
    Returns (T2_estimate, status).  status ∈ {'bracketed', 'above_window', 'below_window'}.
    """
    cross = np.where(coherences <= COHERENCE_THRESHOLD)[0]

    if len(cross) == 0:
        return np.inf, "above_window"
    i = cross[0]
    if i == 0:
        return float(total_times[0]), "below_window"

    t0, t1 = total_times[i - 1], total_times[i]
    c0, c1 = coherences[i - 1], coherences[i]
    if c1 == c0:
        return float(t1), "bracketed"

    T2 = t0 + (COHERENCE_THRESHOLD - c0) * (t1 - t0) / (c1 - c0)
    return float(T2), "bracketed"


def fit_decay(total_times: np.ndarray, coherences: np.ndarray) -> tuple[float, float, float, np.ndarray, str]:
    """
    Fit C(t) = A·exp(-(t/T2)^p) to the full decay curve.
    Returns (T2_fit, A_fit, p_fit, fitted_curve, status).
    """
    if len(total_times) < 5:
        return np.nan, np.nan, np.nan, np.array([]), "too few points"

    # Initial guess for T2: monotone envelope 1/e crossing
    T2_guess, _ = threshold_crossing(total_times, coherences)
    if not np.isfinite(T2_guess):
        T2_guess = float(total_times[-1]) * 0.5

    A0 = float(coherences[0]) if coherences[0] > 0 else 1.0

    try:
        popt, _ = curve_fit(
            stretched_exponential,
            total_times,
            coherences,
            p0=[A0, T2_guess, 1.0],
            bounds=([0.0, total_times[0] * 0.1, 0.25],
                    [1.5, total_times[-1] * 10.0, 5.0]),
            maxfev=20000,
        )
        A_fit, T2_fit, p_fit = popt
        fitted_curve = stretched_exponential(total_times, *popt)
        return float(T2_fit), float(A_fit), float(p_fit), fitted_curve, "ok"

    except Exception as e:
        return np.nan, np.nan, np.nan, np.array([]), f"failed: {e}"


# ---------------------------------------------------------------------------

def measure_T2_dd(n_pulses: int, tau_values: np.ndarray):
    """
    Sweep tau values for a fixed number of pi-pulses.
    x-axis: total_time = n_pulses * tau
    y-axis: coherence = 2|mean(rho01)|   (mean taken over repeats before abs)
    """
    results = Parallel(n_jobs=-1)(
        delayed(coherence_at_tau)(tau, n_pulses) for tau in tau_values
    )

    total_times = np.array([r[0] for r in results])
    coherences  = np.array([r[1] for r in results])

    order = np.argsort(total_times)
    total_times = total_times[order]
    coherences  = coherences[order]

    envelope = np.minimum.accumulate(coherences)

    T2_threshold, threshold_status      = threshold_crossing(total_times, coherences)
    T2_fit, A_fit, p_fit, fitted_curve, fit_status = fit_decay(total_times, coherences)

    return (
        total_times,
        coherences,
        envelope,
        T2_threshold,
        threshold_status,
        T2_fit,
        A_fit,
        p_fit,
        fitted_curve,
        fit_status,
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seq_name = {id(CPMG): "CPMG", id(XY4): "XY4",
                id(XY8): "XY8",  id(KDD): "KDD"}.get(id(DD_SEQUENCE), "DD")

    print(f"\n--- T2 DD  |  {seq_name}  |  N = {N_PULSES} pulses  |  {TEMPERATURE_K} K ---")

    (
        total_times,
        coherences,
        envelope,
        T2_threshold,
        threshold_status,
        T2_fit,
        A_fit,
        p_fit,
        fitted_curve,
        fit_status,
    ) = measure_T2_dd(n_pulses=N_PULSES, tau_values=TAUS)

    taus_from_total = total_times / N_PULSES

    for tau, t, c, e in zip(taus_from_total, total_times, coherences, envelope):
        print(
            f"  tau={tau*1e6:7.1f} µs   "
            f"total={t*1e3:7.3f} ms   "
            f"coherence={c:.4f}   "
            f"envelope={e:.4f}"
        )

    print("\n--- Results ---")

    if threshold_status == "bracketed":
        print(f"  Threshold T2_DD  = {T2_threshold*1e3:.3f} ms  (1/e of raw coherence curve)")
    elif threshold_status == "below_window":
        print(f"  Threshold T2_DD  < {total_times[0]*1e3:.3f} ms  (decay faster than first point)")
    else:
        print(f"  Threshold T2_DD  > {total_times[-1]*1e3:.3f} ms  (no 1/e crossing in window)")

    fit_reliable = np.isfinite(T2_fit) and 0.28 < p_fit < 4.9
    if np.isfinite(T2_fit):
        reliability = "" if fit_reliable else "  [UNRELIABLE: p at bound]"
        print(f"  Fit      T2_DD  = {T2_fit*1e3:.3f} ms   (stretched-exp, p={p_fit:.3f}, A={A_fit:.3f}){reliability}")
        print(f"  Fit status: {fit_status}")
    else:
        print(f"  Fit      T2_DD  = unavailable  ({fit_status})")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    t_ms = total_times * 1e3

    ax.plot(t_ms, coherences, 'o-', linewidth=1.5, markersize=4, label="Coherence  2|⟨ρ₀₁⟩|")
    ax.plot(t_ms, envelope,   '--',  linewidth=1.5, label="Monotone envelope")

    if len(fitted_curve) == len(total_times):
        ax.plot(t_ms, fitted_curve, linewidth=2.0, label="Stretched-exp fit")

    ax.axhline(COHERENCE_THRESHOLD, color='k', linestyle=':', alpha=0.5, label="1/e")

    ax.set_xlabel("Total evolution time  n·τ  (ms)")
    ax.set_ylabel("Coherence  2|ρ₀₁|")
    ax.set_title(f"DD coherence decay  {seq_name}  N={N_PULSES}  @{TEMPERATURE_K} K")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    lines = []
    if threshold_status == "bracketed":
        lines.append(f"Threshold T2 = {T2_threshold*1e3:.2f} ms")
    elif threshold_status == "below_window":
        lines.append(f"Threshold T2 < {total_times[0]*1e3:.2f} ms")
    else:
        lines.append(f"Threshold T2 > {total_times[-1]*1e3:.2f} ms")

    if np.isfinite(T2_fit):
        unreliable_flag = "" if fit_reliable else " [UNRELIABLE]"
        lines.append(f"Fit T2 = {T2_fit*1e3:.2f} ms{unreliable_flag}")
        lines.append(f"Fit p  = {p_fit:.3f}")
    else:
        lines.append("Fit T2 = unavailable")

    ax.text(0.98, 0.95, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.tight_layout()
    plt.show()

    # --- Plain-language summary ---
    print("\n" + "="*55)
    print("  FINAL ANSWER")
    print("="*55)

    best_idx = int(np.argmax(coherences))
    best_tau_us = total_times[best_idx] / N_PULSES * 1e6
    best_total_ms = total_times[best_idx] * 1e3
    best_coherence = coherences[best_idx]
    print(f"  Best measured coherence point:")
    print(f"    tau = {best_tau_us:.1f} us  (total evolution = {best_total_ms:.1f} ms)")
    print(f"    coherence = {best_coherence:.3f}")
    print(f"    (coherence 1.0 = perfectly quantum, 0.0 = fully destroyed)")
    if N_REPEATS < 50:
        print(f"  NOTE: N_REPEATS={N_REPEATS} is low; estimates are noisy. Use >=50 for reliable T2.")
    print()

    if fit_reliable:
        print(f"  T2_DD (qubit coherence time with DD) = {T2_fit*1e3:.0f} ms")
        print(f"    This is how long the qubit stays quantum under the")
        print(f"    {seq_name} pulse sequence with N={N_PULSES} pulses.")
        print(f"    At this time the qubit has lost ~63% of its coherence.")
    elif np.isfinite(T2_threshold):
        print(f"  T2_DD (from threshold, fit unreliable) = {T2_threshold*1e3:.0f} ms")
    else:
        print("  T2_DD could not be extracted -- check tau range or increase N_REPEATS.")

    print("="*55)
