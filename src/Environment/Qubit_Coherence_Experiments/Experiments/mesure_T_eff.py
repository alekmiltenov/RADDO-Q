"""
Phase 0  RAMSEY FID  (short-time sweep to extract T2*)
  Sweep total_steps from 1 to RAMSEY_MAX_STEPS (no pulses).
  Fits exponential decay -> T2_star.
  This is needed because with strong noise T2* can be < DT, making FID
  invisible in the N-sweep range.

Phase 1  TAU SWEEP  (fixed T_total, vary tau)
  Fix T_total = N_TAU_REF * 2 * DT.
  Vary steps_half 1..6  (tau = 2*DT .. 12*DT).
  N_pulses for each tau is chosen so T_total stays the same across all taus.
  Finds which tau gives the best CPMG coherence.

Phase 2  N SWEEP  at best tau from Phase 1
  Fix tau = best tau found above.
  Sweep N = [1, 2, 4, ... 8192]  (T_total = N * tau doubles each step).
  CPMG is measured and fitted.  FID T2* comes from Phase 0.
  Improvement is computed as T2_eff(CPMG) / T2*(FID).

-- Coherence metric --------------------------------------------------
  Average rho_01 (complex) across N_REPEATS THEN take abs.
  Taking abs first inflates the estimate -- complex average first is correct.

-- Statistical noise floor -------------------------------------------
  With pure dephasing, each trajectory keeps |rho_01| = 0.5 but randomises
  the phase.  For N_REPEATS repeats the estimator noise floor is:
      noise_floor ~ 0.5 * sqrt(pi / (4 * N_REPEATS))
  e.g.  N=200 -> 0.031,  N=400 -> 0.022,  N=1000 -> 0.014

-- Run-to-run variability warning ------------------------------------
  Noise parameters are randomised per repeat (each Noise() draws fresh
  uniforms).  tau_c spans 2-50 ms, delta_omega_rms spans 800-3200 rad/s.
  This gives T2*(OU) anywhere from ~50 us to ~1.25 ms -- a 25x swing.
  With N_REPEATS < 500 you will see noticeable run-to-run variation.
  Narrow the uniform() ranges in noise.py for tighter reproducibility.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qubit import Qubit
from qubit_gates import q_Rx, q_Ry, q_init
from noise import Noise


# ===================================================================
#  CONFIG
# ===================================================================

DT            = 1e-5    # Simulation time step (s)
TEMPERATURE_K = 77.0    # Environment temperature (K)

# Phase 1 -- tau sweep
N_REPEATS_TAU   = 150   # repeats per tau point  (noise floor ~ 0.036)
STEPS_HALF_LIST = [1, 2, 3, 4, 5, 6]   # steps per half-tau to sweep
                                         # -> tau = 2,4,6,8,10,12 × DT

# Phase 0 -- Ramsey FID (short-time T2* measurement)
# Uses a FINER time step (DT_RAMSEY) so we can resolve T2* even when it
# is shorter than the main simulation DT.
# Rule of thumb: DT_RAMSEY ~ T2* / 10  for ~10 resolved points before decay.
# With OU_DELTA_OMEGA_RMS ~ 4.8e5 rad/s -> T2* ~ 2 us -> DT_RAMSEY = 2e-7 s
DT_RAMSEY        = 2e-7   # finer dt for Ramsey only (s)
N_REPEATS_RAMSEY = 400    # more repeats -> lower noise floor -> better fit
RAMSEY_MAX_STEPS = 200    # sweep total_steps from 1 to this (each = DT_RAMSEY)
                           # covers up to 200 * DT_RAMSEY = 40 us at DT_RAMSEY=2e-7

# Phase 2 -- N sweep at best tau
N_REPEATS_N  = 200      # repeats per N point  (noise floor ~ 0.031)
N_VALUES     = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Reference total time for Phase 1 tau sweep.
# Rule of thumb: set N_TAU_REF so that T_TOTAL_REF ~ 1-3x your expected T2_eff(CPMG).
# T2_eff scales inversely with noise amplitude squared, so when you scale up noise by
# factor k, divide N_TAU_REF by k^2.
#
#   Weak noise  (base values x1,  T2_eff ~ 300ms)  -> N_TAU_REF = 8192
#   Strong noise (base values x300, T2_eff ~ 0.4ms) -> N_TAU_REF = 32
#
N_TAU_REF   = 32        # <-- adjust this when you change noise amplitude
T_TOTAL_REF = N_TAU_REF * 2 * DT

# ===================================================================


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------

def noise_floor(n_repeats: int) -> float:
    return 0.5 * np.sqrt(np.pi / (4.0 * n_repeats))


def _free_evolve(qubit: Qubit, noise_obj: Noise, n_steps: int) -> None:
    """Apply noise + GAD for n_steps simulation steps (in-place)."""
    for _ in range(n_steps):
        qubit.rho = noise_obj.apply_noise(qubit.rho)
        # qubit.GAD()


def _make_qubit_plus_x() -> tuple:
    """Return a fresh (qubit, noise) pair initialised to |+X>."""
    rho   = q_init()
    qubit = Qubit(rho=rho, dt=DT, temperature_Kelvin=TEMPERATURE_K)
    noise_obj = Noise(dt=DT)
    qubit.rho = q_Rx(np.pi / 2.0, qubit.rho)   # |0> -> |+X>
    return qubit, noise_obj


# -------------------------------------------------------------------
#  Simulation
# -------------------------------------------------------------------

def simulate_fid(total_steps: int, n_repeats: int) -> float:
    """
    Free Induction Decay -- no DD.
    Evolves for `total_steps` simulation steps with no pulses.
    Returns mean |rho_01|.
    """
    coherences = []
    for _ in range(n_repeats):
        qubit, noise_obj = _make_qubit_plus_x()
        _free_evolve(qubit, noise_obj, total_steps)
        coherences.append(qubit.rho[0, 1])
    return float(abs(np.mean(coherences)))


def simulate_cpmg(n_pulses: int, steps_half: int, n_repeats: int) -> float:
    """
    CPMG sequence.
    Each pulse period:  [steps_half free] -> Rx(pi) -> [steps_half free]
    Total steps         = n_pulses * 2 * steps_half
    Returns mean |rho_01|.
    """
    coherences = []
    for _ in range(n_repeats):
        qubit, noise_obj = _make_qubit_plus_x()
        for _ in range(n_pulses):
            _free_evolve(qubit, noise_obj, steps_half)   # tau/2
            qubit.rho = q_Ry(np.pi, qubit.rho)           # pi refocusing pulse
            _free_evolve(qubit, noise_obj, steps_half)   # tau/2
        coherences.append(qubit.rho[0, 1])
    return float(abs(np.mean(coherences)))


# -------------------------------------------------------------------
#  Fitting
# -------------------------------------------------------------------

def _exp_decay(T, A, T2):
    return A * np.exp(-T / T2)


def fit_T2eff(total_times_s: np.ndarray, coherences: np.ndarray,
              n_repeats: int) -> tuple[float, float]:
    """
    Fit  C(T) = A * exp(-T / T2_eff).
    Only fits points clearly above the statistical noise floor.
    Returns (T2_eff_s, A) or (nan, nan) on failure.
    """
    floor = noise_floor(n_repeats) + 0.01
    mask  = coherences > floor

    if mask.sum() < 3:
        return float("nan"), float("nan")

    T = total_times_s[mask]
    C = coherences[mask]

    # Log-linear initial guess
    try:
        slope, intercept = np.polyfit(T, np.log(C), 1)
        T2_g = max(-1.0 / slope, 1e-9) if slope < 0 else float(T[-1])
        A_g  = float(np.exp(intercept))
    except Exception:
        T2_g, A_g = float(T[-1]), float(C[0])

    try:
        popt, _ = curve_fit(
            _exp_decay, T, C,
            p0=[A_g, T2_g],
            bounds=([0.0, 1e-12], [2.0, 1e3]),
            maxfev=10000,
        )
        T2, A = float(popt[1]), float(popt[0])
        return (T2, A) if T2 > 0 else (float("nan"), float("nan"))
    except Exception:
        return (T2_g, A_g) if T2_g > 0 else (float("nan"), float("nan"))


# -------------------------------------------------------------------
#  Phase 0 -- Ramsey FID (short-time T2* measurement)
# -------------------------------------------------------------------

def _simulate_fid_ramsey(total_steps: int, n_repeats: int) -> float:
    """Like simulate_fid but uses DT_RAMSEY for finer time resolution."""
    coherences = []
    for _ in range(n_repeats):
        rho       = q_init()
        qubit     = Qubit(rho=rho, dt=DT_RAMSEY, temperature_Kelvin=TEMPERATURE_K)
        noise_obj = Noise(dt=DT_RAMSEY)
        qubit.rho = q_Rx(np.pi / 2.0, qubit.rho)
        for _ in range(total_steps):
            qubit.rho = noise_obj.apply_noise(qubit.rho)
        coherences.append(qubit.rho[0, 1])
    return float(abs(np.mean(coherences)))


def run_ramsey_fid() -> float:
    """
    Sweep total_steps from 1 to RAMSEY_MAX_STEPS using DT_RAMSEY.
    Fit exponential decay C(T) = A * exp(-T / T2_star).
    Returns T2_star in seconds, or nan on failure.
    """
    nf = noise_floor(N_REPEATS_RAMSEY)
    steps_arr = np.arange(1, RAMSEY_MAX_STEPS + 1)
    times_s   = steps_arr * DT_RAMSEY

    print("=" * 65)
    print("  PHASE 0 -- RAMSEY FID  (T2* measurement)")
    print("=" * 65)
    print(f"  DT_RAMSEY        = {DT_RAMSEY:.2e} s  (finer than main DT={DT:.2e})")
    print(f"  Sweeping total_steps 1..{RAMSEY_MAX_STEPS}  "
          f"(T = {times_s[0]*1e6:.2f} us .. {times_s[-1]*1e6:.2f} us)")
    print(f"  N_REPEATS_RAMSEY = {N_REPEATS_RAMSEY}   noise floor ~ {nf:.4f}")
    print("=" * 65)

    t0 = time.time()
    coh_vals = []
    for k, n_steps in enumerate(steps_arr):
        c = _simulate_fid_ramsey(int(n_steps), N_REPEATS_RAMSEY)
        coh_vals.append(c)
        if (k + 1) % 20 == 0 or k == 0 or k == len(steps_arr) - 1:
            print(f"  step {n_steps:3d}/{RAMSEY_MAX_STEPS}  "
                  f"T={times_s[k]*1e6:.2f} us  coherence={c:.4f}")

    coh_vals = np.array(coh_vals)
    elapsed  = time.time() - t0
    print(f"\n  Phase 0 elapsed: {elapsed:.1f}s")

    T2_star, A_star = fit_T2eff(times_s, coh_vals, N_REPEATS_RAMSEY)

    # -- Plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_s * 1e6, coh_vals, "o-", color="tab:red", ms=4, lw=1,
            label="FID (Ramsey)")
    ax.axhline(nf, color="gray", linestyle=":", label=f"noise floor ~ {nf:.3f}")

    if np.isfinite(T2_star) and np.isfinite(A_star):
        t_fine = np.linspace(times_s[0], times_s[-1], 500)
        fit_y  = _exp_decay(t_fine, A_star, T2_star)
        visible = fit_y >= nf * 0.5
        if visible.any():
            ax.plot(t_fine[visible] * 1e6, fit_y[visible], "--", color="tab:red",
                    alpha=0.8, label=f"fit  T2* = {T2_star*1e6:.2f} us")
        print(f"\n  T2* (Ramsey FID) = {T2_star*1e6:.3f} us  "
              f"({T2_star*1e3:.4f} ms)")
    else:
        print("\n  T2* fit failed -- FID may already be at noise floor at step 1.")
        print(f"  Try reducing DT_RAMSEY (currently {DT_RAMSEY:.2e} s) or")
        print("  increasing N_REPEATS_RAMSEY for a lower noise floor.")

    ax.set_xlabel(f"Total evolution time (us)  [DT_RAMSEY={DT_RAMSEY:.2e} s/step]")
    ax.set_ylabel("Coherence  |rho_01|")
    ax.set_title(f"Phase 0 -- Ramsey FID  |  T={TEMPERATURE_K} K  |  "
                 f"N_repeats={N_REPEATS_RAMSEY}")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p0_path = os.path.join(os.path.dirname(__file__), "CPMG_ramsey_fid.png")
    plt.savefig(p0_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {p0_path}")
    plt.close()

    return T2_star


# -------------------------------------------------------------------
#  Phase 1 -- Tau sweep
# -------------------------------------------------------------------

def run_tau_sweep() -> int:
    """
    For each steps_half in STEPS_HALF_LIST:
      - Fix T_total ~ T_TOTAL_REF by computing N_pulses accordingly
      - Run FID (same total steps, no pulses) and CPMG
    Returns best_steps_half (highest CPMG coherence).
    """
    total_steps_ref = round(T_TOTAL_REF / DT)   # e.g. 16384 steps

    fid_vals  = []
    cpmg_vals = []
    tau_us    = []

    nf = noise_floor(N_REPEATS_TAU)
    est_s = len(STEPS_HALF_LIST) * N_REPEATS_TAU * 2 * total_steps_ref / 1e5

    print("=" * 65)
    print("  PHASE 1 -- TAU SWEEP")
    print("=" * 65)
    print(f"  T_total fixed ~ {T_TOTAL_REF*1e3:.2f} ms  "
          f"({total_steps_ref} steps/repeat)")
    print(f"  steps_half    = {STEPS_HALF_LIST}  "
          f"(tau = {[sh*2 for sh in STEPS_HALF_LIST]} × DT)")
    print(f"  N_REPEATS_TAU = {N_REPEATS_TAU}   noise floor ~ {nf:.4f}")
    print(f"  Est. runtime  ~ {est_s/60:.1f} min")
    print("=" * 65)

    t0 = time.time()
    for i, sh in enumerate(STEPS_HALF_LIST):
        tau       = 2 * sh * DT
        n_pulses  = max(1, round(total_steps_ref / (2 * sh)))
        act_steps = n_pulses * 2 * sh           # actual total steps (may differ by ≤1)
        act_T     = act_steps * DT

        print(f"\n[{i+1}/{len(STEPS_HALF_LIST)}]  steps_half={sh}  "
              f"tau={tau*1e6:.1f} us  N_pulses={n_pulses}  "
              f"T_total={act_T*1e3:.3f} ms")

        fid_c  = simulate_fid(act_steps, N_REPEATS_TAU)
        cpmg_c = simulate_cpmg(n_pulses, sh, N_REPEATS_TAU)

        fid_vals.append(fid_c)
        cpmg_vals.append(cpmg_c)
        tau_us.append(tau * 1e6)

        flag = "  <- best so far" if cpmg_c == max(cpmg_vals) else ""
        print(f"         FID  = {fid_c:.6f}   CPMG = {cpmg_c:.6f}{flag}")

    elapsed = time.time() - t0
    print(f"\n  Phase 1 elapsed: {elapsed:.1f}s")

    best_idx        = int(np.argmax(cpmg_vals))
    best_steps_half = STEPS_HALF_LIST[best_idx]
    best_tau        = 2 * best_steps_half * DT

    print(f"\n  Best tau = {best_tau*1e6:.1f} us  "
          f"(steps_half={best_steps_half})  "
          f"CPMG coherence = {cpmg_vals[best_idx]:.6f}")

    # -- Phase 1 plot ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau_us, fid_vals,  "o-", color="tab:red",  label="FID (no DD)", markersize=7)
    ax.plot(tau_us, cpmg_vals, "s-", color="tab:blue", label="CPMG", markersize=7)
    ax.axhline(nf, color="gray", linestyle=":", label=f"noise floor ~ {nf:.3f}")
    ax.axvline(best_tau * 1e6, color="tab:blue", linestyle="--", alpha=0.5,
               label=f"best tau = {best_tau*1e6:.1f} us")
    ax.set_xlabel("Inter-pulse spacing  tau  (us)")
    ax.set_ylabel("Coherence  |rho_01|  at T_total ~ 163 ms")
    ax.set_title(f"Phase 1 -- Tau Sweep  |  T={TEMPERATURE_K} K  |  "
                 f"N_repeats={N_REPEATS_TAU}")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p1_path = os.path.join(os.path.dirname(__file__), "CPMG_tau_sweep.png")
    plt.savefig(p1_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {p1_path}")
    plt.close()

    return best_steps_half


# -------------------------------------------------------------------
#  Phase 2 -- N sweep at best tau
# -------------------------------------------------------------------

def run_n_sweep(best_steps_half: int, T2_star_fid: float = float("nan")) -> None:
    best_tau = 2 * best_steps_half * DT
    nf       = noise_floor(N_REPEATS_N)

    # Rough estimate: sum(2*sh*N) over all N, × 2 (FID+CPMG) × N_REPEATS
    step_sum = sum(2 * best_steps_half * n for n in N_VALUES)
    est_s    = step_sum * 2 * N_REPEATS_N / 1e5

    print("\n" + "=" * 65)
    print("  PHASE 2 -- N SWEEP")
    print("=" * 65)
    print(f"  Fixed tau     = {best_tau*1e6:.1f} us  (steps_half={best_steps_half})")
    print(f"  N_VALUES      = {N_VALUES}")
    print(f"  N_REPEATS_N   = {N_REPEATS_N}   noise floor ~ {nf:.4f}")
    print(f"  Est. runtime  ~ {est_s/60:.1f} min")
    print("=" * 65)

    fid_coh  = []
    cpmg_coh = []
    T_totals = []

    t0 = time.time()
    for i, n in enumerate(N_VALUES):
        total_steps = n * 2 * best_steps_half   # same for FID and CPMG
        T_total     = total_steps * DT
        T_totals.append(T_total)

        print(f"\n[{i+1:2d}/{len(N_VALUES)}]  N={n:5d}  "
              f"T_total={T_total*1e3:9.4f} ms  steps/repeat={total_steps:6d}")

        t1    = time.time()
        fid_c = simulate_fid(total_steps, N_REPEATS_N)
        fid_coh.append(fid_c)
        floor_tag = "  [noise floor]" if fid_c <= nf + 0.01 else ""
        print(f"         FID  coherence = {fid_c:.6f}   ({time.time()-t1:.1f}s){floor_tag}")

        t1     = time.time()
        cpmg_c = simulate_cpmg(n, best_steps_half, N_REPEATS_N)
        cpmg_coh.append(cpmg_c)
        floor_tag = "  [noise floor]" if cpmg_c <= nf + 0.01 else ""
        print(f"         CPMG coherence = {cpmg_c:.6f}   ({time.time()-t1:.1f}s){floor_tag}")

        ratio = (cpmg_c / fid_c) if fid_c > nf else float("inf")
        print(f"         Ratio CPMG/FID = {ratio:.2f}x")

    elapsed = time.time() - t0
    print(f"\n  Phase 2 elapsed: {elapsed:.1f}s")

    fid_coh  = np.array(fid_coh)
    cpmg_coh = np.array(cpmg_coh)
    T_totals = np.array(T_totals)

    # -- Fit T2_eff ------------------------------------------------
    # FID T2* comes from Phase 0 Ramsey (not from N-sweep, where FID is dead)
    T2_fid  = T2_star_fid   # may be nan if Phase 0 fit failed
    A_fid   = float("nan")  # no amplitude needed for summary; skip N-sweep FID fit

    T2_cpmg, A_cpmg = fit_T2eff(T_totals, cpmg_coh, N_REPEATS_N)

    improvement = float("nan")
    if np.isfinite(T2_fid) and np.isfinite(T2_cpmg) and T2_fid > 0:
        improvement = (T2_cpmg / T2_fid - 1.0) * 100.0

    # -- Summary ---------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Best tau (from Phase 1)      = {best_tau*1e6:.1f} us")
    print(f"  Statistical noise floor      ~ {nf:.4f}")
    print()
    if np.isfinite(T2_fid):
        print(f"  T2* without DD (Ramsey FID)  = {T2_fid*1e6:.3f} us  "
              f"({T2_fid*1e3:.4f} ms)")
    else:
        print("  T2* without DD (Ramsey FID)  = fit failed (see Phase 0)")
    if np.isfinite(T2_cpmg):
        print(f"  T2_eff  with CPMG-{N_VALUES[-1]}        = {T2_cpmg*1e3:.4f} ms")
    else:
        print(f"  T2_eff  with CPMG-{N_VALUES[-1]}        = fit failed")
    if np.isfinite(improvement):
        print(f"  Improvement  (T2_eff/T2*)    = {improvement/100.0 + 1:.1f}x  "
              f"({improvement:.1f} %)")
    print("=" * 65)

    # -- Phase 2 plot ----------------------------------------------
    T_ms = T_totals * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    t2_fid_str  = (f"{T2_fid*1e6:.2f} us" if np.isfinite(T2_fid)
                   else "N/A (see Phase 0)")
    t2_cpmg_str = (f"{T2_cpmg*1e3:.3f} ms" if np.isfinite(T2_cpmg) else "fit failed")
    fig.suptitle(
        f"CPMG N-sweep  |  tau={best_tau*1e6:.1f} us  "
        f"|  T={TEMPERATURE_K} K  |  N_repeats={N_REPEATS_N}  "
        f"|  T2*={t2_fid_str}  T2_eff={t2_cpmg_str}",
        fontsize=10,
    )

    T_fit = np.logspace(np.log10(T_ms[0]), np.log10(T_ms[-1]), 400)

    for ax, use_logy in zip(axes, [False, True]):
        ax.axhline(nf, color="gray", linestyle=":", linewidth=1,
                   label=f"noise floor ~ {nf:.3f}")

        # FID data in N-sweep (mostly at noise floor, shown for reference)
        ax.plot(T_ms, fid_coh, "o-", color="tab:red", lw=1, ms=4, alpha=0.5,
                label="FID (at noise floor)")

        # T2* vertical line from Phase 0 Ramsey
        if np.isfinite(T2_fid):
            ax.axvline(T2_fid * 1e3, color="tab:red", linestyle="--", alpha=0.6,
                       label=f"T2* = {T2_fid*1e6:.2f} us  (Ramsey)")

        ax.plot(T_ms, cpmg_coh, "s-", color="tab:blue", lw=1.5, ms=6,
                label=f"CPMG-{N_VALUES[-1]}")

        # CPMG fit line (clipped at noise floor)
        if np.isfinite(T2_cpmg) and np.isfinite(A_cpmg):
            fit_y = _exp_decay(T_fit * 1e-3, A_cpmg, T2_cpmg)
            visible = fit_y >= nf * 0.5
            if visible.any():
                ax.plot(T_fit[visible], fit_y[visible], "--",
                        color="tab:blue", alpha=0.8,
                        label=f"CPMG fit  T2_eff = {T2_cpmg*1e3:.3f} ms")

        ax.set_xscale("log")
        ax.set_xlabel("Total evolution  T = N x tau  (ms)")
        ax.set_ylabel("Coherence  |rho_01|")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)

        if use_logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(nf * 0.3, 1e-3))   # never below 1e-3
            ax.set_title("Log-Log  (fit quality)")
        else:
            ax.set_ylim(bottom=0)
            ax.set_title("Linear Y  (decay shape)")

    plt.tight_layout()
    p2_path = os.path.join(os.path.dirname(__file__),
                           f"CPMG_T2eff_n{N_VALUES[-1]}.png")
    plt.savefig(p2_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved -> {p2_path}")
    plt.close()


# -------------------------------------------------------------------
#  Entry point
# -------------------------------------------------------------------

def main():
    T2_star         = run_ramsey_fid()
    best_steps_half = run_tau_sweep()
    run_n_sweep(best_steps_half, T2_star_fid=T2_star)


if __name__ == "__main__":
    main()
