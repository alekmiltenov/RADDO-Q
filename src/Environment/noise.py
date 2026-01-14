import numpy as np
from qubit import Qubit
from qubit_gates import q_Rx


def quasi_static_detuning_step(
    rho: np.ndarray,
    dt: float,
    detuning_sigma_hz: float,
    rng: np.random.Generator,
    detuning_rad_s: float | None = None
) -> tuple[np.ndarray, float]:
    if dt <= 0.0:
        return rho, 0.0 if detuning_rad_s is None else float(detuning_rad_s)

    if detuning_sigma_hz <= 0.0:
        detuning = 0.0 if detuning_rad_s is None else float(detuning_rad_s)
        return rho, detuning

    if detuning_rad_s is None:
        detuning_sigma_rad_s = float(2.0 * np.pi * detuning_sigma_hz)
        detuning = float(rng.normal(0.0, detuning_sigma_rad_s))
    else:
        detuning = float(detuning_rad_s)

    phi = detuning * dt

    e_neg = np.exp(-1j * (phi / 2.0))
    e_pos = np.exp(+1j * (phi / 2.0))

    U = np.array([[e_neg, 0.0],
                  [0.0,  e_pos]], dtype=np.complex128)

    rho_next = U @ rho @ U.conj().T
    return rho_next, detuning


def bloch_from_rho(rho: np.ndarray):
    x = 2.0 * np.real(rho[0, 1])
    y = -2.0 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])
    return float(x), float(y), float(z)


def log_step(rho: np.ndarray, t: float, detuning_rad_s: float):
    tr = float(np.real(np.trace(rho)))
    herm_err = float(np.linalg.norm(rho - rho.conj().T))
    x, y, z = bloch_from_rho(rho)

    print(f"t={t:6.3f}  Δ={detuning_rad_s:+.6f} rad/s  tr={tr:.6f}  herm_err={herm_err:.2e}  bloch=({x:+.4f},{y:+.4f},{z:+.4f})")
    print(f"rho01={rho[0,1].real:+.6f}{rho[0,1].imag:+.6f}j   |rho01|={abs(rho[0,1]):.6f}")
    print(rho)
    print("-" * 88)


def main():
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(7)

    dt = 0.01
    t_max = 0.20
    steps = int(round(t_max / dt))

    detuning_sigma_hz = 50.0

    rho = np.array([[0.5 + 0.0j, 0.5 + 0.0j],
                    [0.5 + 0.0j, 0.5 + 0.0j]], dtype=np.complex128)

    detuning = 1.0
    rho = q_Rx(np.pi/6, rho)
    print("=== Quasi-static detuning sanity check ===")
    print(f"dt={dt}, t_max={t_max}, detuning_sigma={detuning_sigma_hz} Hz")
    print("Expected: rho00/rho11 constant, |rho01| constant, (x,y) rotates, trace~1.\n")

    for n in range(steps + 1):
        t = n * dt
        detuning_val = 0.0 if detuning is None else float(detuning)
        log_step(rho, t, detuning_val)

        rho, detuning = quasi_static_detuning_step(
            rho,
            dt,
            detuning_sigma_hz,
            rng,
            detuning
        )


if __name__ == "__main__":
    main()
