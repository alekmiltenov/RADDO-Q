import numpy as np
import random


class Qubit:
    # Qubit Properties
    rho: np.ndarray
    gamma1_0Kelvin: float
    gamma_phi: float
    T1_0Kelvin: float
    omega_01: float
    temperature_K: float

    # Qubit Measurements
    T1: float
    T2_star: float
    T2: float
    T2_eff: float
    T2_eff_agent: float


    def __init__(self, rho, gamma1_0Kelvin, gamma_phi, T1_0Kelvin=20.0, omega_01=0.0, temperature_K=0.0):
        self.rho = rho.astype(np.complex128)
        self.gamma1_0Kelvin = float(gamma1_0Kelvin)
        self.gamma_phi = float(gamma_phi)
        self.T1_0Kelvin = float(T1_0Kelvin)
        self.omega_01 = float(omega_01)
        self.temperature_K = float(temperature_K)


    # Predict T1
    @staticmethod
    def predict_T1(T1_true, measurement_relative_error=0.1):
        eps = measurement_relative_error * (2 * random.random() - 1)
        return T1_true * (1 + eps)


    @staticmethod
    def _nth(omega_01: float, temperature_K: float) -> float:
        if omega_01 <= 0.0 or temperature_K <= 0.0:
            return 0.0
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        x = (hbar * omega_01) / (kB * temperature_K)
        x = min(max(float(x), 1e-12), 700.0)
        return float(1.0 / (np.exp(x) - 1.0))


    @staticmethod
    def _gamma0(gamma1_0Kelvin: float, T1_0Kelvin: float) -> float:
        if gamma1_0Kelvin > 0.0:
            return float(gamma1_0Kelvin)
        if T1_0Kelvin > 0.0:
            return float(1.0 / T1_0Kelvin)
        return 0.0


    # Probability of relaxation
    @staticmethod
    def relaxation_probability(
        rho: np.ndarray,
        gamma1_0Kelvin: float,
        T1_0Kelvin: float,
        omega_01: float,
        temperature_K: float,
        dt: float
    ) -> float:
        if dt <= 0.0:
            return 0.0

        rho11 = float(np.real(rho[1, 1]))
        rho11 = min(max(rho11, 0.0), 1.0)

        gamma0 = Qubit._gamma0(gamma1_0Kelvin, T1_0Kelvin)
        n_th = Qubit._nth(omega_01, temperature_K)

        gamma_down = gamma0 * (1.0 + n_th)
        p_step = 1.0 - np.exp(-gamma_down * dt)

        return float(p_step * rho11)


    # Probability of excitation
    @staticmethod
    def excitation_probability(
        rho: np.ndarray,
        gamma1_0Kelvin: float,
        T1_0Kelvin: float,
        omega_01: float,
        temperature_K: float,
        dt: float
    ) -> float:
        if dt <= 0.0:
            return 0.0

        rho00 = float(np.real(rho[0, 0]))
        rho00 = min(max(rho00, 0.0), 1.0)

        gamma0 = Qubit._gamma0(gamma1_0Kelvin, T1_0Kelvin)
        n_th = Qubit._nth(omega_01, temperature_K)

        gamma_up = gamma0 * n_th
        p_step = 1.0 - np.exp(-gamma_up * dt)

        return float(p_step * rho00)


    # Decide whether to apply relaxation
    @staticmethod
    def relaxation_step(
        rho: np.ndarray,
        gamma1_0Kelvin: float,
        T1_0Kelvin: float,
        omega_01: float,
        temperature_K: float,
        dt: float,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, bool, float]:
        p_jump = Qubit.relaxation_probability(rho, gamma1_0Kelvin, T1_0Kelvin, omega_01, temperature_K, dt)

        if p_jump > 0.0 and float(rng.random()) < p_jump:
            rho_next = Qubit.apply_relaxation_jump(rho, p_jump)
            return rho_next, True, p_jump

        rho_next = Qubit.apply_relaxation_nojump(rho, p_jump)
        return rho_next, False, p_jump


    # Decide whether to apply excitation
    @staticmethod
    def excitation_step(
        rho: np.ndarray,
        gamma1_0Kelvin: float,
        T1_0Kelvin: float,
        omega_01: float,
        temperature_K: float,
        dt: float,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, bool, float]:
        p_jump = Qubit.excitation_probability(rho, gamma1_0Kelvin, T1_0Kelvin, omega_01, temperature_K, dt)

        if p_jump > 0.0 and float(rng.random()) < p_jump:
            rho_next = Qubit.apply_excitation_jump(rho, p_jump)
            return rho_next, True, p_jump

        rho_next = Qubit.apply_excitation_nojump(rho, p_jump)
        return rho_next, False, p_jump


    # Generalized Amplitude Damping
    @staticmethod
    def GAD_step(
        rho: np.ndarray,
        gamma1_0Kelvin: float,
        T1_0Kelvin: float,
        omega_01: float,
        temperature_K: float,
        dt: float,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, str, float, float]:
        if dt <= 0.0:
            return rho, "none", 0.0, 0.0

        rho11 = float(np.real(rho[1, 1]))
        rho00 = float(np.real(rho[0, 0]))
        rho11 = min(max(rho11, 0.0), 1.0)
        rho00 = min(max(rho00, 0.0), 1.0)

        gamma0 = Qubit._gamma0(gamma1_0Kelvin, T1_0Kelvin)
        n_th = Qubit._nth(omega_01, temperature_K)

        gamma_down = gamma0 * (1.0 + n_th)
        gamma_up = gamma0 * n_th

        p_down = 1.0 - np.exp(-gamma_down * dt)
        p_up = 1.0 - np.exp(-gamma_up * dt)

        P_down = float(p_down * rho11)
        P_up = float(p_up * rho00)

        P_down = min(max(P_down, 0.0), 1.0)
        P_up = min(max(P_up, 0.0), 1.0)

        u = float(rng.random())

        if u < P_down and P_down > 0.0:
            rho_next = Qubit.apply_relaxation_jump(rho, p_down)
            return rho_next, "down", P_down, P_up

        if u < (P_down + P_up) and P_up > 0.0:
            rho_next = Qubit.apply_excitation_jump(rho, p_up)
            return rho_next, "up", P_down, P_up

        rho_next = Qubit.apply_GAD_nojump(rho, p_down, p_up)
        return rho_next, "none", P_down, P_up


    # Relaxation jump
    @staticmethod
    def apply_relaxation_jump(rho: np.ndarray, p_down: float) -> np.ndarray:
        p_down = min(max(float(p_down), 0.0), 1.0)

        K_down = np.array([[0.0, np.sqrt(p_down)],
                           [0.0, 0.0]], dtype=np.complex128)

        rho_jump = K_down @ rho @ K_down.conj().T
        tr = float(np.real(np.trace(rho_jump)))
        if tr > 0.0:
            rho_jump /= tr
        else:
            rho_jump = np.array([[1.0, 0.0],
                                 [0.0, 0.0]], dtype=np.complex128)
        return rho_jump


    # Exitation jump
    @staticmethod
    def apply_excitation_jump(rho: np.ndarray, p_up: float) -> np.ndarray:
        p_up = min(max(float(p_up), 0.0), 1.0)

        K_up = np.array([[0.0, 0.0],
                         [np.sqrt(p_up), 0.0]], dtype=np.complex128)

        rho_jump = K_up @ rho @ K_up.conj().T
        tr = float(np.real(np.trace(rho_jump)))
        if tr > 0.0:
            rho_jump /= tr
        else:
            rho_jump = np.array([[0.0, 0.0],
                                 [0.0, 1.0]], dtype=np.complex128)
        return rho_jump


    # Relaxation no jump
    @staticmethod
    def apply_relaxation_nojump(rho: np.ndarray, p_down: float) -> np.ndarray:
        p_down = min(max(float(p_down), 0.0), 1.0)

        K0 = np.array([[1.0, 0.0],
                       [0.0, np.sqrt(1.0 - p_down)]], dtype=np.complex128)

        rho_nojump = K0 @ rho @ K0.conj().T
        tr = float(np.real(np.trace(rho_nojump)))
        if tr > 0.0:
            rho_nojump /= tr
        return rho_nojump


    # Exitation no jump
    @staticmethod
    def apply_excitation_nojump(rho: np.ndarray, p_up: float) -> np.ndarray:
        p_up = min(max(float(p_up), 0.0), 1.0)

        K0 = np.array([[np.sqrt(1.0 - p_up), 0.0],
                       [0.0, 1.0]], dtype=np.complex128)

        rho_nojump = K0 @ rho @ K0.conj().T
        tr = float(np.real(np.trace(rho_nojump)))
        if tr > 0.0:
            rho_nojump /= tr
        return rho_nojump


    # GAD no jump
    @staticmethod
    def apply_GAD_nojump(rho: np.ndarray, p_down: float, p_up: float) -> np.ndarray:
        p_down = min(max(float(p_down), 0.0), 1.0)
        p_up = min(max(float(p_up), 0.0), 1.0)

        K0 = np.array([[np.sqrt(1.0 - p_up), 0.0],
                       [0.0, np.sqrt(1.0 - p_down)]], dtype=np.complex128)

        rho_nojump = K0 @ rho @ K0.conj().T
        tr = float(np.real(np.trace(rho_nojump)))
        if tr > 0.0:
            rho_nojump /= tr
        return rho_nojump
    
    # T phi phase damp
    @staticmethod
    def phase_damp_step(
        rho: np.ndarray,
        T_phi: float,
        dt: float
    ) -> np.ndarray:
        if T_phi <= 0.0 or dt <= 0.0:
            return rho

        deph_factor = np.exp(-dt / T_phi)

        rho_next = rho.copy()
        rho_next[0, 1] *= deph_factor
        rho_next[1, 0] *= deph_factor

        return rho_next
