import numpy as np
import random


class Qubit:
    # Qubit Properties
    rho: np.ndarray
    gamma1_0Kelvin: float
    gamma_phi: float

    # Qubit Measurements
    T1: float
    T2_star: float
    T2: float
    T2_eff: float
    T2_eff_agent: float


    def __init__(self, rho, gamma1, gamma_phi):
        self.rho = rho.astype(np.complex128)
        self.gamma1 = float(gamma1)
        self.gamma_phi = float(gamma_phi)


    # Predict T1
    @staticmethod
    def predict_T1(T1_true, measurement_relative_error=0.1):
        eps = measurement_relative_error * (2 * random.random() - 1)
        return T1_true * (1 + eps)
    

    # Calculate Probability of damp each step based on T1 , Rho, Step length
    @staticmethod
    def amplitude_damping_probability(rho: np.ndarray, T1: float, dt: float) -> float:
        if T1 <= 0 or dt <= 0:
            return 0.0
        p_step = 1.0 - np.exp(-dt / T1)
        rho11 = float(np.real(rho[1, 1]))
        rho11 = min(max(rho11, 0.0), 1.0)
        return float(p_step * rho11)
    

    # Decide whether to apply the damp based on Probability & Rho
    @staticmethod
    def amplitude_damp_step(
        rho: np.ndarray,
        T1: float,
        dt: float,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, bool, float]:
        p_jump = Qubit.amplitude_damping_probability(rho, T1, dt)

        if p_jump > 0.0 and float(rng.random()) < p_jump:
            rho_next = Qubit.apply_amplitude_damp_jump(rho, T1, dt)
            return rho_next, True, p_jump

        rho_next = Qubit.apply_amplitude_damp_nojump(rho, T1, dt)
        return rho_next, False, p_jump


    # Qubit state if jump occurs
    @staticmethod
    def apply_amplitude_damp_jump(rho: np.ndarray, T1: float, dt: float) -> np.ndarray:
        p_step = 1.0 - np.exp(-dt / T1)
        K1 = np.array([[0.0, np.sqrt(p_step)],
                       [0.0, 0.0]], dtype=np.complex128)

        rho_jump = K1 @ rho @ K1.conj().T
        tr = float(np.real(np.trace(rho_jump)))
        if tr > 0.0:
            rho_jump /= tr
        else:
            rho_jump = np.array([[1.0, 0.0],
                                 [0.0, 0.0]], dtype=np.complex128)
        return rho_jump


    # Qubit state if no jump occurs
    @staticmethod
    def apply_amplitude_damp_nojump(rho: np.ndarray, T1: float, dt: float) -> np.ndarray:
        p_step = 1.0 - np.exp(-dt / T1)
        K0 = np.array([[1.0, 0.0],
                       [0.0, np.sqrt(1.0 - p_step)]], dtype=np.complex128)

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