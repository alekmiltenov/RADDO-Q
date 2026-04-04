import numpy as np

class Noise:
    dt                              : float                             # Time step in simulation
    rng                             : np.random.Generator

    ou_slow_tau_c                   : float
    ou_slow_delta_omega_rms         : float
    ou_slow_delta_omega             : float

    ou_fast_tau_c                   : float
    ou_fast_delta_omega_rms         : float
    ou_fast_delta_omega             : float

    qs_delta_omega                  : float                             # Quasi-static detuning rad/s

    white_gamma_phi                 : float                             # White noise strangth

    rtn_lambda                      : float                             # Switching rate
    rtn_nu                          : float                             # Amplitude (switch power)
    rtn_sign                        : int                               # Last sign of rtn switch ( + or - )

    one_over_f_lambdas              : np.ndarray                        # Switching rates for RTN bank
    one_over_f_nus                  : np.ndarray                        # Amplitudes for RTN bank
    one_over_f_signs                : np.ndarray                        # Sign states for RTN bank
    one_over_f_P_switches           : np.ndarray                        # Per-step switch probabilities for RTN bank

    tech_t_pi                       : float                             # Nominal pulse duration
    tech_static_detuning            : float                             # Static detuning for one run
    tech_sigma_amp_frac             : float                             # Per-pulse fractional amplitude error std
    tech_sigma_time_frac            : float                             # Per-pulse fractional pulse duration jitter std
    tech_sigma_phase_rad            : float                             # Per-pulse phase error std

    # Base values
    OU_SLOW_TAU_C_BASE               = 1e-2
    OU_SLOW_DELTA_OMEGA_RMS_BASE     = 8e1
    OU_FAST_TAU_C_BASE               = 1e-4
    OU_FAST_DELTA_OMEGA_RMS_BASE     = 5e1   
    RTN_SWITCHING_RATE_BASE          = 500.0
    RTN_NU_BASE                      = 1.4e2   
    QS_DELTA_OMEGA_BASE              = 4e4
    WHITE_GAMMA_PHI_BASE             = 0.8
    ONE_OVER_F_NUM_FLUCTUATORS       = 8
    ONE_OVER_F_LAMBDA_MIN            = 1e-1
    ONE_OVER_F_LAMBDA_MAX            = 1e2
    ONE_OVER_F_TOTAL_NU_BASE         = 8e1

    TECH_T_PI_BASE                   = 100e-9
    TECH_STATIC_DETUNING_SIGMA_BASE  = 2.0 * np.pi * 2.0e3
    TECH_AMP_FRAC_SIGMA_BASE         = 1.2e-3
    TECH_TIME_FRAC_SIGMA_BASE        = 1e-3
    TECH_PHASE_SIGMA_BASE            = 2e-4

    PAULI_X = np.array([[0.0, 1.0],
                        [1.0, 0.0]], dtype=np.complex128)
    PAULI_Y = np.array([[0.0, -1j],
                        [1j, 0.0]], dtype=np.complex128)
    PAULI_Z = np.array([[1.0, 0.0],
                        [0.0, -1.0]], dtype=np.complex128)
    IDENTITY = np.eye(2, dtype=np.complex128)

    # Avoid Unnecessary Compute helpers
    ou_slow_decay_factor      : float
    ou_slow_random_term_scale : float
    ou_fast_decay_factor      : float
    ou_fast_random_term_scale : float
    rtn_P_switch              : float                                   # 0.5 * (1 - exp(- 2.0 * rtn_lambda * dt))


    def __init__(self, dt: float = 1e-5):
        self.dt = dt
        self.rng = np.random.default_rng()

        self.ou_slow_tau_c           = self.OU_SLOW_TAU_C_BASE * self.rng.uniform(0.2, 5.0)
        self.ou_slow_delta_omega_rms = self.OU_SLOW_DELTA_OMEGA_RMS_BASE * self.rng.uniform(0.5, 2.0)
        self.ou_slow_delta_omega     = self.ou_slow_delta_omega_rms * self.rng.standard_normal()
        self.ou_fast_tau_c           = self.OU_FAST_TAU_C_BASE * self.rng.uniform(0.2, 5.0)
        self.ou_fast_delta_omega_rms = self.OU_FAST_DELTA_OMEGA_RMS_BASE * self.rng.uniform(0.5, 2.0)
        self.ou_fast_delta_omega     = self.ou_fast_delta_omega_rms * self.rng.standard_normal()
        self.rtn_lambda              = self.RTN_SWITCHING_RATE_BASE * self.rng.uniform(0.2, 2)
        self.rtn_nu                  = self.RTN_NU_BASE * self.rng.uniform(0.5, 2)
        self.qs_delta_omega          = self.QS_DELTA_OMEGA_BASE * self.rng.standard_normal()
        self.white_gamma_phi         = self.WHITE_GAMMA_PHI_BASE * self.rng.uniform(0.3, 3)
        self.one_over_f_lambdas = np.logspace(
            np.log10(self.ONE_OVER_F_LAMBDA_MIN),
            np.log10(self.ONE_OVER_F_LAMBDA_MAX),
            self.ONE_OVER_F_NUM_FLUCTUATORS
        )

        self.one_over_f_nus = np.full(
            self.ONE_OVER_F_NUM_FLUCTUATORS,
            self.ONE_OVER_F_TOTAL_NU_BASE / np.sqrt(self.ONE_OVER_F_NUM_FLUCTUATORS)
        )

        self.one_over_f_signs = np.where(
            self.rng.random(self.ONE_OVER_F_NUM_FLUCTUATORS) < 0.5,
            1,
            -1
        ).astype(int)

        self.one_over_f_P_switches = 0.5 * (
            1.0 - np.exp(-2.0 * self.one_over_f_lambdas * self.dt)
        )

        self.tech_t_pi            = self.TECH_T_PI_BASE
        self.tech_static_detuning = self.rng.normal(0.0, self.TECH_STATIC_DETUNING_SIGMA_BASE)
        self.tech_sigma_amp_frac  = self.TECH_AMP_FRAC_SIGMA_BASE
        self.tech_sigma_time_frac = self.TECH_TIME_FRAC_SIGMA_BASE
        self.tech_sigma_phase_rad = self.TECH_PHASE_SIGMA_BASE

        self.rtn_sign                  = 1 if self.rng.random() < 0.5 else -1
        self.ou_slow_decay_factor      = np.exp(-self.dt / self.ou_slow_tau_c)
        self.ou_slow_random_term_scale = self.ou_slow_delta_omega_rms * np.sqrt(1.0 - self.ou_slow_decay_factor**2)
        self.ou_fast_decay_factor      = np.exp(-self.dt / self.ou_fast_tau_c)
        self.ou_fast_random_term_scale = self.ou_fast_delta_omega_rms * np.sqrt(1.0 - self.ou_fast_decay_factor**2)
        self.rtn_P_switch              = 0.5 * (1.0 - np.exp(-2.0 * self.rtn_lambda * self.dt))


    @staticmethod
    def Ornstein_Uhlenbeck(dt: float,
                           ou_delta_omega: float,
                           ou_decay_factor: float,
                           ou_random_term_scale: float,
                           rng: np.random.Generator
                           ) -> tuple[float, float]:

        phi = ou_delta_omega * dt

        xi = rng.standard_normal()
        delta_omega_next = (ou_delta_omega * ou_decay_factor + ou_random_term_scale * xi)

        return phi, float(delta_omega_next)
        
    @staticmethod
    def Random_Telegraph_Noise(dt: float,
                               rtn_P_switch: float,
                               rtn_nu: float,
                               rtn_sign: int,
                               rng: np.random.Generator
                               )-> tuple[float, int]:
        
        if rng.random() < rtn_P_switch:
            rtn_sign = -rtn_sign

        phi = rtn_sign * rtn_nu * dt
        return phi, int(rtn_sign)
    
    @staticmethod
    def Quasi_Static_Detuning(dt: float,
                              qs_delta_omega: float
                              ) -> float:
    
        phi = qs_delta_omega * dt
        return float(phi)


    @staticmethod
    def White_Noise(dt: float,
                    white_gamma_phi: float,
                    rng: np.random.Generator
                    ) -> float :
        
        xi = rng.standard_normal()
        phi = np.sqrt(2.0 * dt * white_gamma_phi) * xi
        return phi
    
    @staticmethod
    def One_Over_F_Pink_Noise(dt: float,
                            one_over_f_P_switches: np.ndarray,
                            one_over_f_nus: np.ndarray,
                            one_over_f_signs: np.ndarray,
                            rng: np.random.Generator
                            ) -> tuple[float, np.ndarray]:

        random_values = rng.random(len(one_over_f_signs))
        switch_mask = random_values < one_over_f_P_switches
        one_over_f_signs[switch_mask] *= -1

        phi = np.sum(one_over_f_signs * one_over_f_nus) * dt

        return float(phi), one_over_f_signs
    

    def apply_imperfect_pulse(self,
                          rho: np.ndarray,
                          axis: str,
                          angle: float = np.pi
                          ) -> np.ndarray:

        amp_frac_error  = self.rng.normal(0.0, self.tech_sigma_amp_frac)
        time_frac_error = self.rng.normal(0.0, self.tech_sigma_time_frac)
        phase_error     = self.rng.normal(0.0, self.tech_sigma_phase_rad)

        pulse_time    = self.tech_t_pi * (angle / np.pi) * (1.0 + time_frac_error)
        pulse_time    = max(pulse_time, 1e-15)
        omega_drive   = (np.pi / self.tech_t_pi) * (1.0 + amp_frac_error)

        if axis == "X":
            drive_phase = phase_error
        elif axis == "Y":
            drive_phase = (np.pi / 2.0) + phase_error
        else:
            raise ValueError("axis must be 'X' or 'Y'")

        omega_x = omega_drive * np.cos(drive_phase)
        omega_y = omega_drive * np.sin(drive_phase)
        omega_z = self.tech_static_detuning

        omega_eff = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        if omega_eff == 0.0:
            return rho

        theta     = omega_eff * pulse_time
        cos_h     = np.cos(theta / 2.0)
        sin_h     = np.sin(theta / 2.0)
        nx        = omega_x / omega_eff
        ny        = omega_y / omega_eff
        nz        = omega_z / omega_eff

        U = np.array([
            [cos_h - 1j * nz * sin_h,       (-1j * nx - ny) * sin_h],
            [(-1j * nx + ny) * sin_h,        cos_h + 1j * nz * sin_h]
        ], dtype=np.complex128)

        return U @ rho @ U.conj().T
    

    def apply_noise(self, rho: np.ndarray) -> np.ndarray:
        
        phi_ou_slow, self.ou_slow_delta_omega = self.Ornstein_Uhlenbeck(self.dt, self.ou_slow_delta_omega, self.ou_slow_decay_factor, self.ou_slow_random_term_scale, self.rng)
        phi_ou_fast, self.ou_fast_delta_omega = self.Ornstein_Uhlenbeck(self.dt, self.ou_fast_delta_omega, self.ou_fast_decay_factor, self.ou_fast_random_term_scale, self.rng)
        phi_rtn, self.rtn_sign = self.Random_Telegraph_Noise(self.dt, self.rtn_P_switch, self.rtn_nu, self.rtn_sign, self.rng)
        phi_qs = self.Quasi_Static_Detuning(self.dt, self.qs_delta_omega)
        phi_white = self.White_Noise(self.dt, self.white_gamma_phi, self.rng)
        phi_one_over_f, self.one_over_f_signs = self.One_Over_F_Pink_Noise(
            self.dt,
            self.one_over_f_P_switches,
            self.one_over_f_nus,
            self.one_over_f_signs,
            self.rng
        )

        phi = phi_ou_slow + phi_ou_fast + phi_rtn + phi_qs + phi_white + phi_one_over_f

        U = np.array([[np.exp(-1j * phi / 2.0), 0.0],
                      [0.0, np.exp(+1j * phi / 2.0)]], dtype=np.complex128)
        
        rho = U @ rho @ U.conj().T

        return rho