import numpy as np

class Noise:
    dt                  : float                             # Time step in simulation
    rng                 : np.random.Generator
    tau_c               : float
    ou_delta_omega_rms  : float
    ou_delta_omega      : float                             # Last calculated ou_delta_omega

    qs_delta_omega      : float                             # Quasi-static detuning rad/s

    white_gamma_phi     : float                             # White noise strangth

    rtn_lambda          : float                             # Switching rate
    rtn_nu              : float                             # Amplitude (switch power)
    rtn_sign            : int                               # Last sign of rtn switch ( + or - )

    one_over_f_lambdas    : np.ndarray                      # Switching rates for RTN bank
    one_over_f_nus        : np.ndarray                      # Amplitudes for RTN bank
    one_over_f_signs      : np.ndarray                      # Sign states for RTN bank
    one_over_f_P_switches : np.ndarray                      # Per-step switch probabilities for RTN bank

    # Base values
    OU_TAU_C_BASE           = 1e-2
    OU_DELTA_OMEGA_RMS_BASE = 1.5e3
    RTN_SWITCHING_RATE_BASE = 500.0
    RTN_NU_BASE             = 2e3
    QS_DELTA_OMEGA_BASE     = 4e4
    WHITE_GAMMA_PHI_BASE    = 1.0
    ONE_OVER_F_NUM_FLUCTUATORS = 8
    ONE_OVER_F_LAMBDA_MIN      = 1.0
    ONE_OVER_F_LAMBDA_MAX      = 1e5
    ONE_OVER_F_TOTAL_NU_BASE   = 6e4

    # Avoid Unnecessary Compute helpers
    ou_decay_factor      : float                            # exp(-dt/tau_c)
    ou_random_term_scale : float                            # ou_delta_omega_rms * sqrt(1 - decay_factor**2 )
    rtn_P_switch         : float                            # 0.5 * (1 - exp(- 2.0 * rtn_lambda * dt))


    def __init__(self, dt: float = 1e-5):
        self.dt = dt
        self.rng = np.random.default_rng()

        self.tau_c = self.OU_TAU_C_BASE * self.rng.uniform(0.2, 5.0)
        self.ou_delta_omega_rms = self.OU_DELTA_OMEGA_RMS_BASE * self.rng.uniform(0.5, 2.0)
        self.rtn_lambda = self.RTN_SWITCHING_RATE_BASE * self.rng.uniform(0.2, 2)
        self.rtn_nu = self.RTN_NU_BASE * self.rng.uniform(0.5, 2)
        self.ou_delta_omega = self.ou_delta_omega_rms * self.rng.standard_normal()
        self.qs_delta_omega = self.QS_DELTA_OMEGA_BASE * self.rng.standard_normal()
        self.white_gamma_phi = self.WHITE_GAMMA_PHI_BASE * self.rng.uniform(0.3, 3)
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

        self.rtn_sign = 1 if self.rng.random() < 0.5 else -1
        self.ou_decay_factor = np.exp(-self.dt / self.tau_c)
        self.ou_random_term_scale = self.ou_delta_omega_rms * np.sqrt(1.0 - (self.ou_decay_factor **2))
        self.rtn_P_switch = 0.5 * (1.0 - np.exp(-2.0 * self.rtn_lambda * self.dt))



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

        updated_signs = one_over_f_signs.copy()
        updated_signs[switch_mask] = -updated_signs[switch_mask]

        phi = np.sum(updated_signs * one_over_f_nus) * dt

        return float(phi), updated_signs
    

    def apply_noise(self, rho: np.ndarray) -> np.ndarray:
        
        phi_ou, self.ou_delta_omega = self.Ornstein_Uhlenbeck(self.dt, self.ou_delta_omega, self.ou_decay_factor, self.ou_random_term_scale, self.rng)
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

        phi = phi_ou + phi_rtn + phi_qs + phi_white + phi_one_over_f

        U = np.array([[np.exp(-1j * phi / 2.0), 0.0],
                      [0.0, np.exp(+1j * phi / 2.0)]], dtype=np.complex128)
        
        rho = U @ rho @ U.conj().T

        return rho
